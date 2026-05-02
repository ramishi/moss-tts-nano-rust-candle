use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;
use serde::Deserialize;

use crate::modules::attention::MultiHeadAttention;
use crate::modules::lfq::{LFQ, ResidualLFQ, WNConv1d};
use crate::modules::pretransform::PatchedPretransform;
use crate::modules::projected::ProjectedTransformer;
use crate::modules::rotary::RotaryEmbedding;
use crate::modules::transformer::{LayerScale, MLP, TransformerLayer};

#[derive(Debug, Deserialize)]
pub struct DecoderKwargs {
    pub module_type: String,

    // For PatchedPretransform
    pub patch_size: Option<usize>,

    // For Transformer
    pub input_dimension: Option<usize>,
    pub output_dimension: Option<usize>,
    pub d_model: Option<usize>,
    pub num_layers: Option<usize>,
    pub num_heads: Option<usize>,
    pub dim_feedforward: Option<usize>,
    pub context_duration: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct QuantizerKwargs {
    pub input_dim: usize,
    pub rvq_dim: usize,
    pub output_dim: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub num_quantizers: usize,
}

#[derive(Debug, Deserialize)]
pub struct TokenizerConfig {
    pub encoder_kwargs: Vec<DecoderKwargs>,
    pub decoder_kwargs: Vec<DecoderKwargs>,
    pub quantizer_kwargs: QuantizerKwargs,
    pub number_channels: Option<usize>,
    pub enable_channel_interleave: Option<bool>,
    pub sampling_rate: usize,
    pub downsample_rate: usize,
}

pub enum DecoderLayer {
    Pretransform(PatchedPretransform),
    Transformer(ProjectedTransformer),
}

pub struct AudioTokenizerModel {
    pub quantizer: ResidualLFQ,
    pub encoder: Vec<DecoderLayer>,
    pub decoder: Vec<DecoderLayer>,
    pub rope: RotaryEmbedding, // Shared rope embedding for encoder and decoder
    pub number_channels: usize,
    pub enable_channel_interleave: bool,
}

/// Helper to build a list of DecoderLayer from kwargs, loading weights from a VarBuilder.
/// For PatchedPretransform: no weights, just reshape/permute.
/// For Transformer: loads input_proj, transformer layers, output_proj.
fn build_layers(
    vb: &VarBuilder,
    kwargs: &[DecoderKwargs],
    is_encoder: bool,
    frame_rate: f64,
    _rope: &RotaryEmbedding,
) -> Result<(Vec<DecoderLayer>, f64)> {
    let mut layers = Vec::new();
    let mut current_frame_rate = frame_rate;

    for (i, cfg) in kwargs.iter().enumerate() {
        let l_vb = vb.pp(i.to_string());
        if cfg.module_type == "PatchedPretransform" {
            // Encoder: downsample (patch_size), Decoder: upsample (patch_size)
            // For encoder: frame_rate /= patch_size (time dimension shrinks)
            // For decoder: frame_rate *= patch_size (time dimension grows)
            let p = PatchedPretransform::new(cfg.patch_size.unwrap_or(2), is_encoder);
            if is_encoder {
                current_frame_rate /= p.downsample_ratio() as f64;
            } else {
                current_frame_rate *= p.downsample_ratio() as f64;
            }
            layers.push(DecoderLayer::Pretransform(p));
        } else if cfg.module_type == "Transformer" {
            let d_model = cfg.d_model.unwrap_or(256);
            let num_heads = cfg.num_heads.unwrap_or(4);
            let head_dim = d_model / num_heads;
            let dim_ff = cfg.dim_feedforward.unwrap_or(1024);
            let context = cfg
                .context_duration
                .map(|c| (c * current_frame_rate).round() as usize);

            let in_dim = cfg.input_dimension.unwrap_or(d_model);
            let out_dim = cfg.output_dimension.unwrap_or(d_model);

            let tr_in_proj = candle_nn::linear_no_bias(in_dim, d_model, l_vb.pp("input_proj"))?;
            let tr_out_proj = candle_nn::linear_no_bias(d_model, out_dim, l_vb.pp("output_proj"))?;

            let mut trans_layers = Vec::new();
            let tvb = l_vb.pp("transformer.layers");

            for j in 0..cfg.num_layers.unwrap_or(0) {
                let jvb = tvb.pp(j.to_string());
                let attn_in =
                    candle_nn::linear_no_bias(d_model, d_model * 3, jvb.pp("self_attn.in_proj"))?;
                let attn_out =
                    candle_nn::linear_no_bias(d_model, d_model, jvb.pp("self_attn.out_proj"))?;
                let attn = MultiHeadAttention::new(attn_in, attn_out, num_heads, head_dim, context);

                let norm1 = candle_nn::layer_norm(d_model, 1e-5, jvb.pp("norm1"))?;
                let norm2 = candle_nn::layer_norm(d_model, 1e-5, jvb.pp("norm2"))?;

                let fc1 = candle_nn::linear_no_bias(d_model, dim_ff, jvb.pp("ffn.0"))?;
                let fc2 = candle_nn::linear_no_bias(dim_ff, d_model, jvb.pp("ffn.2"))?;
                let ffn = MLP::new(fc1, fc2);

                let ls1 = Some(LayerScale::new(jvb.get(d_model, "layer_scale_1.scale")?));
                let ls2 = Some(LayerScale::new(jvb.get(d_model, "layer_scale_2.scale")?));

                trans_layers.push(TransformerLayer {
                    attn,
                    norm1,
                    norm2,
                    ffn,
                    ls1,
                    ls2,
                });
            }

            let pt = ProjectedTransformer::new(tr_in_proj, trans_layers, tr_out_proj);
            current_frame_rate *= pt.downsample_ratio() as f64;
            layers.push(DecoderLayer::Transformer(pt));
        }
    }

    Ok((layers, current_frame_rate))
}

impl AudioTokenizerModel {
    pub fn load(vb: VarBuilder, config: &TokenizerConfig) -> Result<Self> {
        let device = vb.device();
        let rope = RotaryEmbedding::new(256 / 4, 131072, 10000.0, device)?;

        // --- Quantizer ---
        let q_kwargs = &config.quantizer_kwargs;
        let q_vb = vb.pp("quantizer");

        let q_in_proj = WNConv1d::new(
            q_vb.get(
                (q_kwargs.rvq_dim, q_kwargs.input_dim, 1),
                "input_proj.weight",
            )?,
            q_vb.get(q_kwargs.rvq_dim, "input_proj.bias").ok(),
        );
        let q_out_proj = WNConv1d::new(
            q_vb.get(
                (q_kwargs.output_dim, q_kwargs.rvq_dim, 1),
                "output_proj.weight",
            )?,
            q_vb.get(q_kwargs.output_dim, "output_proj.bias").ok(),
        );

        let mut lfqs = Vec::new();
        for i in 0..q_kwargs.num_quantizers {
            let vb_l = q_vb.pp(format!("quantizers.{}", i));
            let in_p = WNConv1d::new(
                vb_l.get(
                    (q_kwargs.codebook_dim, q_kwargs.rvq_dim, 1),
                    "in_proj.weight",
                )?,
                vb_l.get(q_kwargs.codebook_dim, "in_proj.bias").ok(),
            );
            let out_p = WNConv1d::new(
                vb_l.get(
                    (q_kwargs.rvq_dim, q_kwargs.codebook_dim, 1),
                    "out_proj.weight",
                )?,
                vb_l.get(q_kwargs.rvq_dim, "out_proj.bias").ok(),
            );
            let codebook = candle_nn::embedding(
                q_kwargs.codebook_size,
                q_kwargs.codebook_dim,
                vb_l.pp("codebook"),
            )?;

            lfqs.push(LFQ::new(
                Some(in_p),
                Some(out_p),
                codebook,
                q_kwargs.codebook_size,
                q_kwargs.codebook_dim,
                q_kwargs.rvq_dim,
            ));
        }

        let quantizer = ResidualLFQ::new(Some(q_in_proj), Some(q_out_proj), lfqs, q_kwargs.rvq_dim);

        // --- Encoder ---
        let enc_vb = vb.pp("encoder");
        // Base frame rate: samples/sec including channel interleave, matching Python
        let channel_interleave_factor = if config.enable_channel_interleave.unwrap_or(true)
            && config.number_channels.unwrap_or(1) > 1
        {
            config.number_channels.unwrap() as f64
        } else {
            1.0
        };
        let base_frame_rate = config.sampling_rate as f64 * channel_interleave_factor;
        let (encoder, _) = build_layers(
            &enc_vb,
            &config.encoder_kwargs,
            true,
            base_frame_rate,
            &rope,
        )?;

        // --- Decoder ---
        let dec_vb = vb.pp("decoder");
        // Decoder uses same base frame rate (samples/sec at output)
        let (decoder, _) = build_layers(
            &dec_vb,
            &config.decoder_kwargs,
            false,
            base_frame_rate,
            &rope,
        )?;

        Ok(Self {
            quantizer,
            encoder,
            decoder,
            rope,
            number_channels: config.number_channels.unwrap_or(1),
            enable_channel_interleave: config.enable_channel_interleave.unwrap_or(true),
        })
    }

    /// Encode a waveform to audio codes.
    /// waveform: (1, C, T) — stereo interleaved if number_channels > 1
    /// Returns: (T_encoded, n_vq) LongTensor
    pub fn encode(&self, waveform: &Tensor) -> Result<Tensor> {
        let mut x = waveform.clone();

        // Pad BEFORE interleaving (matching Python _flatten_channels_for_codec)
        // Reference: modeling_moss_audio_tokenizer.py _encode_frame()
        let downsample_rate = self.downsample_rate();
        let (b, c, t) = x.dims3()?;

        // Pad each channel to downsample_rate multiple
        let t_padded = if t % downsample_rate != 0 {
            let pad = downsample_rate - (t % downsample_rate);
            let pad_t = Tensor::zeros((b, c, pad), x.dtype(), x.device())?;
            Tensor::cat(&[&x, &pad_t], 2)?
        } else {
            x.clone()
        };
        let t_padded_len = t_padded.dims()[2];

        // Channel de-interleave if needed: (B, C, T) -> (B, 1, C*T) if interleaved
        // Track original_lengths (before any padding) for valid frame calculation.
        let original_lengths: usize;
        if self.number_channels > 1 && self.enable_channel_interleave {
            // (B, C, T) -> (B, T, C) -> (B, 1, C*T) - interleave channels
            // Reference: _flatten_channels_for_codec() in Python
            original_lengths = c * t_padded_len; // Total samples after interleaving (padded)
            x = t_padded
                .transpose(1, 2)?
                .reshape((b, 1, original_lengths))?;
        } else {
            original_lengths = t_padded_len;
            x = t_padded;
        }

        // Debug: print original_lengths (remove in production)
        // eprintln!("DEBUG encode: original_lengths={}, waveform shape={:?}", original_lengths, waveform.dims());

        // No more padding needed - already done before interleave
        let padded_len = x;

        // Track valid frame count through PatchedPretransform layers,
        // matching Python's audio_codes_lengths = input_lengths // downsample_rate.
        // Use original_lengths (before padding) like Python does.
        let mut valid_frames = original_lengths;
        let mut x = padded_len;

        // Forward through encoder layers
        // Create input_lengths tensor for attention masking (B=1)
        let _input_lengths_tensor = Tensor::from_vec(vec![valid_frames as u32], (1,), x.device())?;

        for layer in &self.encoder {
            match layer {
                DecoderLayer::Pretransform(p) => {
                    // Python: output_lengths = input_lengths // patch_size (using original, not padded)
                    valid_frames /= p.downsample_ratio();
                    x = p.encode(&x)?;
                    // Update input_lengths for next layer
                    let _new_lengths =
                        Tensor::from_vec(vec![valid_frames as u32], (1,), x.device())?;
                }
                DecoderLayer::Transformer(t) => {
                    let lengths = Tensor::from_vec(vec![valid_frames as u32], (1,), x.device())?;
                    x = t.forward(&x, Some(&self.rope), None, true, Some(&lengths))?;
                }
            }
        }

        // Quantize: encoder output (B, input_dim, T_enc) -> codes (n_q, B, T_enc)
        // Pass valid_frames as input_lengths to mask padding positions
        let input_lengths_tensor = Tensor::from_vec(vec![valid_frames as u32], (1,), x.device())?;
        let codes = self
            .quantizer
            .encode_codes(&x, Some(&input_lengths_tensor))?;

        // Get the actual time dimension from codes (not trimmed)
        let (nq, _b, t_enc) = codes.dims3()?;

        // Transpose to (T_enc, n_q) — matching Python normalization
        // Note: Python keeps all frames but marks valid_frames as the valid length
        Ok(codes.transpose(0, 2)?.reshape((t_enc, nq))?)
    }

    /// Get the downsample rate (for padding calculation)
    fn downsample_rate(&self) -> usize {
        // The final encoder layer's downsample ratio determines the codec frame rate
        // For MOSS-Audio-Tokenizer-Nano: 240 * 2 * 2 * 2 * 4 = 3840
        self.encoder.iter().fold(1usize, |acc, layer| match layer {
            DecoderLayer::Pretransform(p) => acc * p.downsample_ratio(),
            DecoderLayer::Transformer(_) => acc,
        })
    }

    /// Decode audio codes to waveform.
    /// tokens: (n_q, B, T)
    /// Returns: (B, C, T) waveform
    pub fn decode(&self, tokens: &Tensor) -> Result<Tensor> {
        let mut x = self.quantizer.decode_codes(tokens)?;

        for layer in &self.decoder {
            match layer {
                DecoderLayer::Pretransform(p) => {
                    x = p.decode(&x)?;
                }
                DecoderLayer::Transformer(t) => {
                    x = t.forward(&x, Some(&self.rope), None, true, None)?;
                }
            }
        }

        // Restore channels if interleaved
        if self.number_channels > 1 && self.enable_channel_interleave {
            let (b, _, tl) = x.dims3()?;
            // Output of decoder layers is (B, 1, T * channels)
            // We need (B, channels, T)
            x = x
                .squeeze(1)?
                .reshape((b, tl / self.number_channels, self.number_channels))?
                .transpose(1, 2)?
                .contiguous()?;
        }

        Ok(x)
    }
}
