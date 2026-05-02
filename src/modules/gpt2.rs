use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};
use std::path::Path;

pub struct GPT2RotaryEmbedding {
    inv_freq: Tensor,
}

impl GPT2RotaryEmbedding {
    pub fn new(dim: usize, base: f32, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / (base.powf(i as f32 / dim as f32)))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (dim / 2,), device)?;
        Ok(Self { inv_freq })
    }

    pub fn forward(&self, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        // position_ids: (B, S)
        let (b, s) = position_ids.dims2()?;
        let dim_f = self.inv_freq.dim(0)?;

        let position_ids_f = position_ids.to_dtype(DType::F32)?.unsqueeze(2)?; // (B, S, 1)
        let inv_freq_f = self.inv_freq.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, dim/2)
        let freqs = position_ids_f.broadcast_mul(&inv_freq_f)?; // (B, S, dim/2)

        // repeat_interleave 2 along last dim:
        // (B, S, dim_f) -> (B, S, dim_f, 1) -> (B, S, dim_f, 2) -> (B, S, dim_f * 2)
        let cos = freqs
            .cos()?
            .unsqueeze(3)?
            .broadcast_as((b, s, dim_f, 2))?
            .reshape((b, s, dim_f * 2))?;
        let sin = freqs
            .sin()?
            .unsqueeze(3)?
            .broadcast_as((b, s, dim_f, 2))?
            .reshape((b, s, dim_f * 2))?;

        // Add head dimension for broadcasting: (B, S, 1, dim)
        Ok((cos.unsqueeze(2)?, sin.unsqueeze(2)?))
    }
}

pub fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let x_shape = x.dims();
    let mut reshaped_dims = x_shape.to_vec();
    let last = reshaped_dims.pop().unwrap();
    reshaped_dims.push(last / 2);
    reshaped_dims.push(2);

    let x = x.reshape(reshaped_dims)?;
    // even = x[..., 0], odd = x[..., 1]
    let x_even = x.narrow(x.rank() - 1, 0, 1)?;
    let x_odd = x.narrow(x.rank() - 1, 1, 1)?;

    // stack((-odd, even), dim=-1)
    let out = Tensor::cat(&[&x_odd.neg()?, &x_even], x.rank() - 1)?;
    out.reshape(x_shape)
}

pub fn apply_rotary_pos_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // x: (B, S, H, D)
    // cos/sin: (B, S, 1, D)
    let x_cos = x.broadcast_mul(cos)?;
    let x_sin = rotate_half(x)?.broadcast_mul(sin)?;
    x_cos + x_sin
}

pub struct GPT2Attention {
    c_attn: Linear,
    c_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    embed_dim: usize,
    rotary_emb: Option<GPT2RotaryEmbedding>,
}

impl GPT2Attention {
    pub fn load(
        vb: VarBuilder,
        n_embd: usize,
        n_head: usize,
        rope_base: Option<f32>,
    ) -> Result<Self> {
        let c_attn = candle_nn::linear(n_embd, 3 * n_embd, vb.pp("c_attn"))?;
        let c_proj = candle_nn::linear(n_embd, n_embd, vb.pp("c_proj"))?;
        let head_dim = n_embd / n_head;

        let rotary_emb = if let Some(base) = rope_base {
            Some(GPT2RotaryEmbedding::new(head_dim, base, vb.device())?)
        } else {
            None
        };

        Ok(Self {
            c_attn,
            c_proj,
            num_heads: n_head,
            head_dim,
            embed_dim: n_embd,
            rotary_emb,
        })
    }

    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        position_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
        layer_past: Option<&(Tensor, Tensor)>,
        use_cache: bool,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let (b, s, _) = x.dims3()?;
        let qkv = self.c_attn.forward(x)?; // (B, S, 3*D)

        let mut q = qkv.narrow(2, 0, self.embed_dim)?;
        let mut k = qkv.narrow(2, self.embed_dim, self.embed_dim)?;
        let mut v = qkv.narrow(2, 2 * self.embed_dim, self.embed_dim)?;

        // Reshape to (B, S, H, D)
        q = q.reshape((b, s, self.num_heads, self.head_dim))?;
        k = k.reshape((b, s, self.num_heads, self.head_dim))?;
        v = v.reshape((b, s, self.num_heads, self.head_dim))?;

        if let Some(rope) = &self.rotary_emb {
            let pos = position_ids.ok_or_else(|| {
                candle_core::Error::Msg("position_ids required for RoPE".to_string())
            })?;
            let (cos, sin) = rope.forward(pos)?;
            q = apply_rotary_pos_emb(&q, &cos, &sin)?;
            k = apply_rotary_pos_emb(&k, &cos, &sin)?;
        }

        if let Some((past_k, past_v)) = layer_past {
            k = Tensor::cat(&[past_k, &k], 1)?;
            v = Tensor::cat(&[past_v, &v], 1)?;
        }

        let present = if use_cache {
            Some((k.clone(), v.clone()))
        } else {
            None
        };

        // Attention: (B, H, S, S_kv)
        let q = q.transpose(1, 2)?.contiguous()?; // (B, H, S, D)
        let k = k.transpose(1, 2)?.contiguous()?; // (B, H, S_kv, D)
        let v = v.transpose(1, 2)?.contiguous()?; // (B, H, S_kv, D)

        let mut scores = q.matmul(&k.transpose(2, 3)?)?;
        scores = (scores / (self.head_dim as f64).sqrt())?;

        if let Some(m) = mask {
            // mask is usually (B, 1, S, S_kv) or similar
            scores = scores.broadcast_add(&m.to_dtype(scores.dtype())?)?;
        }

        let probs = candle_nn::ops::softmax(&scores, 3)?;
        let attn_output = probs.matmul(&v)?; // (B, H, S, D)

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b, s, self.embed_dim))?;
        let attn_output = self.c_proj.forward(&attn_output)?;

        Ok((attn_output, present))
    }
}

pub struct GPT2MLP {
    fc_in: Linear,
    fc_out: Linear,
}

impl GPT2MLP {
    pub fn load(vb: VarBuilder, n_embd: usize, n_inner: usize) -> Result<Self> {
        let fc_in = candle_nn::linear(n_embd, n_inner, vb.pp("fc_in"))?;
        let fc_out = candle_nn::linear(n_inner, n_embd, vb.pp("fc_out"))?;
        Ok(Self { fc_in, fc_out })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc_in.forward(x)?;
        // GELU with tanh approximation: 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let x3 = x.powf(3.0)?;
        let inner = ((x.clone() + (x3 * 0.044715)?)? * (2.0f64 / std::f64::consts::PI).sqrt())?;
        let x = (x * (inner.tanh()? + 1.0)?)?;
        let x = (x * 0.5)?;
        self.fc_out.forward(&x)
    }
}

pub struct GPT2Block {
    ln_1: LayerNorm,
    attn: GPT2Attention,
    ln_2: LayerNorm,
    mlp: GPT2MLP,
}

impl GPT2Block {
    pub fn load(
        vb: VarBuilder,
        n_embd: usize,
        n_head: usize,
        n_inner: usize,
        rope_base: Option<f32>,
    ) -> Result<Self> {
        let ln_1 = candle_nn::layer_norm(n_embd, 1e-5, vb.pp("ln_1"))?;
        let attn = GPT2Attention::load(vb.pp("attn"), n_embd, n_head, rope_base)?;
        let ln_2 = candle_nn::layer_norm(n_embd, 1e-5, vb.pp("ln_2"))?;
        let mlp = GPT2MLP::load(vb.pp("mlp"), n_embd, n_inner)?;
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }

    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        position_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
        layer_past: Option<&(Tensor, Tensor)>,
        use_cache: bool,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let residual = x;
        let x = self.ln_1.forward(x)?;
        let (attn_out, present) =
            self.attn
                .forward_with_cache(&x, position_ids, mask, layer_past, use_cache)?;
        let x = (residual + attn_out)?;

        let residual = &x;
        let x = self.ln_2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let x = (residual + x)?;

        Ok((x, present))
    }
}

pub struct GPT2Model {
    pub wte: Option<Embedding>,
    pub wpe: Option<Embedding>,
    pub h: Vec<GPT2Block>,
    pub ln_f: LayerNorm,
    pub config: GPT2Config,
}

use crate::models::lm::GPT2Config;
use candle_nn::Embedding;

impl GPT2Model {
    pub fn load(vb: VarBuilder, config: &GPT2Config) -> Result<Self> {
        let wte = if vb.contains_tensor("wte.weight") {
            Some(candle_nn::embedding(
                config.vocab_size,
                config.n_embd,
                vb.pp("wte"),
            )?)
        } else {
            None
        };

        let wpe = if config.rope_base.is_none() && vb.contains_tensor("wpe.weight") {
            Some(candle_nn::embedding(
                config.n_positions,
                config.n_embd,
                vb.pp("wpe"),
            )?)
        } else {
            None
        };

        let mut h = Vec::new();
        let h_vb = vb.pp("h");
        for i in 0..config.n_layer {
            h.push(GPT2Block::load(
                h_vb.pp(i.to_string()),
                config.n_embd,
                config.n_head,
                config.n_inner.unwrap_or(4 * config.n_embd),
                config.rope_base,
            )?);
        }

        let ln_f = candle_nn::layer_norm(config.n_embd, config.layer_norm_epsilon, vb.pp("ln_f"))?;

        Ok(Self {
            wte,
            wpe,
            h,
            ln_f,
            config: config.clone(),
        })
    }

    pub fn forward_with_embeds(
        &self,
        inputs_embeds: &Tensor,
        past_key_values: Option<&Vec<(Tensor, Tensor)>>,
        use_cache: bool,
        export: Option<(&Path, String)>,
    ) -> Result<(Tensor, Option<Vec<(Tensor, Tensor)>>)> {
        let (b, s, _) = inputs_embeds.dims3()?;
        let device = inputs_embeds.device();

        let mut x = inputs_embeds.clone();

        let past_len = if let Some(p) = past_key_values {
            if p.is_empty() { 0 } else { p[0].0.dim(1)? }
        } else {
            0
        };

        let position_ids = Tensor::arange(past_len as u32, (past_len + s) as u32, device)?
            .unsqueeze(0)?
            .expand((b, s))?;

        if let Some(wpe) = &self.wpe {
            x = (x + wpe.forward(&position_ids)?)?;
        }

        // Causal mask for eager attention
        let mask = if s > 1 || past_len > 0 {
            let mask: Vec<f32> = (0..s)
                .flat_map(|i| {
                    (0..past_len + s).map(move |j| {
                        if j > i + past_len {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        }
                    })
                })
                .collect();
            Some(Tensor::from_vec(mask, (1, 1, s, past_len + s), device)?)
        } else {
            None
        };

        let mut presents = if use_cache { Some(Vec::new()) } else { None };

        for (i, block) in self.h.iter().enumerate() {
            let past = past_key_values.and_then(|p| p.get(i));
            let (out, present) = block.forward_with_cache(
                &x,
                Some(&position_ids),
                mask.as_ref(),
                past,
                use_cache,
            )?;
            x = out;

            if let Some((dir, ref prefix)) = export {
                crate::testing::save_npy(dir.join(format!("{}_hidden_L{}.npy", prefix, i)), &x)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            }

            if let Some(p) = present {
                if let Some(presents) = &mut presents {
                    presents.push(p);
                }
            }
        }

        x = self.ln_f.forward(&x)?;

        Ok((x, presents))
    }
}
