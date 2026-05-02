use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

use crate::models::audio_io;
use crate::models::audio_tokenizer::{AudioTokenizerModel, TokenizerConfig};
use crate::models::input_builder::{self, InferenceMode};
use crate::models::lm::{LMConfig, MossTTSNanoLM};
use crate::models::prompting::Prompting;
use crate::models::text_normalize::{self};
use crate::models::tokenizer::MossTTSNanoTokenizer;
use crate::sampling::Sampler;

/// Sample rate for audio output.
pub const SAMPLE_RATE: usize = 48000;

/// Parameters shared between TTS inference modes.
pub struct GenerateParams {
    pub max_frames: usize,
    pub do_sample: bool,
    pub text_temperature: f64,
    pub text_top_p: f64,
    pub text_top_k: usize,
    pub audio_temperature: f64,
    pub audio_top_p: f64,
    pub audio_top_k: usize,
    pub audio_repetition_penalty: f64,
    pub normalize_text: bool,
    pub seed: Option<u64>,
    pub export_dir: Option<std::path::PathBuf>,
    /// Max text tokens per chunk for voice_clone mode.
    /// Set to 0 to disable chunking (single-pass generation).
    pub voice_clone_max_text_tokens: usize,
}

pub struct MossTTS {
    lm: MossTTSNanoLM,
    tokenizer: MossTTSNanoTokenizer,
    audio_tokenizer: AudioTokenizerModel,
    prompting: Prompting,
}

impl MossTTS {
    pub fn load<P: AsRef<Path>>(
        tokenizer_path: P,
        lm_config_path: P,
        lm_weights_path: P,
        codec_config_path: P,
        codec_weights_path: P,
        device: &Device,
    ) -> Result<Self> {
        // 1. Load Tokenizer
        let tokenizer = MossTTSNanoTokenizer::load(tokenizer_path)?;

        // 2. Load LM
        let lm_config: LMConfig = serde_json::from_str(&std::fs::read_to_string(lm_config_path)?)?;
        let lm_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[lm_weights_path], DType::F32, device)? };
        let lm = MossTTSNanoLM::load(lm_vb, &lm_config)?;

        // 3. Load AudioTokenizer
        let codec_config: TokenizerConfig =
            serde_json::from_str(&std::fs::read_to_string(codec_config_path)?)?;

        let codec_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[codec_weights_path], DType::F32, device)?
        };
        let audio_tokenizer = AudioTokenizerModel::load(codec_vb, &codec_config)?;

        let prompting = Prompting::new(&lm_config);

        Ok(Self {
            lm,
            tokenizer,
            audio_tokenizer,
            prompting,
        })
    }

    /// Get a reference to the text tokenizer (for token counting, etc.).
    pub fn tokenizer_ref(&self) -> &MossTTSNanoTokenizer {
        &self.tokenizer
    }

    /// Text-to-speech in continuation mode (original behavior, no reference audio).
    pub fn text_to_speech(&self, text: &str, params: &GenerateParams) -> Result<Tensor> {
        let device = self.lm.transformer.ln_f.weight().device();

        let normalized_text = if params.normalize_text {
            text_normalize::prepare_text_for_sentence_chunking(text)
        } else {
            text.to_string()
        };
        println!("[DEBUG] Normalized text: \"{}\"", normalized_text);

        // Build input_ids using input_builder for consistency
        let inference_input = input_builder::build_inference_input_ids(
            &normalized_text,
            InferenceMode::Continuation,
            None, // no prompt_text
            None, // no prompt_audio_codes
            &self.lm.config,
            &self.prompting,
            &self.tokenizer,
            device,
        )?;

        let input_ids = inference_input.input_ids;

        self.generate_and_decode(&input_ids, params)
    }

    /// Voice clone: synthesize text in the style of a reference audio clip.
    /// Supports long text via sentence chunking when `voice_clone_max_text_tokens > 0`
    /// and the tokenized text exceeds the budget.
    pub fn voice_clone(
        &self,
        text: &str,
        prompt_audio_path: &Path,
        params: &GenerateParams,
    ) -> Result<Tensor> {
        let device = self.lm.transformer.ln_f.weight().device();

        let normalized_text = if params.normalize_text {
            text_normalize::prepare_text_for_sentence_chunking(text)
        } else {
            text.to_string()
        };
        println!("[DEBUG] Normalized text: \"{}\"", normalized_text);
        println!(
            "[DEBUG] Loading prompt audio: {}",
            prompt_audio_path.display()
        );

        // 1. Load and prepare reference audio → (1, C, T) on device
        let waveform = audio_io::load_and_prepare_wav(
            prompt_audio_path,
            SAMPLE_RATE,
            2, // target_channels (stereo)
            device,
        )?;
        println!("[DEBUG] Prompt audio shape: {:?}", waveform.dims());

        // 2. Encode waveform to audio codes → (T_encoded, n_vq)
        let audio_codes = self.audio_tokenizer.encode(&waveform)?;
        let (n_frames, n_vq) = audio_codes.dims2()?;
        println!(
            "[DEBUG] Encoded prompt audio: {} frames, {} VQ channels",
            n_frames, n_vq
        );

        // 3. Split text into chunks if needed
        let max_text_tokens = params.voice_clone_max_text_tokens;
        let text_chunks = if max_text_tokens > 0 {
            let count_fn = |t: &str| self.tokenizer.encode(t).map(|v| v.len()).unwrap_or(0);
            let chunks = text_normalize::split_into_best_sentences(
                &normalized_text,
                max_text_tokens,
                &count_fn,
            );
            if chunks.len() > 1 {
                println!(
                    "[INFO] Text split into {} chunks (max_tokens={})",
                    chunks.len(),
                    max_text_tokens
                );
                for (i, chunk) in chunks.iter().enumerate() {
                    let token_count = self.tokenizer.encode(chunk).map(|v| v.len()).unwrap_or(0);
                    println!(
                        "[INFO]   [chunk {}] ({} tokens) \"{}\"",
                        i + 1,
                        token_count,
                        chunk
                    );
                }
                chunks
            } else {
                vec![normalized_text]
            }
        } else {
            vec![normalized_text]
        };

        // 4. Generate audio for each chunk
        let mut all_waveform_segments: Vec<Tensor> = Vec::new();

        for (chunk_index, chunk_text) in text_chunks.iter().enumerate() {
            println!(
                "[INFO] Generating chunk {}/{}: \"{}\"",
                chunk_index + 1,
                text_chunks.len(),
                chunk_text
            );

            let chunk_waveform =
                self.voice_clone_single_chunk(chunk_text, &audio_codes, device, params)?;

            // The waveform shape from generate_and_decode is (C, T) — squeeze batch dim
            let chunk_dims = chunk_waveform.dims();
            if chunk_dims.len() == 3 {
                // (B, C, T) → (C, T)
                let squeezed = chunk_waveform.squeeze(0)?;
                all_waveform_segments.push(squeezed);
            } else {
                all_waveform_segments.push(chunk_waveform);
            }

            // Insert silence between chunks (not after the last one)
            if chunk_index < text_chunks.len() - 1 {
                let pause_seconds = text_normalize::estimate_inter_chunk_pause_seconds(chunk_text);
                let pause_samples = (SAMPLE_RATE as f64 * pause_seconds).round() as usize;
                if pause_samples > 0 {
                    let num_channels = all_waveform_segments
                        .last()
                        .and_then(|t| t.dims().first().copied())
                        .unwrap_or(2);
                    let silence = Tensor::zeros((num_channels, pause_samples), DType::F32, device)?;
                    println!(
                        "[INFO] Inserting {:.2}s silence between chunks ({} samples, {} channels)",
                        pause_seconds, pause_samples, num_channels
                    );
                    all_waveform_segments.push(silence);
                }
            }
        }

        // 5. Concatenate all segments along the time dimension
        if all_waveform_segments.is_empty() {
            return Ok(Tensor::zeros((2, 0), DType::F32, device)?);
        }

        let waveform = Tensor::cat(&all_waveform_segments, 1)?;
        println!("[DEBUG] Final waveform shape: {:?}", waveform.dims());
        Ok(waveform)
    }

    /// Generate a single voice clone chunk (internal).
    fn voice_clone_single_chunk(
        &self,
        text: &str,
        audio_codes: &Tensor,
        device: &Device,
        params: &GenerateParams,
    ) -> Result<Tensor> {
        let inference_input = input_builder::build_inference_input_ids(
            text,
            InferenceMode::VoiceClone,
            None,
            Some(audio_codes),
            &self.lm.config,
            &self.prompting,
            &self.tokenizer,
            device,
        )?;
        let input_ids = inference_input.input_ids;
        println!("[DEBUG] Input IDs shape: {:?}", input_ids.dims());

        self.generate_and_decode(&input_ids, params)
    }

    /// Core generate + decode logic shared by all modes.
    fn generate_and_decode(&self, input_ids: &Tensor, params: &GenerateParams) -> Result<Tensor> {
        let device = input_ids.device();
        let export_dir = params.export_dir.as_deref();

        if let Some(dir) = export_dir {
            crate::testing::save_npy(dir.join("input_ids.npy"), &input_ids.to_dtype(DType::I64)?)
                .map_err(|e| anyhow::anyhow!(e))?;
        }

        // Sampling parameters
        let text_top_k_clamped = params.text_top_k.min(2);
        let mut text_sampler = Sampler::new(
            params.do_sample,
            params.text_temperature,
            params.text_top_p,
            text_top_k_clamped,
            params.seed,
        );
        let mut audio_sampler = Sampler::new(
            params.do_sample,
            params.audio_temperature,
            params.audio_top_p,
            params.audio_top_k,
            params.seed,
        );

        let audio_assistant_slot_token_id = self.lm.config.audio_assistant_slot_token_id;
        let audio_end_token_id = self.lm.config.audio_end_token_id;

        // Generate audio token frames
        let frames = self.lm.generate(
            input_ids,
            params.max_frames,
            &mut text_sampler,
            &mut audio_sampler,
            audio_assistant_slot_token_id,
            audio_end_token_id,
            params.audio_repetition_penalty,
            export_dir,
        )?;

        if frames.is_empty() {
            return Ok(Tensor::zeros((1, 0), DType::F32, device)?);
        }

        // Extract audio token IDs from generated frames
        let all_frames = Tensor::cat(&frames, 1)?;
        let audio_token_ids = all_frames.i((0, .., 1..))?; // (T, n_vq)

        if let Some(dir) = export_dir {
            crate::testing::save_npy(
                dir.join("audio_tokens.npy"),
                &audio_token_ids.to_dtype(DType::I64)?,
            )
            .map_err(|e| anyhow::anyhow!(e))?;
        }

        // AudioTokenizer expects (n_vq, 1, T)
        let audio_token_ids = audio_token_ids.transpose(0, 1)?.unsqueeze(1)?; // (n_vq, 1, T)

        // Decode to waveform
        let waveform = self.audio_tokenizer.decode(&audio_token_ids)?;

        if let Some(dir) = export_dir {
            crate::testing::save_npy(dir.join("decoded_waveform.npy"), &waveform.squeeze(0)?)
                .map_err(|e| anyhow::anyhow!(e))?;
        }

        // Shape is (B, C, T). Squeeze batch dim if 1 → (C, T)
        Ok(waveform.squeeze(0)?)
    }
}
