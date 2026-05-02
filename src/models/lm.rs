use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};
use serde::Deserialize;
use std::path::Path;
use std::time::Instant;

use crate::modules::gpt2::GPT2Model;
use crate::sampling::Sampler;
use crate::sampling::apply_repetition_penalty;

#[derive(Debug, Deserialize, Clone)]
pub struct GPT2Config {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_inner: Option<usize>,
    pub n_layer: usize,
    pub n_positions: usize,
    pub layer_norm_epsilon: f64,
    pub activation_function: String,
    pub rope_base: Option<f32>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LMConfig {
    pub gpt2_config: GPT2Config,
    pub n_vq: usize,
    pub audio_codebook_sizes: Vec<usize>,
    pub audio_pad_token_id: usize,
    pub audio_vocab_size: usize,
    pub audio_start_token_id: u32,
    pub audio_end_token_id: u32,
    pub audio_user_slot_token_id: u32,
    pub audio_assistant_slot_token_id: u32,
    pub local_transformer_layers: usize,
    pub im_start_token_id: u32,
    pub im_end_token_id: u32,
}

pub struct MossTTSNanoLM {
    pub transformer: GPT2Model,
    pub local_transformer: GPT2Model,
    pub audio_embeddings: Vec<Embedding>,
    pub text_lm_head: Linear,
    pub audio_lm_heads: Vec<Linear>,
    pub config: LMConfig,
}

impl MossTTSNanoLM {
    pub fn load(vb: VarBuilder, config: &LMConfig) -> anyhow::Result<Self> {
        let gpt2_vb = vb.pp("transformer");
        let transformer = GPT2Model::load(gpt2_vb, &config.gpt2_config)?;

        let _text_head_vb = vb.pp("text_lm_head");
        let text_emb_vb = vb.pp("transformer.wte");
        let text_lm_head = Linear::new(
            text_emb_vb.get(
                (config.gpt2_config.vocab_size, config.gpt2_config.n_embd),
                "weight",
            )?,
            None,
        ); // Weight tying

        // Local transformer: same hidden size, but only 1 layer (config.local_transformer_layers)
        let mut local_gpt2_config = config.gpt2_config.clone();
        local_gpt2_config.n_layer = config.local_transformer_layers;
        let local_transformer = GPT2Model::load(vb.pp("local_transformer"), &local_gpt2_config)?;

        let mut audio_embeddings = Vec::new();
        let mut audio_lm_heads = Vec::new();
        let eb_vb = vb.pp("audio_embeddings");

        for i in 0..config.n_vq {
            let weight = eb_vb.pp(i.to_string()).get(
                (config.audio_codebook_sizes[i], config.gpt2_config.n_embd),
                "weight",
            )?;
            let emb = Embedding::new(weight.clone(), config.gpt2_config.n_embd);
            let h_vb = vb.pp(format!("audio_lm_heads.{}", i));
            let head = Linear::new(
                h_vb.get(
                    (config.audio_codebook_sizes[i], config.gpt2_config.n_embd),
                    "weight",
                )?,
                None,
            );

            audio_embeddings.push(emb);
            audio_lm_heads.push(head);
        }

        Ok(Self {
            transformer,
            local_transformer,
            audio_embeddings,
            text_lm_head,
            audio_lm_heads,
            config: config.clone(),
        })
    }

    #[allow(dead_code)]
    fn load_global_transformer(
        vb: VarBuilder,
        config: &GPT2Config,
    ) -> anyhow::Result<(GPT2Model, Linear)> {
        let model = GPT2Model::load(vb.clone(), config)?;
        // Tied head
        let wte = model
            .wte
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Global transformer must have wte"))?;
        let head = Linear::new(wte.embeddings().clone(), None);
        Ok((model, head))
    }

    pub fn build_inputs_embeds(
        &self,
        input_ids: &Tensor,
        export_dir: Option<&Path>,
    ) -> candle_core::Result<Tensor> {
        // input_ids: (B, S, NQ+1)
        let dims = input_ids.dims();
        let _b = dims[0];
        let _s = dims[1];

        let text_ids = input_ids.i((.., .., 0))?;
        let wte =
            self.transformer.wte.as_ref().ok_or_else(|| {
                candle_core::Error::Msg("Global transformer missing wte".to_string())
            })?;
        let mut embeds = wte.forward(&text_ids)?;

        if let Some(dir) = export_dir {
            crate::testing::save_npy(dir.join("input_ids.npy"), input_ids)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            crate::testing::save_npy(dir.join("text_embeddings.npy"), &embeds)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            eprintln!(
                "[DUMP] audio_pad_token_id: {}",
                self.config.audio_pad_token_id
            );
            eprintln!("[DUMP] n_vq: {}", self.config.n_vq);
        }

        for i in 0..self.config.n_vq {
            let channel_ids = input_ids.i((.., .., i + 1))?;
            // valid_mask = channel_ids != audio_pad_token_id
            let mask = channel_ids
                .ne(self.config.audio_pad_token_id as u32)?
                .to_dtype(DType::F32)?
                .unsqueeze(2)?;

            // Mask out of bounds to 0 for embedding forward
            let mask_bool = channel_ids.ne(self.config.audio_pad_token_id as u32)?;
            let safe_ids =
                mask_bool.where_cond(&channel_ids, &Tensor::zeros_like(&channel_ids)?)?;

            let audio_emb = self.audio_embeddings[i].forward(&safe_ids)?;
            embeds = (embeds + audio_emb.broadcast_mul(&mask)?)?;
        }

        if let Some(dir) = export_dir {
            crate::testing::save_npy(dir.join("input_embeds.npy"), &embeds)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        }

        Ok(embeds)
    }

    pub fn decode_local_last_hidden_state(
        &self,
        local_inputs_embeds: &Tensor,
        past_key_values: Option<&Vec<(Tensor, Tensor)>>,
    ) -> candle_core::Result<(Tensor, Option<Vec<(Tensor, Tensor)>>)> {
        let (x, pkv) = self.local_transformer.forward_with_embeds(
            local_inputs_embeds,
            past_key_values,
            true,
            None,
        )?;
        // Return only the last hidden state: (B, 1, D)
        Ok((x.i((.., x.dim(1)? - 1, ..))?, pkv))
    }
    pub fn forward(
        &self,
        input_ids: &Tensor,
        past_key_values: Option<&Vec<(Tensor, Tensor)>>,
        use_cache: bool,
        export_dir: Option<&Path>,
    ) -> candle_core::Result<(Tensor, Option<Vec<(Tensor, Tensor)>>)> {
        let inputs_embeds = self.build_inputs_embeds(input_ids, export_dir)?;
        self.transformer.forward_with_embeds(
            &inputs_embeds,
            past_key_values,
            use_cache,
            export_dir.map(|d| (d, "global".to_string())),
        )
    }

    pub fn generate_step(
        &self,
        input_ids: &Tensor,
        past_key_values: Option<&Vec<(Tensor, Tensor)>>,
        text_sampler: &mut Sampler,
        audio_sampler: &mut Sampler,
        audio_assistant_slot_token_id: u32,
        audio_repetition_penalty: f64,
        generated_audio_history: &[Vec<u32>],
        export_dir: Option<&Path>,
        step_index: usize,
    ) -> Result<(Tensor, u32, Option<Vec<(Tensor, Tensor)>>)> {
        let is_first_step = step_index == 0;
        let (batch_size, seq_len, _) = input_ids.dims3()?;
        let device = input_ids.device();

        // 1. Global forward
        let (global_hidden, presents) = self.forward(
            input_ids,
            past_key_values,
            true,
            if is_first_step { export_dir } else { None },
        )?;
        // Last hidden state: (B, 1, D)
        let last_hidden = global_hidden.i((.., seq_len - 1, ..))?.unsqueeze(1)?;

        // 2. Text token prediction
        let (local_hidden, mut local_past_key_values) =
            self.decode_local_last_hidden_state(&last_hidden, None)?;
        if is_first_step {
            if let Some(dir) = export_dir {
                crate::testing::save_npy(dir.join("local_hidden_step_0.npy"), &local_hidden)
                    .map_err(|e| anyhow::anyhow!(e))?;
            }
        }

        let text_logits = self.text_lm_head.forward(&local_hidden)?;

        // Restricted sampling for text token: only assistant_slot (9) or end_token (7)
        let assistant_slot = self.config.audio_assistant_slot_token_id;
        let end_token = self.config.audio_end_token_id;

        let next_text_token = {
            let logits_v = text_logits
                .flatten_all()?
                .to_device(&candle_core::Device::Cpu)?
                .to_vec1::<f32>()?;
            let slot_logit = logits_v[assistant_slot as usize];
            let eos_logit = logits_v[end_token as usize];

            // println!("[DEBUG] Step {}: slot(9)={:.4}, eos(7)={:.4}", step_index, slot_logit, eos_logit);

            if !text_sampler.do_sample {
                // Greedy: pick the token with higher logit
                if slot_logit >= eos_logit {
                    assistant_slot
                } else {
                    end_token
                }
            } else {
                // Subset sampling: re-run sampler on just these two logits
                let candidate_logits =
                    Tensor::from_vec(vec![slot_logit, eos_logit], (2,), &Device::Cpu)?;
                let sampled_idx = text_sampler.sample(&candidate_logits)?;
                // println!("[DEBUG] Step {}: sampled_idx={}", step_index, sampled_idx);
                if sampled_idx == 0 {
                    assistant_slot
                } else {
                    end_token
                }
            }
        };

        if is_first_step {
            println!("Predicted next_text_token: {}", next_text_token);
        }

        // 3. Audio tokens prediction (16 codebooks)
        let mut audio_tokens = Vec::new();

        // Assistant slot embedding
        let text_token_tensor = Tensor::from_vec(vec![next_text_token], (batch_size, 1), device)?;
        let mut current_local_emb = self
            .transformer
            .wte
            .as_ref()
            .unwrap()
            .forward(&text_token_tensor)?;

        for i in 0..self.config.n_vq {
            // Predict next audio token using local KV cache
            let (local_hidden, next_local_pkv) = self.decode_local_last_hidden_state(
                &current_local_emb,
                local_past_key_values.as_ref(),
            )?;
            local_past_key_values = next_local_pkv;

            if is_first_step {
                if let Some(dir) = export_dir {
                    crate::testing::save_npy(
                        dir.join(format!("local_input_embeds_{}.npy", i + 1)),
                        &current_local_emb,
                    )
                    .map_err(|e| anyhow::anyhow!(e))?;
                    crate::testing::save_npy(
                        dir.join(format!("local_hidden_step_{}.npy", i + 1)),
                        &local_hidden,
                    )
                    .map_err(|e| anyhow::anyhow!(e))?;
                }
            }

            let audio_logits = self.audio_lm_heads[i].forward(&local_hidden)?;

            if is_first_step {
                if let Some(dir) = export_dir {
                    crate::testing::save_npy(
                        dir.join(format!("audio_logits_C{}.npy", i)),
                        &audio_logits.squeeze(1)?,
                    )
                    .map_err(|e| anyhow::anyhow!(e))?;
                }
            }

            // Apply repetition penalty before sampling
            let mut audio_logits_vec = audio_logits
                .flatten_all()?
                .to_device(&candle_core::Device::Cpu)?
                .to_vec1::<f32>()?;
            if i < generated_audio_history.len() {
                apply_repetition_penalty(
                    &mut audio_logits_vec,
                    &generated_audio_history[i],
                    audio_repetition_penalty as f32,
                );
            }
            let penalized_logits = Tensor::from_vec(
                audio_logits_vec,
                audio_logits.dims(),
                &candle_core::Device::Cpu,
            )?;

            let sampled_token = audio_sampler.sample(&penalized_logits.squeeze(0)?)?;
            audio_tokens.push(sampled_token);

            // Next channel embedding
            let channel_token_tensor =
                Tensor::from_vec(vec![sampled_token], (batch_size, 1), device)?;
            current_local_emb = self.audio_embeddings[i].forward(&channel_token_tensor)?;
        }

        // Result row: (B, 1, NQ+1)
        // CRITICAL: The global sequence row ALWAYS uses the assistant slot token
        // for the text dimension in generated frames, matching build_generation_row in Python.
        let mut row_vec = vec![audio_assistant_slot_token_id];
        row_vec.extend(audio_tokens);
        let next_row = Tensor::from_vec(row_vec, (batch_size, 1, self.config.n_vq + 1), device)?;

        if is_first_step {
            if let Some(dir) = export_dir {
                crate::testing::save_npy(
                    dir.join("audio_tokens_one_frame.npy"),
                    &next_row.i((0, 0, ..))?.to_dtype(DType::I64)?,
                )
                .map_err(|e| anyhow::anyhow!(e))?;
            }
        }

        Ok((next_row, next_text_token, presents))
    }

    pub fn generate(
        &self,
        input_ids: &Tensor,
        max_new_frames: usize,
        text_sampler: &mut Sampler,
        audio_sampler: &mut Sampler,
        audio_assistant_slot_token_id: u32,
        audio_end_token_id: u32,
        audio_repetition_penalty: f64,
        export_dir: Option<&Path>,
    ) -> Result<Vec<Tensor>> {
        let mut generated_frames = Vec::new();
        let mut current_input_ids = input_ids.clone();
        let mut past_key_values = None;
        // Track generated audio tokens per channel for repetition penalty.
        // generated_audio_history[channel_index] = Vec of all token ids generated for that channel so far.
        let mut generated_audio_history: Vec<Vec<u32>> = vec![Vec::new(); self.config.n_vq];

        let mut first_frame_time = None;
        let start_gen = Instant::now();

        for i in 0..max_new_frames {
            let (next_row, next_text_token, next_pkv) = self.generate_step(
                &current_input_ids,
                past_key_values.as_ref(),
                text_sampler,
                audio_sampler,
                audio_assistant_slot_token_id,
                audio_repetition_penalty,
                &generated_audio_history,
                export_dir,
                i, // Pass step index
            )?;

            if i == 0 {
                first_frame_time = Some(start_gen.elapsed());
            }

            // Check if text token is end token
            if next_text_token == audio_end_token_id {
                break;
            }

            // Extract audio tokens from this frame and update history
            // next_row shape: (B, 1, NQ+1), first dim is text (assistant_slot), rest are audio channels
            let frame_tokens = next_row
                .i((0, 0, ..))?
                .to_dtype(DType::U32)?
                .to_vec1::<u32>()?;
            for ch in 0..self.config.n_vq {
                // audio tokens are at indices 1..=n_vq in the row
                if ch + 1 < frame_tokens.len() {
                    generated_audio_history[ch].push(frame_tokens[ch + 1]);
                }
            }

            generated_frames.push(next_row.clone());
            current_input_ids = next_row; // With KV cache, we only pass the LAST token
            past_key_values = next_pkv;
        }

        if let Some(ttfb) = first_frame_time {
            println!("[METRIC] TTFB: {:?}", ttfb);
        }

        Ok(generated_frames)
    }
}
