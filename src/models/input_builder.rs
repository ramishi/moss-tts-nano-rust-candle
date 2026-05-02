use crate::models::lm::LMConfig;
use crate::models::prompting::Prompting;
use crate::models::tokenizer::MossTTSNanoTokenizer;
use anyhow::Result;
use candle_core::{Device, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceMode {
    Continuation,
    VoiceClone,
}

pub struct InferenceInput {
    /// (1, S, n_vq + 1) — input token IDs for the LM
    pub input_ids: Tensor,
}

/// Build a text-only row section for the input_ids tensor.
/// Each text token becomes a row: [token_id, pad, pad, ..., pad]  (length = n_vq + 1)
fn build_text_rows(
    token_ids: &[u32],
    n_vq: usize,
    audio_pad_token_id: u32,
    device: &Device,
) -> Result<Tensor> {
    let row_width = n_vq + 1;
    if token_ids.is_empty() {
        // Return empty tensor with shape (0, row_width)
        return Ok(Tensor::zeros(
            (0, row_width),
            candle_core::DType::U32,
            device,
        )?);
    }

    let num_rows = token_ids.len();
    let mut data = vec![audio_pad_token_id; num_rows * row_width];
    for (i, &token_id) in token_ids.iter().enumerate() {
        data[i * row_width] = token_id;
    }

    let tensor = Tensor::from_slice(&data, (num_rows, row_width), device)?;
    Ok(tensor)
}

/// Build audio prefix rows for the input_ids tensor.
/// audio_codes: (T, n_vq) — encoded prompt audio
/// Each row becomes: [slot_token_id, code_0, code_1, ..., code_15]
fn build_audio_prefix_rows(
    audio_codes: &Tensor,
    slot_token_id: u32,
    audio_pad_token_id: u32,
) -> Result<Tensor> {
    let (t, n_vq) = audio_codes.dims2()?;
    let row_width = n_vq + 1;
    let device = audio_codes.device();

    if t == 0 {
        return Ok(Tensor::zeros(
            (0, row_width),
            candle_core::DType::U32,
            device,
        )?);
    }

    // audio_codes is (T, n_vq), flatten to get all codes in order
    let codes_flat = audio_codes.flatten_all()?.to_vec1::<u32>()?;

    // Build rows: each row = [slot, code_0, ..., code_{n_vq-1}]
    let mut data = vec![audio_pad_token_id; t * row_width];
    for i in 0..t {
        data[i * row_width] = slot_token_id;
        data[i * row_width + 1..i * row_width + 1 + n_vq]
            .copy_from_slice(&codes_flat[i * n_vq..(i + 1) * n_vq]);
    }

    let tensor = Tensor::from_slice(&data, (t, row_width), device)?;
    Ok(tensor)
}

/// Build the full input_ids tensor for inference.
///
/// # Arguments
/// * `text` - The target text to synthesize
/// * `mode` - voice_clone or continuation
/// * `prompt_text` - For continuation mode, the prompt text before the target text
/// * `prompt_audio_codes` - Encoded reference audio codes, shape (T, n_vq)
pub fn build_inference_input_ids(
    text: &str,
    mode: InferenceMode,
    prompt_text: Option<&str>,
    prompt_audio_codes: Option<&Tensor>,
    config: &LMConfig,
    prompting: &Prompting,
    tokenizer: &MossTTSNanoTokenizer,
    device: &Device,
) -> Result<InferenceInput> {
    let n_vq = config.n_vq;
    let audio_pad_token_id = config.audio_pad_token_id as u32;

    let sections: Vec<Tensor> = match mode {
        InferenceMode::VoiceClone => {
            // Voice clone: reference audio + target text, no prompt text
            let prompt_audio = prompt_audio_codes
                .ok_or_else(|| anyhow::anyhow!("Voice clone mode requires prompt_audio_codes"))?;

            let text_token_ids = tokenizer.encode(text)?;
            let prefix_token_ids = prompting.build_user_prompt_prefix(tokenizer)?;
            let mut prefix_with_audio_start = prefix_token_ids;
            prefix_with_audio_start.push(config.audio_start_token_id);

            let mut suffix_token_ids = Vec::new();
            suffix_token_ids.push(config.audio_end_token_id);
            suffix_token_ids.extend(prompting.build_user_prompt_after_reference(tokenizer)?);
            suffix_token_ids.extend(&text_token_ids);
            suffix_token_ids.extend(prompting.build_assistant_prompt_prefix(tokenizer)?);
            suffix_token_ids.push(config.audio_start_token_id);

            vec![
                build_text_rows(&prefix_with_audio_start, n_vq, audio_pad_token_id, device)?,
                build_audio_prefix_rows(
                    prompt_audio,
                    config.audio_user_slot_token_id,
                    audio_pad_token_id,
                )?,
                build_text_rows(&suffix_token_ids, n_vq, audio_pad_token_id, device)?,
            ]
        }

        InferenceMode::Continuation => {
            // Continuation mode
            let effective_text = match prompt_text {
                Some(pt) => format!("{}{}", pt, text),
                None => text.to_string(),
            };

            let prompt_token_ids = prompting.build_prompt_token_ids(tokenizer, &effective_text)?;
            let mut sections = vec![
                build_text_rows(&prompt_token_ids, n_vq, audio_pad_token_id, device)?,
                build_text_rows(
                    &[config.audio_start_token_id],
                    n_vq,
                    audio_pad_token_id,
                    device,
                )?,
            ];

            if let Some(audio) = prompt_audio_codes {
                sections.push(build_audio_prefix_rows(
                    audio,
                    config.audio_assistant_slot_token_id,
                    audio_pad_token_id,
                )?);
            }

            sections
        }
    };

    // Concatenate all sections along dim=0
    let input_ids = Tensor::cat(&sections, 0)?;
    // Add batch dimension: (S, n_vq+1) -> (1, S, n_vq+1)
    let input_ids = input_ids.unsqueeze(0)?;

    Ok(InferenceInput { input_ids })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    fn test_config() -> LMConfig {
        LMConfig {
            gpt2_config: crate::models::lm::GPT2Config {
                vocab_size: 1024,
                n_embd: 256,
                n_head: 4,
                n_inner: None,
                n_layer: 1,
                n_positions: 1024,
                layer_norm_epsilon: 1e-5,
                activation_function: "gelu".to_string(),
                rope_base: None,
            },
            n_vq: 16,
            audio_codebook_sizes: vec![1024; 16],
            audio_pad_token_id: 1024,
            audio_vocab_size: 1024,
            audio_start_token_id: 6,
            audio_end_token_id: 7,
            audio_user_slot_token_id: 8,
            audio_assistant_slot_token_id: 9,
            local_transformer_layers: 1,
            im_start_token_id: 4,
            im_end_token_id: 5,
        }
    }

    #[test]
    fn test_build_text_rows() {
        let device = Device::Cpu;
        let token_ids = vec![10u32, 20, 30];
        let rows = build_text_rows(&token_ids, 16, 1024, &device).unwrap();

        assert_eq!(rows.dims(), &[3, 17]);
        let data = rows.flatten_all().unwrap().to_vec1::<u32>().unwrap();

        // Row 0: [10, 1024, ..., 1024]  (width=17)
        assert_eq!(data[0], 10);
        assert_eq!(data[1], 1024);
        assert_eq!(data[17], 20); // Row 1 first element
        assert_eq!(data[34], 30); // Row 2 first element
    }

    #[test]
    fn test_build_text_rows_empty() {
        let device = Device::Cpu;
        let rows = build_text_rows(&[], 16, 1024, &device).unwrap();
        assert_eq!(rows.dims(), &[0, 17]);
    }

    #[test]
    fn test_build_audio_prefix_rows() {
        let device = Device::Cpu;
        // 3 frames, 4 quantizers (using small n_vq for test)
        let codes =
            Tensor::from_slice(&[1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], (3, 4), &device)
                .unwrap();

        let rows = build_audio_prefix_rows(&codes, 8, 99).unwrap();

        assert_eq!(rows.dims(), &[3, 5]);
        let data = rows.flatten_all().unwrap().to_vec1::<u32>().unwrap();

        // Row 0: [8, 1, 2, 3, 4]
        assert_eq!(data[0], 8); // slot
        assert_eq!(data[1], 1);
        assert_eq!(data[2], 2);
        assert_eq!(data[3], 3);
        assert_eq!(data[4], 4);
        // Row 1: [8, 5, 6, 7, 8]
        assert_eq!(data[5], 8);
        assert_eq!(data[9], 8);
    }

    #[test]
    fn test_build_audio_prefix_rows_empty() {
        let device = Device::Cpu;
        let codes = Tensor::from_slice(&[0u32; 0], (0, 4), &device).unwrap();
        let rows = build_audio_prefix_rows(&codes, 8, 99).unwrap();
        assert_eq!(rows.dims(), &[0, 5]);
    }

    #[test]
    fn test_voice_clone_input_shape() {
        let device = Device::Cpu;
        let config = test_config();
        let prompting = Prompting::new(&config);

        // Mock audio codes: 5 frames, 16 quantizers
        let audio_codes =
            Tensor::from_slice(&(0u32..80).collect::<Vec<u32>>(), (5, 16), &device).unwrap();

        // We can't use a real tokenizer in tests without the model files,
        // so just test the shape of continuation mode (which works without audio)
        // Voice clone needs a tokenizer which requires model files.
        // We'll test this via integration tests instead.
    }

    #[test]
    fn test_continuation_input_shape() {
        let device = Device::Cpu;
        let config = test_config();
        let prompting = Prompting::new(&config);

        // We need a real tokenizer for this test, but we can't instantiate one
        // without model files. This is tested via integration tests.
    }
}
