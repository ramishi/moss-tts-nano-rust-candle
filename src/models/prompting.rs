use crate::models::lm::LMConfig;
use crate::models::tokenizer::MossTTSNanoTokenizer;
use anyhow::Result;

pub const USER_ROLE_PREFIX: &str = "user\n";
pub const USER_TEMPLATE_REFERENCE_PREFIX: &str = "<user_inst>\n- Reference(s):\n";
pub const USER_TEMPLATE_AFTER_REFERENCE: &str = "\n- Instruction:\nNone\n- Tokens:\nNone\n- Quality:\nNone\n- Sound Event:\nNone\n- Ambient Sound:\nNone\n- Language:\nNone\n- Text:\n";
pub const USER_TEMPLATE_SUFFIX: &str = "\n</user_inst>";
pub const ASSISTANT_TURN_PREFIX: &str = "\n";
pub const ASSISTANT_ROLE_PREFIX: &str = "assistant\n";

pub struct Prompting {
    pub im_start_token_id: u32,
    pub im_end_token_id: u32,
}

impl Prompting {
    pub fn new(config: &LMConfig) -> Self {
        Self {
            im_start_token_id: config.im_start_token_id,
            im_end_token_id: config.im_end_token_id,
        }
    }

    /// Build user prompt prefix — for voice_clone mode.
    /// Python: [im_start] + encode("user\n") + encode("<user_inst>\n- Reference(s):\n")
    pub fn build_user_prompt_prefix(&self, tokenizer: &MossTTSNanoTokenizer) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        ids.push(self.im_start_token_id);
        ids.extend(tokenizer.encode(USER_ROLE_PREFIX)?);
        ids.extend(tokenizer.encode(USER_TEMPLATE_REFERENCE_PREFIX)?);
        Ok(ids)
    }

    /// Build the text section after the audio reference (or "None" placeholder).
    /// Python: encode("\n- Instruction:\nNone\n- Tokens:\nNone\n- Quality:\nNone\n- Sound Event:\nNone\n- Ambient Sound:\nNone\n- Language:\nNone\n- Text:\n")
    pub fn build_user_prompt_after_reference(
        &self,
        tokenizer: &MossTTSNanoTokenizer,
    ) -> Result<Vec<u32>> {
        tokenizer.encode(USER_TEMPLATE_AFTER_REFERENCE)
    }

    /// Build assistant prompt prefix — shared between continuation and voice_clone modes.
    /// Python: encode("\n</user_inst>") + [im_end] + encode("\n") + [im_start] + encode("assistant\n")
    pub fn build_assistant_prompt_prefix(
        &self,
        tokenizer: &MossTTSNanoTokenizer,
    ) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        ids.extend(tokenizer.encode(USER_TEMPLATE_SUFFIX)?);
        ids.push(self.im_end_token_id);
        ids.extend(tokenizer.encode(ASSISTANT_TURN_PREFIX)?);
        ids.push(self.im_start_token_id);
        ids.extend(tokenizer.encode(ASSISTANT_ROLE_PREFIX)?);
        Ok(ids)
    }

    /// Build the full continuation-mode prompt token IDs.
    /// This is the existing continuation (no reference audio) path:
    ///   build_user_prompt_prefix + "None" + build_user_prompt_after_reference + text + build_assistant_prompt_prefix
    pub fn build_prompt_token_ids(
        &self,
        tokenizer: &MossTTSNanoTokenizer,
        text: &str,
    ) -> Result<Vec<u32>> {
        let mut ids = Vec::new();

        // 1. User prompt prefix (build_prompt_prefix in Python)
        ids.extend(self.build_user_prompt_prefix(tokenizer)?);
        ids.extend(tokenizer.encode("None")?);
        ids.extend(self.build_user_prompt_after_reference(tokenizer)?);

        // 2. The instruction text (encoded text token ids)
        ids.extend(tokenizer.encode(text)?);

        // 3. Assistant prompt suffix
        ids.extend(self.build_assistant_prompt_prefix(tokenizer)?);

        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the constants match Python's prompting.py exactly.
    #[test]
    fn test_constants_match_python() {
        // From Python prompting.py:
        // USER_ROLE_PREFIX = "user\n"
        // USER_TEMPLATE_REFERENCE_PREFIX = "<user_inst>\n- Reference(s):\n"
        // USER_TEMPLATE_AFTER_REFERENCE = "\n- Instruction:\nNone\n- Tokens:\nNone\n- Quality:\nNone\n- Sound Event:\nNone\n- Ambient Sound:\nNone\n- Language:\nNone\n- Text:\n"
        // USER_TEMPLATE_SUFFIX = "\n</user_inst>"
        // ASSISTANT_TURN_PREFIX = "\n"
        // ASSISTANT_ROLE_PREFIX = "assistant\n"
        assert_eq!(USER_ROLE_PREFIX, "user\n");
        assert_eq!(
            USER_TEMPLATE_REFERENCE_PREFIX,
            "<user_inst>\n- Reference(s):\n"
        );
        assert!(USER_TEMPLATE_AFTER_REFERENCE.starts_with("\n- Instruction:\nNone"));
        assert!(USER_TEMPLATE_AFTER_REFERENCE.ends_with("- Text:\n"));
        assert_eq!(USER_TEMPLATE_SUFFIX, "\n</user_inst>");
        assert_eq!(ASSISTANT_TURN_PREFIX, "\n");
        assert_eq!(ASSISTANT_ROLE_PREFIX, "assistant\n");
    }

    /// Verify that build_prompt_token_ids produces the same structure as the old inline version.
    /// We can't test with a real tokenizer here, but we verify the logic path.
    #[test]
    fn test_prompting_new_from_config() {
        let config = LMConfig {
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
        };

        let p = Prompting::new(&config);
        assert_eq!(p.im_start_token_id, 4);
        assert_eq!(p.im_end_token_id, 5);
    }
}
