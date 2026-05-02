use super::attention::KVCache;
use super::rotary::RotaryEmbedding;
use super::transformer::TransformerLayer;
use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module};

pub struct ProjectedTransformer {
    pub input_proj: Linear,
    pub layers: Vec<TransformerLayer>,
    pub output_proj: Linear,
}

impl ProjectedTransformer {
    pub fn new(input_proj: Linear, layers: Vec<TransformerLayer>, output_proj: Linear) -> Self {
        Self {
            input_proj,
            layers,
            output_proj,
        }
    }

    pub fn downsample_ratio(&self) -> usize {
        1
    }

    pub fn forward(
        &self,
        x: &Tensor,
        rotary: Option<&RotaryEmbedding>,
        mut kv_caches: Option<&mut Vec<KVCache>>,
        causal_mask: bool,
        input_lengths: Option<&Tensor>, // (B,) - valid sequence lengths
    ) -> Result<Tensor> {
        // x is (B, D, T)
        let mut x = x.transpose(1, 2)?.contiguous()?; // -> (B, T, D)

        // input_proj
        x = self.input_proj.forward(&x)?;

        // passes through layers
        for (i, layer) in self.layers.iter().enumerate() {
            let cache = kv_caches.as_mut().map(|caches| &mut caches[i]);
            x = layer.forward(&x, rotary, cache, causal_mask, input_lengths)?;
        }

        // output_proj
        x = self.output_proj.forward(&x)?;

        // -> (B, D_out, T)
        x.transpose(1, 2)?.contiguous()
    }
}
