use super::attention::{KVCache, MultiHeadAttention};
use super::rotary::RotaryEmbedding;
use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module};

#[derive(Debug)]
pub struct LayerScale {
    pub scale: Tensor,
}

impl LayerScale {
    pub fn new(scale: Tensor) -> Self {
        Self { scale }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_mul(&self.scale)
    }
}

pub struct MLP {
    pub fc1: Linear,
    pub fc2: Linear,
}

impl MLP {
    pub fn new(fc1: Linear, fc2: Linear) -> Self {
        Self { fc1, fc2 }
    }

    /// Exact GELU implementation matching PyTorch's GELU(approximate='none').
    /// GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    fn gelu_exact(x: &Tensor) -> Result<Tensor> {
        let x_div_sqrt2 =
            x.broadcast_div(&Tensor::new(std::f64::consts::SQRT_2 as f32, x.device())?)?;
        let erf = x_div_sqrt2.erf()?;
        let one = Tensor::new(1.0f32, x.device())?;
        let half = Tensor::new(0.5f32, x.device())?;
        let phi = (erf.broadcast_add(&one)?).broadcast_mul(&half)?;
        x.broadcast_mul(&phi)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = Self::gelu_exact(&x)?;
        self.fc2.forward(&x)
    }
}

pub struct TransformerLayer {
    pub attn: MultiHeadAttention,
    pub norm1: candle_nn::LayerNorm,
    pub norm2: candle_nn::LayerNorm,
    pub ffn: MLP,
    pub ls1: Option<LayerScale>,
    pub ls2: Option<LayerScale>,
}

impl TransformerLayer {
    pub fn forward(
        &self,
        x: &Tensor,
        rotary: Option<&RotaryEmbedding>,
        kv_cache: Option<&mut KVCache>,
        causal_mask: bool,
        input_lengths: Option<&Tensor>, // (B,)
    ) -> Result<Tensor> {
        let mut residual = x.clone();
        let mut h = self.norm1.forward(x)?;

        h = self
            .attn
            .forward(&h, rotary, kv_cache, causal_mask, input_lengths)?;
        if let Some(ls) = &self.ls1 {
            h = ls.forward(&h)?;
        }

        h = (residual + h)?;
        residual = h.clone();

        h = self.norm2.forward(&h)?;
        h = self.ffn.forward(&h)?;

        if let Some(ls) = &self.ls2 {
            h = ls.forward(&h)?;
        }

        residual + h
    }
}
