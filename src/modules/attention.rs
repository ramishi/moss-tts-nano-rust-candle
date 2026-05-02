use super::rotary::RotaryEmbedding;
use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module};

#[derive(Debug)]
pub struct KVCache {
    pub k_cache: Option<Tensor>,
    pub v_cache: Option<Tensor>,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            k_cache: None,
            v_cache: None,
        }
    }

    pub fn seq_len(&self) -> usize {
        self.k_cache
            .as_ref()
            .map(|k| k.dim(2).unwrap_or(0))
            .unwrap_or(0)
    }

    pub fn update(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (k_out, v_out) = match (&self.k_cache, &self.v_cache) {
            (Some(kc), Some(vc)) => {
                let k_new = Tensor::cat(&[kc, k], 2)?;
                let v_new = Tensor::cat(&[vc, v], 2)?;
                (k_new, v_new)
            }
            _ => (k.clone(), v.clone()),
        };

        self.k_cache = Some(k_out.clone());
        self.v_cache = Some(v_out.clone());

        Ok((k_out, v_out))
    }
}

pub struct MultiHeadAttention {
    pub in_proj: Linear,
    pub o_proj: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale: f64,
    pub context: Option<usize>,
}

impl MultiHeadAttention {
    pub fn new(
        in_proj: Linear,
        o_proj: Linear,
        num_heads: usize,
        head_dim: usize,
        context: Option<usize>,
    ) -> Self {
        let scale = 1.0 / (head_dim as f64).sqrt();
        Self {
            in_proj,
            o_proj,
            num_heads,
            head_dim,
            scale,
            context,
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        rotary: Option<&RotaryEmbedding>,
        kv_cache: Option<&mut KVCache>,
        causal_mask: bool,
        input_lengths: Option<&Tensor>, // (B,) - valid sequence lengths for each batch item
    ) -> Result<Tensor> {
        let (batch, seq, hidden_size) = x.dims3()?;

        let qkv = self.in_proj.forward(x)?;

        let q = qkv.narrow(2, 0, hidden_size)?;
        let k = qkv.narrow(2, hidden_size, hidden_size)?;
        let v = qkv.narrow(2, hidden_size * 2, hidden_size)?;

        let q = q.reshape((batch, seq, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch, seq, self.num_heads, self.head_dim))?;
        let v = v.reshape((batch, seq, self.num_heads, self.head_dim))?;

        let (q, k) = if let Some(rope) = rotary {
            let offset = kv_cache.as_ref().map(|c| c.seq_len()).unwrap_or(0);
            rope.forward(&q, &k, offset)?
        } else {
            (q, k)
        };

        // transpose to (B, Heads, Seq, Head_Dim)
        let q = q.transpose(1, 2)?.contiguous()?;
        let mut k = k.transpose(1, 2)?.contiguous()?;
        let mut v = v.transpose(1, 2)?.contiguous()?;

        if let Some(cache) = kv_cache {
            let (k_new, v_new) = cache.update(&k, &v)?;
            k = k_new;
            v = v_new;
        }

        // Attention weights
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let mut attn_weights = q.matmul(&k_t)?;
        attn_weights = (attn_weights * self.scale)?;

        if causal_mask || input_lengths.is_some() {
            let mask = self.create_mask(seq, k.dim(2)?, causal_mask, input_lengths, x.device())?;
            attn_weights = attn_weights.broadcast_add(&mask)?;
        }

        let attn_probs = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;
        let attn_output = attn_probs.matmul(&v)?;

        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((batch, seq, self.num_heads * self.head_dim))?;

        // Zero out padded query positions to match Python behavior:
        // Python: out = torch.where(valid_q, out, torch.zeros(...))
        // This prevents NaN/invalid values from leaking into later layers
        let attn_output = if let Some(lengths) = input_lengths {
            let lengths_vec = lengths.to_vec1::<u32>()?;
            if !lengths_vec.is_empty() {
                let valid_len = lengths_vec[0] as usize;
                if valid_len < seq {
                    // Create a mask: 1.0 for valid positions, 0.0 for padded positions
                    let mut mask_data = vec![1.0f32; seq];
                    for i in valid_len..seq {
                        mask_data[i] = 0.0;
                    }
                    let mask = Tensor::from_vec(mask_data, (1, seq, 1), x.device())?;
                    attn_output.broadcast_mul(&mask)?
                } else {
                    attn_output
                }
            } else {
                attn_output
            }
        } else {
            attn_output
        };

        self.o_proj.forward(&attn_output)
    }

    fn create_mask(
        &self,
        q_len: usize,
        kv_len: usize,
        causal_mask: bool,
        input_lengths: Option<&Tensor>,
        device: &Device,
    ) -> Result<Tensor> {
        let mut mask = Vec::with_capacity(q_len * kv_len);
        let shift = kv_len.saturating_sub(q_len);

        // Get input_lengths as Vec<usize> if provided
        let lengths: Option<Vec<usize>> = input_lengths
            .map(|t| {
                t.to_vec1::<u32>()
                    .map(|v| v.iter().map(|&x| x as usize).collect())
                    .ok()
            })
            .flatten();

        for i in 0..q_len {
            for j in 0..kv_len {
                let mut allow = true;

                // Causal mask: can only attend to positions <= current position
                if causal_mask && j > i + shift {
                    allow = false;
                }

                // Context window: can only attend to positions within context window
                if allow && causal_mask {
                    if let Some(ctx) = self.context {
                        let delta = (i + shift).saturating_sub(j);
                        if delta >= ctx {
                            allow = false;
                        }
                    }
                }

                // Valid key mask: key position must be < input_lengths
                if allow && j < kv_len {
                    if let Some(ref lens) = lengths {
                        // For batch item 0 (we only support B=1 for now)
                        if lens.len() > 0 && j >= lens[0] {
                            allow = false;
                        }
                    }
                }

                if allow {
                    mask.push(0.0f32);
                } else {
                    mask.push(f32::NEG_INFINITY);
                }
            }
        }

        Tensor::from_vec(mask, (1, 1, q_len, kv_len), device)
    }

    #[allow(dead_code)]
    fn create_causal_mask(&self, q_len: usize, kv_len: usize, device: &Device) -> Result<Tensor> {
        let mut mask = Vec::with_capacity(q_len * kv_len);
        let shift = kv_len.saturating_sub(q_len);

        for i in 0..q_len {
            for j in 0..kv_len {
                let mut allow = false;
                if j <= i + shift {
                    // causal matches
                    allow = true;
                    if let Some(ctx) = self.context {
                        let delta = (i + shift).saturating_sub(j);
                        if delta >= ctx {
                            allow = false;
                        }
                    }
                }

                if allow {
                    mask.push(0.0);
                } else {
                    mask.push(f32::NEG_INFINITY);
                }
            }
        }

        Tensor::from_vec(mask, (1, 1, q_len, kv_len), device)
    }
}
