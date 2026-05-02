use candle_core::{Device, Result, Tensor};

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    cos_cache: Tensor,
    sin_cache: Tensor,
    max_seq_len: usize,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, base: f32, device: &Device) -> Result<Self> {
        let half_dim = dim / 2;
        let mut inv_freq = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            let freq = 1.0 / base.powf((2 * i) as f32 / dim as f32);
            inv_freq.push(freq);
        }
        let inv_freq = Tensor::from_vec(inv_freq, (half_dim,), device)?;

        // Compute frequencies for max_seq_len
        let seq = (0..max_seq_len).map(|i| i as f32).collect::<Vec<_>>();
        let seq = Tensor::from_vec(seq, (max_seq_len,), device)?;

        // Freqs: (max_seq_len, half_dim)
        let freqs = seq.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;

        let cos_cache = freqs.cos()?;
        let sin_cache = freqs.sin()?;

        Ok(Self {
            cos_cache,
            sin_cache,
            max_seq_len,
        })
    }

    /// Input shape must be (batch, seq, heads, head_dim)
    pub fn forward(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(1)?;
        let end = offset + seq_len;

        if end > self.max_seq_len {
            candle_core::bail!("Sequence length {} exceeds max {}", end, self.max_seq_len);
        }

        // Narrow caches to sequence window
        let cos = self.cos_cache.narrow(0, offset, seq_len)?;
        let sin = self.sin_cache.narrow(0, offset, seq_len)?;

        let q_out = self.apply_rotary(q, &cos, &sin)?;
        let k_out = self.apply_rotary(k, &cos, &sin)?;

        Ok((q_out, k_out))
    }

    fn apply_rotary(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (batch, seq, heads, head_dim) = x.dims4()?;
        let half_dim = head_dim / 2;

        let x = x.reshape((batch, seq, heads, half_dim, 2))?;

        let xr = x.narrow(4, 0, 1)?.squeeze(4)?; // (B, S, H, D/2)
        let xi = x.narrow(4, 1, 1)?.squeeze(4)?; // (B, S, H, D/2)

        // Caches are (S, D/2), need to broadcast to (B, S, H, D/2)
        // reshape to (1, S, 1, D/2)
        let cos_b = cos.unsqueeze(0)?.unsqueeze(2)?;
        let sin_b = sin.unsqueeze(0)?.unsqueeze(2)?;

        let rotr_x = xr.broadcast_mul(&cos_b)?;
        let rotr_y = xi.broadcast_mul(&sin_b)?;
        let rotated_r = rotr_x.sub(&rotr_y)?;

        let roti_x = xr.broadcast_mul(&sin_b)?;
        let roti_y = xi.broadcast_mul(&cos_b)?;
        let rotated_i = roti_x.add(&roti_y)?;

        let rotated_r = rotated_r.unsqueeze(4)?;
        let rotated_i = rotated_i.unsqueeze(4)?;

        let stacked = Tensor::cat(&[&rotated_r, &rotated_i], 4)?;
        stacked.reshape((batch, seq, heads, head_dim))
    }
}
