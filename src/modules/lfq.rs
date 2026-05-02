//! LFQ (Lookup-Free Quantization) module for Audio Tokenizer.
//!
//! This module implements the Residual LFQ (Lookup-Free Quantization) used in the
//! MOSS Audio Tokenizer. It follows the exact implementation from the Python code.

use candle_core::{DType, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Linear, Module};

/// Weight-normalized Conv1d - used for in_proj and out_proj in LFQ.
/// This is effectively a 1x1 convolution which is equivalent to a linear layer
/// applied per time step.
pub struct WNConv1d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl WNConv1d {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x is (B, D_in, T)
        // weight is (D_out, D_in, 1) -> effectively (D_out, D_in)
        // We can reshape to linear:
        // x_trans = (B, T, D_in)
        // matmul with weight_trans = (D_in, D_out)
        // -> (B, T, D_out) -> transpose back to (B, D_out, T)

        let weight_2d = self.weight.squeeze(2)?; // (D_out, D_in)
        let linear = Linear::new(weight_2d, self.bias.clone());

        let x_trans = x.transpose(1, 2)?; // (B, T, D_in)
        let out = linear.forward(&x_trans)?; // (B, T, D_out)
        out.transpose(1, 2) // (B, D_out, T)
    }
}

/// LFQ (Lookup-Free Quantization) - single quantizer.
/// Matches Python: MossAudioTokenizerLFQ
pub struct LFQ {
    pub in_proj: Option<WNConv1d>,
    pub out_proj: Option<WNConv1d>,
    pub codebook: Embedding,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub input_dim: usize,
}

impl LFQ {
    pub fn new(
        in_proj: Option<WNConv1d>,
        out_proj: Option<WNConv1d>,
        codebook: Embedding,
        codebook_size: usize,
        codebook_dim: usize,
        input_dim: usize,
    ) -> Self {
        Self {
            in_proj,
            out_proj,
            codebook,
            codebook_size,
            codebook_dim,
            input_dim,
        }
    }

    /// Forward pass - matches Python: MossAudioTokenizerLFQ.forward()
    /// Input: z (B, input_dim, T)
    /// Output: (z_q, indices, z_e)
    ///   z_q: (B, input_dim, T) - quantized output after out_proj
    ///   indices: (B, T) - codebook indices
    ///   z_e: (B, codebook_dim, T) - encoder output before quantization
    pub fn forward(&self, z: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        // z = z.float()
        let z = z.to_dtype(DType::F32)?;

        // z_e = self.in_proj(z).float()
        let z_e = match &self.in_proj {
            Some(proj) => proj.forward(&z)?.to_dtype(DType::F32)?,
            None => z.clone(),
        };

        // z_q, indices = self.decode_latents(z_e)
        let (z_q, indices) = self.decode_latents(&z_e)?;

        // z_q = (z_e + (z_q - z_e).detach()).float()
        let diff = z_q.sub(&z_e)?;
        let z_q_ste = z_e.broadcast_add(&diff)?;
        let z_q_ste = z_q_ste.to_dtype(DType::F32)?;

        // z_q = self.out_proj(z_q).float()
        let z_q_out = match &self.out_proj {
            Some(proj) => proj.forward(&z_q_ste)?.to_dtype(DType::F32)?,
            None => z_q_ste,
        };
        Ok((z_q_out, indices, z_e))
    }

    /// Decode latents to codebook vectors - matches Python: MossAudioTokenizerLFQ.decode_latents()
    /// Input: latents (B, codebook_dim, T)
    /// Output: (z_q, indices)
    ///   z_q: (B, codebook_dim, T) - quantized vectors (before out_proj)
    ///   indices: (B, T) - codebook indices
    fn decode_latents(&self, latents: &Tensor) -> Result<(Tensor, Tensor)> {
        // encodings = latents.transpose(1, 2).reshape(-1, latents.shape[1]).float()
        let batch_size = latents.dims()[0];
        let dim = latents.dims()[1];
        let time = latents.dims()[2];

        let encodings = latents
            .transpose(1, 2)? // (B, T, dim)
            .reshape((batch_size * time, dim))? // (B*T, dim)
            .to_dtype(DType::F32)?;

        // codebook = self.codebook.weight.float()
        let codebook = self.codebook.embeddings().to_dtype(DType::F32)?; // (codebook_size, codebook_dim)

        // encodings = F.normalize(encodings)
        // codebook = F.normalize(codebook)
        let encodings_norm = self.l2_normalize(&encodings)?;
        let codebook_norm = self.l2_normalize(&codebook)?;

        // dist = (
        //     encodings.pow(2).sum(1, keepdim=True)
        //     - 2 * encodings @ codebook.t()
        //     + codebook.pow(2).sum(1, keepdim=True).t()
        // )
        let encodings_sq_sum = encodings_norm.sqr()?.sum_keepdim(1)?; // (B*T, 1)
        let codebook_sq_sum = codebook_norm.sqr()?.sum_keepdim(1)?; // (codebook_size, 1)
        let codebook_sq_sum_t = codebook_sq_sum.t()?; // (1, codebook_size)

        let dot = encodings_norm.matmul(&codebook_norm.t()?)?; // (B*T, codebook_size)
        let dot_scaled = dot.broadcast_mul(&Tensor::new(-2.0f32, dot.device())?)?;

        let dist = encodings_sq_sum
            .broadcast_add(&dot_scaled)?
            .broadcast_add(&codebook_sq_sum_t)?;

        // indices = (-dist).max(1)[1]
        let neg_dist = dist.neg()?;
        let indices = neg_dist.argmax(1)?; // (B*T,) or (B*T, 1) depending on Candle version
        // Squeeze if needed to ensure 1D
        let indices = if indices.dims().len() > 1 {
            indices.squeeze(1)?
        } else {
            indices
        };

        // indices = indices.reshape(latents.size(0), -1)
        let indices = indices.reshape((batch_size, time))?; // (B, T)

        // z_q = self.decode_code_wo_out_proj(indices).float()
        let z_q = self
            .decode_code_wo_out_proj(&indices)?
            .to_dtype(DType::F32)?;

        Ok((z_q, indices))
    }

    /// L2 normalize along last dimension - matches Python: F.normalize(x, p=2, dim=1)
    fn l2_normalize(&self, x: &Tensor) -> Result<Tensor> {
        // Compute L2 norm along last dimension (dim=1)
        let norm = x.sqr()?.sum_keepdim(1)?.sqrt()?;
        // Clamp to avoid division by zero
        let eps = 1e-12f32;
        let norm_clamped = norm.clamp(eps, f32::INFINITY)?;
        x.broadcast_div(&norm_clamped)
    }

    /// Decode code without out_proj - matches Python: decode_code_wo_out_proj
    /// Input: embed_id (B, T)
    /// Output: z_q (B, codebook_dim, T)
    fn decode_code_wo_out_proj(&self, embed_id: &Tensor) -> Result<Tensor> {
        // self.embed_code(embed_id).transpose(1, 2)
        self.embed_code(embed_id)?.transpose(1, 2)
    }

    /// Embed code - matches Python: embed_code
    /// Input: embed_id (B, T)
    /// Output: (B, T, codebook_dim)
    fn embed_code(&self, embed_id: &Tensor) -> Result<Tensor> {
        // F.embedding(embed_id, self.codebook.weight)
        self.codebook.forward(embed_id)
    }

    /// Encode and return indices and quantized vectors.
    /// Input: residual (B, rvq_dim, T)
    /// Output: (indices, z_q)
    ///   indices: (B, T)
    ///   z_q: (B, rvq_dim, T) - after out_proj
    pub fn encode(&self, residual: &Tensor) -> Result<(Tensor, Tensor)> {
        let (z_q, indices, _z_e) = self.forward(residual)?;
        Ok((indices, z_q))
    }
}

/// Residual LFQ - matches Python: MossAudioTokenizerResidualLFQ
pub struct ResidualLFQ {
    pub input_proj: Option<WNConv1d>,
    pub output_proj: Option<WNConv1d>,
    pub quantizers: Vec<LFQ>,
    pub rvq_dim: usize,
}

impl ResidualLFQ {
    pub fn new(
        input_proj: Option<WNConv1d>,
        output_proj: Option<WNConv1d>,
        quantizers: Vec<LFQ>,
        rvq_dim: usize,
    ) -> Self {
        Self {
            input_proj,
            output_proj,
            quantizers,
            rvq_dim,
        }
    }

    /// Returns the number of quantizers
    pub fn num_quantizers(&self) -> usize {
        self.quantizers.len()
    }

    /// Encode input latents to RVQ codes - matches Python forward() logic
    /// x: (B, input_dim, T)
    /// input_lengths: (B,) - valid lengths for each sample
    /// Output: codes (n_q, B, T)
    pub fn encode_codes(&self, x: &Tensor, input_lengths: Option<&Tensor>) -> Result<Tensor> {
        // z = self.input_proj(z).float()
        let z = match &self.input_proj {
            Some(proj) => proj.forward(x)?.to_dtype(DType::F32)?,
            None => x.to_dtype(DType::F32)?,
        };

        let batch_size = z.dims()[0];
        let max_time = z.dims()[2];

        // mask = torch.arange(max_time, device=z.device).expand(batch_size, max_time) < input_length.unsqueeze(1)
        let mask = if let Some(lengths) = input_lengths {
            let lengths_vec = lengths.to_vec1::<u32>()?;
            let mut mask_data = Vec::with_capacity(batch_size * max_time);
            for b in 0..batch_size {
                let valid_len = lengths_vec[b] as usize;
                for t in 0..max_time {
                    mask_data.push(if t < valid_len { 1.0f32 } else { 0.0f32 });
                }
            }
            Tensor::from_vec(mask_data, (batch_size, max_time), z.device())?
        } else {
            Tensor::ones((batch_size, max_time), DType::F32, z.device())?
        };

        // quantized_out = torch.zeros_like(z, dtype=torch.float32)
        let mut quantized_out = Tensor::zeros_like(&z)?.to_dtype(DType::F32)?;

        // residual = z.clone().float()
        let mut residual = z.clone();

        // all_indices = []
        let mut all_indices: Vec<Tensor> = Vec::with_capacity(self.num_quantizers());

        // for i, quantizer in enumerate(self.quantizers):
        for quantizer in &self.quantizers {
            // masked_residual = residual * mask.unsqueeze(1)
            let mask_expanded = mask.unsqueeze(1)?; // (B, 1, T)
            let masked_residual = residual.broadcast_mul(&mask_expanded)?;

            // z_q_i, indices_i, _ = quantizer(masked_residual.float())
            let (z_q_i, indices_i, _z_e) = quantizer.forward(&masked_residual)?;

            // update_mask = mask.unsqueeze(1)
            let update_mask = mask_expanded;

            // quantized_out = quantized_out + z_q_i * update_mask
            let z_q_masked = z_q_i.broadcast_mul(&update_mask)?;
            quantized_out = quantized_out.broadcast_add(&z_q_masked)?;

            // residual = residual - z_q_i * update_mask
            residual = residual.broadcast_sub(&z_q_masked)?;

            // all_indices.append(indices_i)
            all_indices.push(indices_i);
        }

        // all_indices = torch.stack(all_indices)  # (N, B, T)
        let mut stacked = all_indices[0].unsqueeze(0)?;
        for idx in &all_indices[1..] {
            stacked = Tensor::cat(&[&stacked, &idx.unsqueeze(0)?], 0)?;
        }

        // Note: output_proj is applied in the main forward, not here
        // quantized_out = self.output_proj(quantized_out.float()).float()

        // Suppress unused variable warning
        let _ = quantized_out;

        Ok(stacked) // (n_q, B, T)
    }

    /// Decode codes - matches Python: decode_codes
    /// codes: (n_q, B, T)
    /// Output: emb (B, output_dim, T)
    pub fn decode_codes(&self, codes: &Tensor) -> Result<Tensor> {
        let nq = codes.dims()[0];
        let batch_size = codes.dims()[1];
        let time = codes.dims()[2];

        // emb = torch.zeros(B, self.rvq_dim, T, device=codes.device, dtype=torch.float32)
        let mut emb = Tensor::zeros((batch_size, self.rvq_dim, time), DType::F32, codes.device())?;

        // for i, quantizer in enumerate(self.quantizers[:nq]):
        for i in 0..nq {
            let quantizer = &self.quantizers[i];
            // codes[i] is (B, T)
            let codes_i = codes.i(i)?;
            // quantized_i = quantizer.decode_code(codes[i]).float()
            let quantized_i = quantizer.decode_code(&codes_i)?.to_dtype(DType::F32)?;
            // emb += quantized_i
            emb = emb.broadcast_add(&quantized_i)?;
        }

        // emb = self.output_proj(emb.float()).float()
        match &self.output_proj {
            Some(proj) => proj.forward(&emb)?.to_dtype(DType::F32),
            None => Ok(emb),
        }
    }
}

impl LFQ {
    /// Decode code - matches Python: decode_code
    /// embed_id: (B, T)
    /// Output: z_q (B, input_dim, T) - after out_proj
    pub fn decode_code(&self, embed_id: &Tensor) -> Result<Tensor> {
        // z_q = self.decode_code_wo_out_proj(embed_id).float()
        let z_q = self
            .decode_code_wo_out_proj(embed_id)?
            .to_dtype(DType::F32)?;
        // z_q = self.out_proj(z_q).float()
        match &self.out_proj {
            Some(proj) => proj.forward(&z_q)?.to_dtype(DType::F32),
            None => Ok(z_q),
        }
    }
}
