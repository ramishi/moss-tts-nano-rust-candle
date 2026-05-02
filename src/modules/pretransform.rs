use candle_core::{Result, Tensor};

#[derive(Debug, Clone)]
pub struct PatchedPretransform {
    patch_size: usize,
    is_downsample: bool,
}

impl PatchedPretransform {
    pub fn new(patch_size: usize, is_downsample: bool) -> Self {
        Self {
            patch_size,
            is_downsample,
        }
    }

    pub fn downsample_ratio(&self) -> usize {
        self.patch_size
    }

    /// Encode (Downsample): (B, D, T) -> (B, D * patch_size, T / patch_size)
    /// If T is not divisible by patch_size, the input is zero-padded to the next multiple.
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, dim, seq) = x.dims3()?;
        let out_seq = seq.div_ceil(self.patch_size);
        let padded_seq = out_seq * self.patch_size;

        // Pad if necessary: (B, D, seq) -> (B, D, padded_seq)
        let x = if padded_seq > seq {
            // Build padding tensor: (B, D, padded_seq - seq)
            let pad_amount = padded_seq - seq;
            let padding = Tensor::zeros((batch, dim, pad_amount), x.dtype(), x.device())?;
            Tensor::cat(&[x, &padding], 2)?
        } else {
            x.clone()
        };

        // Python: x = x.view(B, D, -1, P)
        let x = x.reshape((batch, dim, out_seq, self.patch_size))?;
        // Python: x = x.transpose(2, 3).reshape(B, D * P, -1)
        x.transpose(2, 3)?
            .contiguous()?
            .reshape((batch, dim * self.patch_size, out_seq))
    }

    /// Decode (Upsample): (B, D * patch_size, T) -> (B, D, T * patch_size)
    pub fn decode(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, dim, seq) = x.dims3()?;
        let out_dim = dim / self.patch_size;

        // Python: x = x.view(B, D, P, S)
        let x = x.reshape((batch, out_dim, self.patch_size, seq))?;
        // Python: x = x.transpose(2, 3).reshape(B, D, -1)
        x.transpose(2, 3)?
            .contiguous()?
            .reshape((batch, out_dim, seq * self.patch_size))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.is_downsample {
            self.encode(x)
        } else {
            self.decode(x)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_encode_decode_parity() -> Result<()> {
        let device = Device::Cpu;
        let patch_size = 2;
        let b = 1;
        let d = 2; // depth
        let t = 4; // time

        // Shape: (1, 2, 4)
        // [ [ [1, 2, 3, 4],
        //     [5, 6, 7, 8] ] ]
        let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let input = Tensor::from_slice(&data, (b, d, t), &device)?;

        let encoder = PatchedPretransform::new(patch_size, true);
        let decoder = PatchedPretransform::new(patch_size, false);

        let encoded = encoder.forward(&input)?;
        // Shape should be (1, 2*2, 4/2) = (1, 4, 2)
        // Reshape1 (1, 2, 2, 2)
        // Permute to (1, 2, 2, 2)
        // Result manually:
        // D0: [1, 3], [2, 4]
        // D1: [5, 7], [6, 8]
        assert_eq!(encoded.dims(), &[1, 4, 2]);
        let enc_data = encoded.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(enc_data, vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0]);

        let decoded = decoder.forward(&encoded)?;
        assert_eq!(decoded.dims(), &[1, 2, 4]);

        let dec_data = decoded.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(dec_data, data);

        Ok(())
    }
}
