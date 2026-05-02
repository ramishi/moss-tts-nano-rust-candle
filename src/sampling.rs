use candle_core::{DType, Result, Tensor};
use rand::distributions::{Distribution, WeightedIndex};
use rand::{SeedableRng, rngs::StdRng};

pub struct Sampler {
    pub do_sample: bool,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    /// Internal RNG. Uses `StdRng` so results are reproducible when seeded.
    rng: StdRng,
}

impl Sampler {
    pub fn new(
        do_sample: bool,
        temperature: f64,
        top_p: f64,
        top_k: usize,
        seed: Option<u64>,
    ) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self {
            do_sample,
            temperature,
            top_p,
            top_k,
            rng,
        }
    }

    /// Create a greedy sampler (do_sample=false, uses argmax)
    pub fn greedy(seed: Option<u64>) -> Self {
        Self::new(false, 1.0, 1.0, 0, seed)
    }

    /// Create a sampling sampler with given parameters
    pub fn sampling(temperature: f64, top_p: f64, top_k: usize, seed: Option<u64>) -> Self {
        Self::new(true, temperature, top_p, top_k, seed)
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        let logits = logits.flatten_all()?.to_dtype(DType::F32)?;
        let logits_v = logits
            .to_device(&candle_core::Device::Cpu)?
            .to_vec1::<f32>()?;

        // When do_sample is false, use greedy decoding (argmax)
        // This matches Python's behavior: if not do_sample: return _argmax(values)
        if !self.do_sample {
            return Self::argmax_from_vec(&logits_v);
        }

        // When do_sample is true, temperature must be positive
        // This matches Python's: if not (temperature > 0): raise ValueError
        if self.temperature.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return Err(candle_core::Error::Msg(
                "temperature must be positive when do_sample=true".to_string(),
            ));
        }

        // Apply temperature scaling
        let mut scores: Vec<f32> = logits_v
            .iter()
            .map(|&v| v / self.temperature as f32)
            .collect();

        // top-k: keep only top_k highest probability tokens
        // Python: if top_k > 0 and top_k < scores.shape[0]: set others to -inf
        if self.top_k > 0 && self.top_k < scores.len() {
            let mut indexed_scores: Vec<(usize, f32)> =
                scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
            indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let threshold = indexed_scores[self.top_k - 1].1;
            for score in &mut scores {
                if *score < threshold {
                    *score = f32::NEG_INFINITY;
                }
            }
        }

        // top-p (nucleus sampling): keep tokens until cumulative probability > top_p
        // Python: if top_p > 0 and top_p < 1: apply nucleus filtering
        if self.top_p > 0.0 && self.top_p < 1.0 {
            let mut indexed_scores: Vec<(usize, f32)> =
                scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
            indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let sorted_probs =
                Self::softmax(&indexed_scores.iter().map(|&(_, s)| s).collect::<Vec<_>>());
            let mut cumulative = 0.0;
            let mut remove_mask = vec![false; indexed_scores.len()];
            for (index, &prob) in sorted_probs.iter().enumerate() {
                cumulative += prob;
                if cumulative > self.top_p as f32 {
                    remove_mask[index] = true;
                }
            }
            // Shift mask right by 1 (Python behavior)
            for index in (1..remove_mask.len()).rev() {
                remove_mask[index] = remove_mask[index - 1];
            }
            remove_mask[0] = false;
            for (index, &should_remove) in remove_mask.iter().enumerate() {
                if should_remove {
                    scores[indexed_scores[index].0] = f32::NEG_INFINITY;
                }
            }
        }

        // Compute probabilities
        let probabilities = Self::softmax(&scores);

        // Sample from the probability distribution using the seeded RNG
        let dist = WeightedIndex::new(&probabilities)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        Ok(dist.sample(&mut self.rng) as u32)
    }

    /// Compute softmax probabilities from scores
    fn softmax(scores: &[f32]) -> Vec<f32> {
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        if sum > 0.0 {
            exp_scores.iter().map(|&e| e / sum).collect()
        } else {
            // Fallback: uniform distribution over non-neg-inf
            let valid_count = scores.iter().filter(|&&s| s > f32::NEG_INFINITY).count();
            scores
                .iter()
                .map(|&s| {
                    if s > f32::NEG_INFINITY {
                        1.0 / valid_count as f32
                    } else {
                        0.0
                    }
                })
                .collect()
        }
    }

    /// Greedy decoding: return the index of the maximum value
    fn argmax_from_vec(values: &[f32]) -> Result<u32> {
        let mut max_idx = 0;
        for (i, &v) in values.iter().enumerate() {
            if v > values[max_idx] {
                max_idx = i;
            }
        }
        Ok(max_idx as u32)
    }
}

/// Apply repetition penalty to logits based on previously generated tokens.
/// Matches Python's _apply_repetition_penalty:
///   - For tokens that appeared before: if logit < 0, multiply by penalty; if logit >= 0, divide by penalty.
///   - penalty > 1.0 discourages repetition; penalty == 1.0 is a no-op.
pub fn apply_repetition_penalty(logits: &mut [f32], previous_token_ids: &[u32], penalty: f32) {
    if penalty <= 0.0 || penalty == 1.0 || previous_token_ids.is_empty() {
        return;
    }
    // Collect unique token ids that are within vocabulary range
    let vocab_size = logits.len() as u32;
    let mut seen = std::collections::HashSet::new();
    for &tid in previous_token_ids {
        if tid < vocab_size {
            seen.insert(tid as usize);
        }
    }
    // Apply penalty
    for &idx in &seen {
        let logit = logits[idx];
        if logit < 0.0 {
            logits[idx] = logit * penalty;
        } else {
            logits[idx] = logit / penalty;
        }
    }
}
