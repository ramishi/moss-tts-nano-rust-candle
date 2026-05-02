use anyhow::Result;
use sentencepiece::{PieceWithId, SentencePieceProcessor};
use std::path::Path;

pub struct MossTTSNanoTokenizer {
    spp: SentencePieceProcessor,
}

impl MossTTSNanoTokenizer {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let spp = SentencePieceProcessor::open(path)
            .map_err(|e| anyhow::anyhow!("Failed to load SentencePiece model: {:?}", e))?;
        Ok(Self { spp })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let pieces: Vec<PieceWithId> = self
            .spp
            .encode(text)
            .map_err(|e| anyhow::anyhow!("Encode error: {:?}", e))?;
        Ok(pieces.iter().map(|p| p.id).collect())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.spp
            .decode_piece_ids(ids)
            .map_err(|e| anyhow::anyhow!("Decode error: {:?}", e))
    }
}
