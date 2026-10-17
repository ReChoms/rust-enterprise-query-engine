use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::info;

async fn download_file(url: &str, dest: &str) -> Result<String> {
    if !Path::new(dest).exists() {
        info!("Downloading {}...", dest);
        let resp = reqwest::get(url).await?;
        let bytes = resp.bytes().await?;
        std::fs::write(dest, bytes)?;
    }
    Ok(dest.to_string())
}

/// Downloads (if missing) and loads the BAAI/bge-base-en-v1.5 embedding model into memory.
pub async fn load_model() -> Result<(Arc<BertModel>, Arc<Tokenizer>)> {
    info!("Fetching safetensors and config...");

    let data_dir = std::env::var("MODEL_CACHE_DIR").unwrap_or_else(|_| "data".to_string());
    std::fs::create_dir_all(&data_dir)?;

    let config_path = download_file(
        "https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/config.json",
        &format!("{}/config.json", data_dir),
    ).await?;
    let tokenizer_path = download_file(
        "https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/tokenizer.json",
        &format!("{}/tokenizer.json", data_dir),
    ).await?;
    let weights_path = download_file(
        "https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/model.safetensors",
        &format!("{}/model.safetensors", data_dir),
    ).await?;

    info!("Building Tokenizer and Tensor Neural Network...");

    let config = std::fs::read_to_string(config_path)?;
    let config: Config = serde_json::from_str(&config)?;
    if config.hidden_size == 0 {
        anyhow::bail!("Invalid model config: hidden_size cannot be 0");
    }

    let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
    tokenizer.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        ..Default::default()
    }));

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &Device::Cpu)?
    };

    let model = BertModel::load(vb, &config)?;

    Ok((Arc::new(model), Arc::new(tokenizer)))
}

/// Contiguous flat embedding buffer (Cloudflare Zero-Copy Optimization).
#[derive(Debug, Clone)]
pub struct EmbeddingBatch {
    data: Box<[f32]>,
    dim: usize,
    count: usize,
}

impl EmbeddingBatch {
    #[inline]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    pub fn try_new(data: Box<[f32]>, dim: usize, count: usize) -> Result<Self> {
        let expected_len = count
            .checked_mul(dim)
            .ok_or_else(|| anyhow::anyhow!("Dimension overflow"))?;
        if data.len() != expected_len {
            anyhow::bail!("Length mismatch: expected {}, got {}", expected_len, data.len());
        }
        Ok(Self { data, dim, count })
    }

    #[inline]
    pub fn row(&self, idx: usize) -> Option<&[f32]> {
        if idx >= self.count || self.dim == 0 {
            return None;
        }
        let start = idx.checked_mul(self.dim)?;
        let end = start.checked_add(self.dim)?;
        self.data.get(start..end)
    }

    pub fn into_vecs(self) -> Vec<Vec<f32>> {
        if self.dim == 0 {
            return Vec::new();
        }
        (0..self.count)
            .filter_map(|i| self.row(i).map(|r| r.to_vec()))
            .collect()
    }
}

/// Normalizes a float vector slice in-place using Blue's Scaled Hypotenuse Algorithm.
///
/// ### Mathematical Immunity to Float Overflow / Underflow:
/// Direct squaring (x * x) can overflow to `INFINITY` if floats exceed ~1.8e19.
/// By scaling by `max_abs` first, every element is bounded within [-1.0, 1.0],
/// guaranteeing that squared numbers never overflow `f32::MAX`.
/// Precomputes `inv_norm` once so the loop uses single-cycle CPU multiplications.
fn normalize_vector_slice(slice: &mut [f32]) {
    let max_abs = slice.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    if max_abs == 0.0 || !max_abs.is_finite() {
        return;
    }
    let inv_max = 1.0 / max_abs;
    let scaled_sum: f32 = slice.iter().map(|&x| (x * inv_max).powi(2)).sum();
    let norm = max_abs * scaled_sum.sqrt();
    if norm <= f32::MIN_POSITIVE || !norm.is_finite() {
        return;
    }
    let inv_norm = 1.0 / norm;
    slice.iter_mut().for_each(|x| *x *= inv_norm);
}

/// Generates normalized dense vector embeddings in a contiguous flat buffer (Cloudflare zero-copy format).
#[tracing::instrument(name = "engine_candle_embeddings", skip(tokenizer, model), fields(batch_size = sentences.len()))]
pub async fn get_embeddings_batch(
    sentences: Vec<String>,
    tokenizer: Arc<Tokenizer>,
    model: Arc<BertModel>,
) -> Result<EmbeddingBatch> {
    let max_batch = std::env::var("MAX_EMBEDDING_BATCH_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_024);
    if sentences.len() > max_batch {
        anyhow::bail!(
            "Embedding batch size {} exceeds maximum allowed of {}",
            sentences.len(),
            max_batch
        );
    }

    if sentences.is_empty() {
        return EmbeddingBatch::try_new(Box::new([]), 768, 0);
    }

    let result = tokio::task::spawn_blocking(move || -> Result<EmbeddingBatch> {
        let inputs: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();
        let count = inputs.len();
        let tokens = tokenizer
            .encode_batch(inputs, true)
            .map_err(anyhow::Error::msg)?;

        let token_ids = tokens
            .iter()
            .map(|t| Tensor::new(t.get_ids(), &Device::Cpu))
            .collect::<Result<Vec<_>, _>>()?;
        let token_ids = Tensor::stack(&token_ids, 0)?;

        let token_type_ids = token_ids.zeros_like()?;
        let embeddings = model.forward(&token_ids, &token_type_ids, None)?;

        let cls_embeddings = embeddings.i((.., 0, ..))?;
        let dim = cls_embeddings.dim(1)?;
        let mut flat_data = cls_embeddings.flatten_all()?.to_vec1::<f32>()?;
        for chunk in flat_data.chunks_exact_mut(dim) {
            normalize_vector_slice(chunk);
        }

        EmbeddingBatch::try_new(flat_data.into_boxed_slice(), dim, count)
    })
    .await??;

    Ok(result)
}

/// Generates normalized dense vector embeddings using Candle BERT on dedicated CPU threads.
pub async fn get_embeddings(
    sentences: Vec<String>,
    tokenizer: Arc<Tokenizer>,
    model: Arc<BertModel>,
) -> Result<Vec<Vec<f32>>> {
    Ok(get_embeddings_batch(sentences, tokenizer, model).await?.into_vecs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_unit_vector() {
        let mut vec = vec![3.0, 4.0];
        normalize_vector_slice(&mut vec);
        assert_eq!(vec, vec![0.6, 0.8]);
        let new_mag = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((new_mag - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let mut vec = vec![0.0, 0.0, 0.0];
        normalize_vector_slice(&mut vec);
        assert_eq!(vec, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_embedding_batch_bounds_and_invariants() {
        let data = vec![1.0, 2.0, 3.0, 4.0].into_boxed_slice();
        let batch = EmbeddingBatch::try_new(data, 2, 2).unwrap();
        assert_eq!(batch.row(0), Some(&[1.0, 2.0][..]));
        assert_eq!(batch.row(1), Some(&[3.0, 4.0][..]));
        assert_eq!(batch.row(2), None);
        assert_eq!(batch.into_vecs(), vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        assert!(EmbeddingBatch::try_new(vec![1.0].into_boxed_slice(), 2, 2).is_err());

        let empty = EmbeddingBatch::try_new(Box::new([]), 768, 0).unwrap();
        assert_eq!(empty.row(0), None);
        assert!(empty.into_vecs().is_empty());
    }
}
