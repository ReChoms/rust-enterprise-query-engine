use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use anyhow::Result;
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

    let mut tokenizer =
        Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

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

fn normalize_vector(mut vec: Vec<f32>) -> Vec<f32> {
    let magnitude = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        vec.iter_mut().for_each(|x| *x /= magnitude);
    }
    vec
}

pub async fn get_embeddings(
    sentences: Vec<String>,
    tokenizer: Arc<Tokenizer>,
    model: Arc<BertModel>,
) -> Result<Vec<Vec<f32>>> {
    let result = tokio::task::spawn_blocking(move || -> Result<Vec<Vec<f32>>> {
        let inputs: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();
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

        let raw_vecs = cls_embeddings.to_vec2::<f32>()?;
        let normalized_vecs = raw_vecs.into_iter().map(normalize_vector).collect();

        Ok(normalized_vecs)
    })
    .await??;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_unit_vector() {
        // A classic 3-4-5 triangle vector
        let vec = vec![3.0, 4.0];
        let normalized = normalize_vector(vec);
        
        // Magnitude is 5. Normalized should be 3/5 (0.6) and 4/5 (0.8)
        assert_eq!(normalized, vec![0.6, 0.8]);
        
        // Ensure new magnitude is precisely 1.0 (with float tolerance)
        let new_mag = normalized.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((new_mag - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let vec = vec![0.0, 0.0, 0.0];
        let normalized = normalize_vector(vec);
        // Should bypass division entirely and safely return the 0 vector
        assert_eq!(normalized, vec![0.0, 0.0, 0.0]);
    }
}
