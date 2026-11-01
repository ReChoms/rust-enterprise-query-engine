//! Altitude 1: Cross-Encoder Re-Ranking Engine

use std::sync::Arc;
use candle_core::{Device, Tensor};
use candle_transformers::models::xlm_roberta::XLMRobertaForSequenceClassification;
use tokenizers::Tokenizer;

pub struct CrossEncoder {
    pub(crate) model: XLMRobertaForSequenceClassification,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) device: Device,
}

impl CrossEncoder {
    pub fn new(model: XLMRobertaForSequenceClassification, tokenizer: Tokenizer, device: Device) -> Self {
        Self { model, tokenizer, device }
    }

    pub fn rerank<T>(&self, query: &str, candidates: Vec<(T, Arc<str>)>, top_k: usize) -> anyhow::Result<Vec<(T, Arc<str>)>> {
        rerank_candidates_with_scorer(candidates, query, top_k, |q, p| self.score_pairs(q, p))
    }

    pub fn score_pairs(&self, query: &str, passages: &[&str]) -> anyhow::Result<Vec<f32>> {
        if passages.is_empty() {
            return Ok(Vec::new());
        }
        let inputs: Vec<(&str, &str)> = passages.iter().map(|p| (query, *p)).collect();
        let encodings = self.tokenizer.encode_batch(inputs, true).map_err(anyhow::Error::msg)?;
        let mut ids = Vec::with_capacity(encodings.len());
        let mut masks = Vec::with_capacity(encodings.len());
        for enc in &encodings {
            ids.push(Tensor::new(enc.get_ids(), &self.device)?);
            masks.push(Tensor::new(enc.get_attention_mask(), &self.device)?);
        }
        let input_ids = Tensor::stack(&ids, 0)?;
        let attention_mask = Tensor::stack(&masks, 0)?;
        let token_type_ids = input_ids.zeros_like()?;
        let logits = self.model.forward(&input_ids, &attention_mask, &token_type_ids)?;
        Ok(logits.flatten_all()?.to_vec1::<f32>()?)
    }
}

pub async fn load_cross_encoder(device: Device) -> anyhow::Result<Arc<CrossEncoder>> {
    let model_directory = std::env::var("RERANKER_MODEL_DIR").unwrap_or_else(|_| "data/reranker".to_string());
    std::fs::create_dir_all(&model_directory)?;
    let base_url = "https://huggingface.co/BAAI/bge-reranker-base/resolve/main";
    let config_path = crate::engines::embeddings::download_file(
        &format!("{base_url}/config.json"),
        &format!("{model_directory}/config.json"),
    ).await?;
    let tokenizer_path = crate::engines::embeddings::download_file(
        &format!("{base_url}/tokenizer.json"),
        &format!("{model_directory}/tokenizer.json"),
    ).await?;
    let weights_path = crate::engines::embeddings::download_file(
        &format!("{base_url}/model.safetensors"),
        &format!("{model_directory}/model.safetensors"),
    ).await?;

    let config_json = std::fs::read_to_string(config_path)?;
    let config: candle_transformers::models::xlm_roberta::Config = serde_json::from_str(&config_json)?;
    let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
    tokenizer.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        ..Default::default()
    }));
    let var_builder = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &device)?
    };
    let model = XLMRobertaForSequenceClassification::new(1, &config, var_builder)?;
    Ok(Arc::new(CrossEncoder::new(model, tokenizer, device)))
}

pub fn rerank_by_scores<T>(
    candidates: Vec<(T, Arc<str>)>,
    scores: &[f32],
    top_k: usize,
) -> Vec<(T, Arc<str>)> {
    let mut paired: Vec<_> = candidates.into_iter().zip(scores.iter().copied()).collect();
    paired.sort_by(|(_, score_a), (_, score_b)| score_b.total_cmp(score_a));
    paired.truncate(top_k);
    paired.into_iter().map(|(candidate, _)| candidate).collect()
}

pub fn rerank_candidates_with_scorer<T, F>(
    candidates: Vec<(T, Arc<str>)>,
    query: &str,
    top_k: usize,
    scorer: F,
) -> anyhow::Result<Vec<(T, Arc<str>)>>
where
    F: FnOnce(&str, &[&str]) -> anyhow::Result<Vec<f32>>,
{
    if candidates.is_empty() || top_k == 0 {
        return Ok(Vec::new());
    }
    let passages: Vec<&str> = candidates.iter().map(|(_, c)| c.as_ref()).collect();
    let scores = scorer(query, &passages)?;
    if scores.len() != candidates.len() {
        anyhow::bail!(
            "Scorer returned {} scores for {} candidates",
            scores.len(),
            candidates.len()
        );
    }
    Ok(rerank_by_scores(candidates, &scores, top_k))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rerank_by_scores_orders_descending_and_truncates() {
        let candidates = vec![
            ("id_low", Arc::from("low score chunk")),
            ("id_high", Arc::from("high score chunk")),
            ("id_mid", Arc::from("mid score chunk")),
        ];
        let scores = [-1.5, 4.8, 1.2];
        let reranked = rerank_by_scores(candidates, &scores, 2);
        assert_eq!(reranked.len(), 2);
        assert_eq!(reranked[0].0, "id_high");
        assert_eq!(reranked[1].0, "id_mid");
    }

    #[test]
    fn test_rerank_handles_nan_with_total_cmp() {
        let candidates = vec![
            ("id_low", Arc::from("low score chunk")),
            ("id_high", Arc::from("high score chunk")),
            ("id_nan", Arc::from("nan score chunk")),
        ];
        let scores = [-5.0, 10.0, f32::NAN];
        let reranked = rerank_by_scores(candidates, &scores, 3);
        assert_eq!(reranked[0].0, "id_nan");
        assert_eq!(reranked[1].0, "id_high");
        assert_eq!(reranked[2].0, "id_low");
    }

    #[test]
    fn test_rerank_candidates_with_scorer_orders_and_truncates() {
        let candidates = vec![
            ("id_low", Arc::from("low score passage")),
            ("id_high", Arc::from("high score passage")),
            ("id_mid", Arc::from("mid score passage")),
        ];
        let result = rerank_candidates_with_scorer(candidates, "query text", 2, |_query, passages| {
            assert_eq!(passages.len(), 3);
            Ok(vec![0.1, 0.9, 0.5])
        })
        .unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, "id_high");
        assert_eq!(result[1].0, "id_mid");
    }
}
