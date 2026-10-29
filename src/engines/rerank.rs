//! Altitude 1: Cross-Encoder Re-Ranking Engine

use std::sync::Arc;

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
}
