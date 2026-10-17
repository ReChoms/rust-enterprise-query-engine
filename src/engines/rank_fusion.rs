//! Altitude 1: Rank Fusion Engine

use std::collections::HashMap;

pub fn fuse_ranked_candidates(
    dense: &[&str],
    lexical: &[&str],
    k: f32,
) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();

    for (r, &id) in dense.iter().enumerate() {
        *scores.entry(id.to_string()).or_default() += 1.0 / (k + (r + 1) as f32);
    }

    for (r, &id) in lexical.iter().enumerate() {
        *scores.entry(id.to_string()).or_default() += 1.0 / (k + (r + 1) as f32);
    }

    let mut res: Vec<(String, f32)> = scores.into_iter().collect();
    res.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuse_ranked_candidates_scoring_and_ordering() {
        let dense = ["docA", "docB"];
        let lexical = ["docB", "docC"];
        let fused = fuse_ranked_candidates(&dense, &lexical, 60.0);
        assert_eq!(fused[0].0, "docB");
    }

    #[test]
    fn test_fuse_ranked_candidates_empty_inputs() {
        let fused = fuse_ranked_candidates(&[], &[], 60.0);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_fuse_ranked_candidates_tie_breaking_by_id() {
        let dense = ["docZ"];
        let lexical = ["docA"];
        let fused = fuse_ranked_candidates(&dense, &lexical, 60.0);
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].0, "docA");
        assert_eq!(fused[1].0, "docZ");
        assert_eq!(fused[0].1, fused[1].1);
    }
}
