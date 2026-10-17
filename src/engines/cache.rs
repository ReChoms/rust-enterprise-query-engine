//! Altitude 1: In-Memory Semantic Cache
use std::sync::RwLock;
use std::time::Instant;

pub struct CacheEntry {
    pub query: String,
    pub embedding: Box<[f32]>,
    pub response: Box<[u8]>,
    pub last_accessed: Instant,
}

pub struct SemanticCache {
    pub entries: RwLock<Vec<CacheEntry>>,
    pub capacity: usize,
    pub threshold: f32,
}

impl SemanticCache {
    pub fn new(capacity: usize, threshold: f32) -> Self {
        Self { entries: RwLock::new(Vec::with_capacity(capacity)), capacity, threshold }
    }
    pub fn get_exact(&self, query: &str) -> Option<Box<[u8]>> {
        let mut entries = self.entries.write().ok()?;
        let norm = query.trim();
        for entry in entries.iter_mut() {
            if entry.query.trim().eq_ignore_ascii_case(norm) {
                entry.last_accessed = Instant::now();
                return Some(entry.response.clone());
            }
        }
        None
    }
    pub fn get_similar(&self, query_emb: &[f32]) -> Option<Box<[u8]>> {
        let mut entries = self.entries.write().ok()?;
        let mut best_sim = -1.0f32;
        let mut best_idx: Option<usize> = None;
        for (idx, entry) in entries.iter().enumerate() {
            if entry.embedding.len() == query_emb.len() {
                let dot: f32 = entry.embedding.iter().zip(query_emb).map(|(a, b)| a * b).sum();
                if dot > best_sim {
                    best_sim = dot;
                    best_idx = Some(idx);
                }
            }
        }
        if best_sim >= self.threshold {
            if let Some(idx) = best_idx {
                entries[idx].last_accessed = Instant::now();
                return Some(entries[idx].response.clone());
            }
        }
        None
    }
    pub fn insert(&self, query: String, embedding: Box<[f32]>, response: Box<[u8]>) {
        let mut entries = match self.entries.write() { Ok(e) => e, Err(_) => return };
        if entries.len() >= self.capacity && !entries.is_empty() {
            if let Some((idx, _)) = entries.iter().enumerate().min_by_key(|(_, e)| e.last_accessed) {
                entries.remove(idx);
            }
        }
        entries.push(CacheEntry { query, embedding, response, last_accessed: Instant::now() });
    }
    pub fn len(&self) -> usize { self.entries.read().map(|e| e.len()).unwrap_or(0) }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_insert_and_exact_get() {
        let cache = SemanticCache::new(2, 0.95);
        cache.insert("query A".into(), Box::new([1.0, 0.0]), Box::new([1, 2, 3]));
        let hit = cache.get_exact("query A");
        assert_eq!(hit.as_deref(), Some(&[1, 2, 3][..]));
    }

    #[test]
    fn test_cache_semantic_match_threshold() {
        let cache = SemanticCache::new(2, 0.95);
        cache.insert("query A".into(), Box::new([1.0, 0.0]), Box::new([1, 2, 3]));
        let similar_vec = [0.99, 0.141];
        let hit = cache.get_similar(&similar_vec);
        assert_eq!(hit.as_deref(), Some(&[1, 2, 3][..]));
        let dissimilar_vec = [0.0, 1.0];
        let miss = cache.get_similar(&dissimilar_vec);
        assert_eq!(miss, None);
    }

    #[test]
    fn test_cache_lru_capacity_eviction() {
        let cache = SemanticCache::new(2, 0.95);
        cache.insert("Q1".into(), Box::new([1.0, 0.0]), Box::new([1]));
        cache.insert("Q2".into(), Box::new([0.0, 1.0]), Box::new([2]));
        assert_eq!(cache.len(), 2);
        assert!(cache.get_exact("Q1").is_some());
        cache.insert("Q3".into(), Box::new([0.7, 0.7]), Box::new([3]));
        assert_eq!(cache.len(), 2);
        assert!(cache.get_exact("Q1").is_some());
        assert!(cache.get_exact("Q2").is_none());
        assert!(cache.get_exact("Q3").is_some());
    }
}
