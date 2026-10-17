//! Altitude 1: Heavy Compute Engines
//!
//! Contains hermetic compute, memory-mapped safetensors, Arrow zero-copy memory arrays,
//! and disk storage. Zero network/HTTP dependencies.

pub mod ast_guard;
pub mod cache;
pub mod embeddings;
pub mod rank_fusion;
pub mod sql;
pub mod vector;

pub use ast_guard::validate_sql_is_safe;
pub use cache::SemanticCache;
pub use embeddings::{get_embeddings, get_embeddings_batch, load_model, EmbeddingBatch};
pub use rank_fusion::fuse_ranked_candidates;
pub use sql::{execute_sql_query, init_datafusion, record_batches_to_json_lines, write_record_batches_as_json_lines};
pub use vector::{
    build_fallback_sql, check_lancedb_health, create_scalar_index, create_vector_index,
    execute_fallback_search, execute_semantic_search, insert_batch, is_vector_indexed,
};
