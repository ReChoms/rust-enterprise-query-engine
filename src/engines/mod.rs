//! Altitude 1: Heavy Compute Engines
//!
//! Contains hermetic compute, memory-mapped safetensors, Arrow zero-copy memory arrays,
//! and disk storage. Zero network/HTTP dependencies.

pub mod ast_guard;
pub mod embeddings;
pub mod sql;
pub mod vector;

pub use ast_guard::validate_sql_is_safe;
pub use embeddings::{get_embeddings, load_model};
pub use sql::{execute_sql_query, init_datafusion, record_batches_to_json_lines, write_record_batches_as_json_lines};
pub use vector::{
    build_fallback_sql, check_lancedb_health, execute_fallback_search, execute_semantic_search,
    insert_batch,
};
