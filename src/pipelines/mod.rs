//! Altitude 2: Orchestration Pipelines
//!
//! Dual-stream workflows:
//! - Read Stream: `pipelines::query`
//! - Write Stream: `pipelines::ingest`

pub mod ingest;
pub mod query;

pub use ingest::execute_ingestion;
pub use query::{
    classify_intent, parse_llm_json, parse_question_intent, run_query_pipeline, run_semantic_pipeline,
    run_sql_pipeline, verify_and_parse_llm_generation, AppState, OllamaClient, QueryOutput,
    SemanticQueryResult,
};
