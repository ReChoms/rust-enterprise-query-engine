//! Altitude 2: Query Read River
//!
//! Orchestrates natural language understanding, LLM prompt generation,
//! AST-validated SQL execution, hybrid semantic search, and anti-hallucination verification.

pub mod firewall;
pub mod intent;
pub mod ollama;
pub mod prompts;
pub mod router;

use anyhow::{bail, Result};
use candle_transformers::models::bert::BertModel;
use datafusion::arrow::array::RecordBatch;
use datafusion::prelude::SessionContext;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::{info, warn};

use crate::common::types::{DegradedChunk, DegradedResponse, ParsedQuestion, SemanticResponse, SqlResponse};
use crate::engines::sql::execute_sql_query;
use crate::engines::vector::execute_fallback_search;

pub use firewall::{parse_llm_json, verify_and_parse_llm_generation};
pub use intent::parse_question_intent;
pub use ollama::OllamaClient;
pub use prompts::{
    build_question_parser_prompt, build_routing_prompt, build_semantic_prompt, build_sql_prompt,
    SAP_KNA1_SCHEMA,
};
pub use router::classify_intent;

/// Shared application state containing read-only singleton models and contexts.
#[derive(Clone)]
pub struct AppState {
    pub llm_client: Arc<OllamaClient>,
    pub sql_engine: Arc<SessionContext>,
    pub embedding_model: Arc<BertModel>,
    pub tokenizer: Arc<Tokenizer>,
    pub vector_db_uri: String,
    pub semantic_cache: Arc<crate::engines::SemanticCache>,
}

impl AppState {
    pub fn new(
        llm_client: Arc<OllamaClient>,
        sql_engine: Arc<SessionContext>,
        embedding_model: Arc<BertModel>,
        tokenizer: Arc<Tokenizer>,
        vector_db_uri: String,
    ) -> Self {
        let capacity = std::env::var("SEMANTIC_CACHE_CAPACITY").ok().and_then(|v| v.parse().ok()).unwrap_or(1_024);
        let threshold = std::env::var("SEMANTIC_CACHE_THRESHOLD").ok().and_then(|v| v.parse().ok()).unwrap_or(0.95);
        Self {
            llm_client,
            sql_engine,
            embedding_model,
            tokenizer,
            vector_db_uri,
            semantic_cache: Arc::new(crate::engines::SemanticCache::new(capacity, threshold)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SemanticQueryResult {
    Verified(SemanticResponse),
    Degraded(DegradedResponse),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum QueryOutput {
    Sql(Vec<String>),
    Semantic(SemanticQueryResult),
}

/// Dynamic Intent Router Pipeline: classifies query into SQL vs Semantic and dispatches.
#[tracing::instrument(name = "pipeline_query_dispatch", skip(state))]
pub async fn run_query_pipeline(state: &AppState, query: &str) -> Result<QueryOutput> {
    let decision = classify_intent(&state.llm_client, query).await?;
    match decision.route.to_uppercase().as_str() {
        "SQL" => {
            let batches = run_sql_pipeline(state, query).await?;
            let lines = crate::engines::sql::record_batches_to_json_lines(&batches)?;
            Ok(QueryOutput::Sql(lines))
        }
        _ => {
            let res = run_semantic_pipeline(state, query).await?;
            Ok(QueryOutput::Semantic(res))
        }
    }
}

/// Executes natural language to SQL generation and AST-validated DataFusion execution directly.
#[tracing::instrument(name = "pipeline_sql_query", skip(llm, sql_engine))]
pub async fn run_sql_query(
    llm: &OllamaClient,
    sql_engine: &SessionContext,
    query: &str,
) -> Result<Vec<RecordBatch>> {
    let full_prompt = build_sql_prompt(query);

    let raw_json = match llm.prompt_model(&full_prompt).await {
        Ok(json) => json,
        Err(err) => bail!("Cannot execute AI SQL query: Ollama is offline or timed out: {}", err),
    };
    let response: SqlResponse = parse_llm_json(&raw_json)?;
    let batches = execute_sql_query(sql_engine, &response.query).await?;
    Ok(batches)
}

/// Executes natural language to SQL generation and AST-validated DataFusion execution via AppState.
pub async fn run_sql_pipeline(state: &AppState, query: &str) -> Result<Vec<RecordBatch>> {
    run_sql_query(&state.llm_client, &state.sql_engine, query).await
}

/// Unified Semantic Query Pipeline (shared between CLI and HTTP gateway).
/// Eliminates the duplicated orchestration logic.
#[tracing::instrument(name = "pipeline_semantic_query", skip(state))]
pub async fn run_semantic_pipeline(state: &AppState, query: &str) -> Result<SemanticQueryResult> {
    if let Some(cached_bytes) = state.semantic_cache.get_exact(query) {
        info!("Semantic Cache HIT (Tier 1: Exact match) for query: '{}'", query);
        if let Ok(res) = serde_json::from_slice::<SemanticQueryResult>(&cached_bytes) {
            return Ok(res);
        }
    }

    info!("Parsing raw query to separate semantic intent from exact filters...");
    let parser_prompt = build_question_parser_prompt(query);
    let parsed_query = match state.llm_client.prompt_model(&parser_prompt).await {
        Ok(raw_parser_output) => parse_llm_json::<ParsedQuestion>(&raw_parser_output).unwrap_or_else(|_| {
            ParsedQuestion {
                intent: query.to_string(),
                filters: vec![],
            }
        }),
        Err(err) => {
            warn!("LLM parser offline ({}). Using raw query as intent without filters.", err);
            ParsedQuestion {
                intent: query.to_string(),
                filters: vec![],
            }
        }
    };

    let embeddings = crate::engines::embeddings::get_embeddings(
        vec![parsed_query.intent.clone()],
        Arc::clone(&state.tokenizer),
        Arc::clone(&state.embedding_model),
    ).await?;
    let query_vector = embeddings.into_iter().next().ok_or_else(|| anyhow::anyhow!("Failed to generate embedding"))?;

    if let Some(cached_bytes) = state.semantic_cache.get_similar(&query_vector) {
        info!("Semantic Cache HIT (Tier 2: Vector similarity match) for query: '{}'", query);
        if let Ok(res) = serde_json::from_slice::<SemanticQueryResult>(&cached_bytes) {
            return Ok(res);
        }
    }

    let chunks = crate::engines::vector::execute_semantic_search_vector(
        query_vector.clone(),
        &state.vector_db_uri,
        &parsed_query.filters,
    ).await?;

    info!("Passing retrieved chunks to LLM for deterministic generation...");
    let semantic_prompt = build_semantic_prompt(query, &chunks);
    match state.llm_client.prompt_model(&semantic_prompt).await {
        Ok(raw_llm_output) => {
            let mut final_payload = verify_and_parse_llm_generation(&raw_llm_output, &chunks)?;

            if !final_payload.answer_found {
                info!("Vector search failed. Triggering deterministic fallback (Absence Proof)...");
                let fallback_chunks = execute_fallback_search(&parsed_query.intent, &state.vector_db_uri).await?;

                if !fallback_chunks.is_empty() {
                    info!("Fallback search found missing chunks. Re-querying LLM...");
                    let fallback_prompt = build_semantic_prompt(query, &fallback_chunks);
                    if let Ok(raw_fallback_output) = state.llm_client.prompt_model(&fallback_prompt).await {
                        final_payload = verify_and_parse_llm_generation(&raw_fallback_output, &fallback_chunks)?;
                    }
                } else {
                    info!("Absence mathematically proven. The data does not exist in the corpus.");
                }
            }

            let verified_res = SemanticQueryResult::Verified(final_payload);
            if let Ok(bytes) = serde_json::to_vec(&verified_res) {
                state.semantic_cache.insert(
                    query.to_string(),
                    query_vector.into_boxed_slice(),
                    bytes.into_boxed_slice(),
                );
            }
            Ok(verified_res)
        }
        Err(err) => {
            warn!("LLM generation offline ({}). Returning raw vector chunks (degraded mode).", err);
            let degraded_chunks = chunks
                .into_iter()
                .map(|(chunk_id, content)| DegradedChunk {
                    chunk_id,
                    content: content.to_string(),
                })
                .collect();
            let degraded_payload = DegradedResponse {
                degraded: true,
                message: format!("LLM is offline or timed out: {}", err),
                retrieved_chunks: degraded_chunks,
            };
            Ok(SemanticQueryResult::Degraded(degraded_payload))
        }
    }
}
