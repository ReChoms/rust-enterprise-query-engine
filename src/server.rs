use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use candle_transformers::models::bert::BertModel;
use datafusion::prelude::SessionContext;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::{info, warn};

use crate::llm::{
    build_question_parser_prompt, build_routing_prompt, build_semantic_prompt, build_sql_prompt,
    parse_llm_json, verify_and_parse_llm_generation, OllamaClient,
};
use crate::sql_engine::{execute_sql_query, record_batches_to_json_lines};
use crate::types::{
    DegradedChunk, DegradedResponse, HealthResponse, ParsedQuestion, RouterDecision,
    SqlResponse,
};
use crate::vector_db::{
    check_lancedb_health, execute_fallback_search, execute_semantic_search,
};

#[derive(Clone)]
pub struct AppState {
    pub llm_client: Arc<OllamaClient>,
    pub sql_engine: Arc<SessionContext>,
    pub embedding_model: Arc<BertModel>,
    pub tokenizer: Arc<Tokenizer>,
    pub vector_db_uri: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct QueryRequest {
    pub query: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SqlQueryRequest {
    pub query: String,
}

pub async fn health_handler(
    State(state): State<AppState>,
) -> (StatusCode, Json<HealthResponse>) {
    let llm_online = state.llm_client.is_healthy().await;
    let lancedb_stats = check_lancedb_health(&state.vector_db_uri).await;

    let (vector_db_connected, total_records) = match lancedb_stats {
        Ok(count) => (true, count),
        Err(_) => (false, 0),
    };

    let (status, code) = if llm_online && vector_db_connected {
        ("healthy".to_string(), StatusCode::OK)
    } else if vector_db_connected {
        ("degraded".to_string(), StatusCode::OK)
    } else {
        ("unhealthy".to_string(), StatusCode::SERVICE_UNAVAILABLE)
    };

    let report = HealthResponse {
        status,
        llm_connected: llm_online,
        vector_db_connected,
        total_records,
    };

    (code, Json(report))
}

pub async fn sql_query_handler(
    State(state): State<AppState>,
    Json(payload): Json<SqlQueryRequest>,
) -> Result<Json<Vec<serde_json::Value>>, (StatusCode, String)> {
    let query = payload.query.trim();
    if query.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "SQL query cannot be empty".to_string()));
    }

    let batches = execute_sql_query(&state.sql_engine, query)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("SQL Execution Error: {}", e)))?;

    let lines = record_batches_to_json_lines(&batches)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Serialization Error: {}", e)))?;

    let mut json_rows = Vec::new();
    for line in lines {
        if let Ok(val) = serde_json::from_str(&line) {
            json_rows.push(val);
        }
    }

    Ok(Json(json_rows))
}

pub async fn semantic_query_handler(
    State(state): State<AppState>,
    Json(payload): Json<QueryRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let query = payload.query.trim();
    if query.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Query cannot be empty".to_string()));
    }

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

    let chunks = execute_semantic_search(
        &parsed_query.intent,
        &state.vector_db_uri,
        &parsed_query.filters,
        Arc::clone(&state.embedding_model),
        Arc::clone(&state.tokenizer),
    )
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Vector search error: {}", e)))?;

    let semantic_prompt = build_semantic_prompt(query, &chunks);
    match state.llm_client.prompt_model(&semantic_prompt).await {
        Ok(raw_llm_output) => {
            let mut final_payload = verify_and_parse_llm_generation(&raw_llm_output, &chunks)
                .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, format!("Verification error: {}", e)))?;

            if !final_payload.answer_found {
                let fallback_chunks = execute_fallback_search(&parsed_query.intent, &state.vector_db_uri)
                    .await
                    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Fallback search error: {}", e)))?;

                if !fallback_chunks.is_empty() {
                    let fallback_prompt = build_semantic_prompt(query, &fallback_chunks);
                    if let Ok(raw_fallback_output) = state.llm_client.prompt_model(&fallback_prompt).await {
                        if let Ok(payload) = verify_and_parse_llm_generation(&raw_fallback_output, &fallback_chunks) {
                            final_payload = payload;
                        }
                    }
                }
            }

            let val = serde_json::to_value(&final_payload)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("JSON serialization error: {}", e)))?;
            Ok(Json(val))
        }
        Err(err) => {
            warn!("LLM generation offline ({}). Returning degraded vector chunks.", err);
            let degraded_chunks: Vec<DegradedChunk> = chunks
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
            let val = serde_json::to_value(&degraded_payload)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("JSON serialization error: {}", e)))?;
            Ok(Json(val))
        }
    }
}

pub async fn router_query_handler(
    State(state): State<AppState>,
    Json(payload): Json<QueryRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let query = payload.query.trim();
    if query.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Query cannot be empty".to_string()));
    }

    let full_prompt = build_routing_prompt(query);
    match state.llm_client.prompt_model(&full_prompt).await {
        Ok(raw_json) => {
            let decision: RouterDecision = parse_llm_json(&raw_json).unwrap_or_else(|_| {
                RouterDecision {
                    route: "SEMANTIC".to_string(),
                }
            });

            if decision.route == "SQL" {
                let sql_prompt = build_sql_prompt(query);
                let raw_sql_json = state
                    .llm_client
                    .prompt_model(&sql_prompt)
                    .await
                    .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Ollama SQL generator offline: {}", e)))?;
                let sql_resp: SqlResponse = parse_llm_json(&raw_sql_json)
                    .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, format!("Invalid SQL schema: {}", e)))?;
                let batches = execute_sql_query(&state.sql_engine, &sql_resp.query)
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, format!("SQL Execution Error: {}", e)))?;
                let lines = record_batches_to_json_lines(&batches)
                    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Serialization Error: {}", e)))?;
                let mut json_rows: Vec<serde_json::Value> = Vec::new();
                for line in lines {
                    if let Ok(val) = serde_json::from_str(&line) {
                        json_rows.push(val);
                    }
                }
                Ok(Json(serde_json::json!({
                    "route": "SQL",
                    "generated_sql": sql_resp.query,
                    "results": json_rows
                })))
            } else {
                let sem_resp = semantic_query_handler(State(state), Json(payload)).await?;
                Ok(Json(serde_json::json!({
                    "route": "SEMANTIC",
                    "response": sem_resp.0
                })))
            }
        }
        Err(_) => {
            let sem_resp = semantic_query_handler(State(state), Json(payload)).await?;
            Ok(Json(serde_json::json!({
                "route": "SEMANTIC_FALLBACK",
                "response": sem_resp.0
            })))
        }
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C signal. Initiating graceful drain & shutdown...");
        },
        _ = terminate => {
            info!("Received SIGTERM signal. Initiating graceful drain & shutdown...");
        },
    }
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/query/sql", post(sql_query_handler))
        .route("/query/semantic", post(semantic_query_handler))
        .route("/query", post(router_query_handler))
        .with_state(state)
}

pub async fn start_server(state: AppState, addr: SocketAddr) -> anyhow::Result<()> {
    let app = create_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("Axum REST API listening on http://{}", addr);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    info!("Axum REST API server shutdown completed gracefully.");
    Ok(())
}
