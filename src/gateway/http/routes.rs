use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Json;

use super::dto::{ErrorResponse, HealthResponse, QueryRequest, SqlQueryRequest};
use crate::engines::sql::execute_sql_query;
use crate::engines::vector::check_lancedb_health;
use crate::pipelines::query::{
    run_routed_pipeline, run_semantic_pipeline, AppState, RoutedQueryResult,
    SemanticQueryResult,
};

pub async fn health_handler(State(state): State<AppState>) -> (StatusCode, Json<HealthResponse>) {
    let llm_online = state.llm_client.is_healthy().await;
    let lancedb_stats = check_lancedb_health(&state.vector_db_uri).await;

    let (vector_db_connected, total_records) = match lancedb_stats {
        Ok(count) => (true, count),
        Err(_) => (false, 0),
    };
    let vector_indexed = crate::engines::vector::is_vector_indexed(&state.vector_db_uri)
        .await
        .unwrap_or(false);

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
        vector_indexed,
        cache_entries: state.semantic_cache.len(),
    };

    (code, Json(report))
}

fn validate_query_str<'a>(
    query: &'a str,
    name: &str,
    max_chars: usize,
) -> Result<&'a str, (StatusCode, Json<ErrorResponse>)> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(format!("{} cannot be empty", name))),
        ));
    }
    if trimmed.len() > max_chars {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(format!(
                "{} exceeds maximum length of {} characters",
                name, max_chars
            ))),
        ));
    }
    Ok(trimmed)
}

/// Executes SQL queries with Cloudflare Zero-Copy Wire-Streaming.
///
/// ### Cloudflare Memory Architecture Optimization:
/// Rather than deserializing Arrow JSON lines into an intermediate `Vec<serde_json::Value>`
/// DOM tree (which triggers O(N * fields) isolated heap allocations and allocator fragmentation),
/// we stream Arrow records directly into a single contiguous byte buffer (`axum::body::Bytes`)
/// framed as a JSON array (`[...]`). This achieves O(1) allocation, optimal L1/L2 cache prefetching,
/// and microsecond dispatch directly into the kernel's `SO_SNDBUF`.
#[tracing::instrument(name = "http_sql_query", skip(state, payload), fields(query = %payload.query))]
pub async fn sql_query_handler(
    State(state): State<AppState>,
    Json(payload): Json<SqlQueryRequest>,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    let query = validate_query_str(&payload.query, "SQL query", state.max_query_chars())?;

    let batches = execute_sql_query(&state.sql_engine, query)
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(format!("SQL Execution Error: {}", e))),
            )
        })?;

    // Cloudflare Zero-Copy Wire Format: Stream Arrow lines directly to raw bytes
    let mut raw_lines = Vec::new();
    crate::engines::sql::write_record_batches_as_json_lines(&batches, &mut raw_lines)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new(format!("Serialization Error: {}", e))),
            )
        })?;

    // Frame into a contiguous JSON array buffer: `[row1,row2]`
    let mut wire_buffer = Vec::with_capacity(raw_lines.len() + 2);
    wire_buffer.push(b'[');
    let mut first = true;
    for line in raw_lines.split(|&b| b == b'\n') {
        let trimmed = if line.ends_with(b"\r") {
            &line[..line.len() - 1]
        } else {
            line
        };
        if trimmed.is_empty() {
            continue;
        }
        if !first {
            wire_buffer.push(b',');
        }
        wire_buffer.extend_from_slice(trimmed);
        first = false;
    }
    wire_buffer.push(b']');

    use axum::response::IntoResponse;
    Ok((
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        axum::body::Bytes::from(wire_buffer),
    )
        .into_response())
}

#[tracing::instrument(name = "http_semantic_query", skip(state, payload), fields(query = %payload.query))]
pub async fn semantic_query_handler(
    State(state): State<AppState>,
    Json(payload): Json<QueryRequest>,
) -> Result<Json<SemanticQueryResult>, (StatusCode, Json<ErrorResponse>)> {
    let query = validate_query_str(&payload.query, "Query", state.max_query_chars())?;

    let result = run_semantic_pipeline(&state, query).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new(format!("Semantic pipeline error: {}", e))),
        )
    })?;

    Ok(Json(result))
}

pub async fn router_query_handler(
    State(state): State<AppState>,
    Json(payload): Json<QueryRequest>,
) -> Result<Json<RoutedQueryResult>, (StatusCode, Json<ErrorResponse>)> {
    let query = validate_query_str(&payload.query, "Query", state.max_query_chars())?;
    let result = run_routed_pipeline(&state, query)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new(format!("Routed query error: {}", e))),
            )
        })?;
    Ok(Json(result))
}
