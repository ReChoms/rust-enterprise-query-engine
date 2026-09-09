use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Json;

use crate::common::types::{HealthResponse, QueryRequest, SqlQueryRequest, SqlResponse, RouterDecision};
use crate::engines::sql::{execute_sql_query, record_batches_to_json_lines};
use crate::engines::vector::check_lancedb_health;
use crate::pipelines::query::{
    build_routing_prompt, build_sql_prompt, parse_llm_json, run_semantic_pipeline, AppState,
};

pub async fn health_handler(State(state): State<AppState>) -> (StatusCode, Json<HealthResponse>) {
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
        return Err((
            StatusCode::BAD_REQUEST,
            "SQL query cannot be empty".to_string(),
        ));
    }

    let batches = execute_sql_query(&state.sql_engine, query)
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                format!("SQL Execution Error: {}", e),
            )
        })?;

    let lines = record_batches_to_json_lines(&batches).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Serialization Error: {}", e),
        )
    })?;

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

    let result = run_semantic_pipeline(&state, query).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Semantic pipeline error: {}", e),
        )
    })?;

    let val = serde_json::to_value(&result).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("JSON serialization error: {}", e),
        )
    })?;

    Ok(Json(val))
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
            let decision: RouterDecision =
                parse_llm_json(&raw_json).unwrap_or_else(|_| RouterDecision {
                    route: "SEMANTIC".to_string(),
                });

            if decision.route == "SQL" {
                let sql_prompt = build_sql_prompt(query);
                let raw_sql_json =
                    state
                        .llm_client
                        .prompt_model(&sql_prompt)
                        .await
                        .map_err(|e| {
                            (
                                StatusCode::BAD_GATEWAY,
                                format!("Ollama SQL generator offline: {}", e),
                            )
                        })?;
                let sql_resp: SqlResponse = parse_llm_json(&raw_sql_json).map_err(|e| {
                    (
                        StatusCode::UNPROCESSABLE_ENTITY,
                        format!("Invalid SQL schema: {}", e),
                    )
                })?;
                let batches = execute_sql_query(&state.sql_engine, &sql_resp.query)
                    .await
                    .map_err(|e| {
                        (
                            StatusCode::BAD_REQUEST,
                            format!("SQL Execution Error: {}", e),
                        )
                    })?;
                let lines = record_batches_to_json_lines(&batches).map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Serialization Error: {}", e),
                    )
                })?;
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
