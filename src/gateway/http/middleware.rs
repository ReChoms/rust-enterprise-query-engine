use axum::http::StatusCode;
use axum::response::Json;
use axum::BoxError;

/// Converts middleware errors into structured HTTP JSON responses.
/// Overloaded requests (exceeding 128 concurrent queries) fail fast in ~5µs with HTTP 503.
pub async fn handle_load_shed_error(err: BoxError) -> (StatusCode, Json<serde_json::Value>) {
    if err.is::<tower::load_shed::error::Overloaded>() {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "Server overloaded, request shed to protect memory stability",
                "status_code": 503
            })),
        )
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Unhandled middleware error: {}", err),
                "status_code": 500
            })),
        )
    }
}
