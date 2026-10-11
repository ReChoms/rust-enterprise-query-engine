//! Altitude 3: Ingress Data Transfer Objects (DTOs)
//!
//! Wire-format JSON request and response payloads for HTTP routes.
//! Scoped strictly to Altitude 3 so lower layers (Altitude 2, 1, 0)
//! remain completely oblivious to HTTP transport schemas.

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct QueryRequest {
    pub query: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SqlQueryRequest {
    pub query: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct HealthResponse {
    pub status: String,
    pub llm_connected: bool,
    pub vector_db_connected: bool,
    pub total_records: usize,
    pub vector_indexed: bool,
    pub cache_entries: usize,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct ErrorResponse {
    pub error: String,
}

impl ErrorResponse {
    pub fn new(error: impl Into<String>) -> Self {
        Self {
            error: error.into(),
        }
    }
}
