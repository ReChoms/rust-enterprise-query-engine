use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Kna1Row {
    pub kunnr: Option<String>,
    pub name1: Option<String>,
    pub ort01: Option<String>,
    pub land1: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct QueryRequest {
    pub query: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SqlQueryRequest {
    pub query: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct RouterDecision {
    pub route: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct SqlResponse {
    pub query: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct SemanticResponse {
    pub answer_found: bool,
    pub answer: String,
    pub exact_quote: String,
    pub source_chunk_id: String,
}

/// Data schema to separate natural language intent from exact alphanumeric constraints.
///
/// Prevents dense vector embedding degradation from raw IDs (like "1000") or logical negations.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ParsedQuestion {
    pub intent: String,
    pub filters: Vec<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct DegradedChunk {
    pub chunk_id: String,
    pub content: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct DegradedResponse {
    pub degraded: bool,
    pub message: String,
    pub retrieved_chunks: Vec<DegradedChunk>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct HealthResponse {
    pub status: String,
    pub llm_connected: bool,
    pub vector_db_connected: bool,
    pub total_records: usize,
}
