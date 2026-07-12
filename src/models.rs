use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct Kna1Row {
    pub kunnr: Option<String>,
    pub name1: Option<String>,
    pub ort01: Option<String>,
    pub land1: Option<String>,
}

#[derive(Serialize)]
pub struct OllamaRequest {
    pub model: String,
    pub prompt: String,
    pub stream: bool,
}

#[derive(Deserialize)]
pub struct OllamaResponse {
    pub response: String,
}

#[derive(Deserialize, Debug)]
pub struct RouterDecision {
    pub route: String,
    pub query: String,
}

#[derive(Deserialize, Debug)]
pub struct SqlResponse {
    pub query: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct SemanticResponse {
    pub answer_found: bool,
    pub answer: String,
    pub exact_quote: String,
    pub source_chunk_id: String,
}

/// Data schema to separate natural language intent from exact alphanumeric constraints.
/// 
/// Why: Fixes Pitfall 4. We cannot feed raw SAP IDs (like "1000") or logical 
/// negations (like "NOT") into dense vector embeddings, because the mathematics 
/// will incorrectly index on the numbers rather than the semantic meaning.
/// By forcing the LLM to map its response into this strict JSON schema, we isolate 
/// the pure `intent` for vector search, and extract the `filters` for deterministic Hybrid Search.
#[derive(Deserialize, Debug)]
pub struct ParsedQuestion {
    pub intent: String,
    pub filters: Vec<String>,
}
