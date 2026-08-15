use crate::types::{OllamaRequest, OllamaResponse, SemanticResponse};
use anyhow::{anyhow, bail, Result};
use serde::de::DeserializeOwned;
use std::collections::HashMap;

const SAP_KNA1_SCHEMA: &str = "\
Database Schema for table `kna1`:\n\
- kunnr (String): Customer Number / ID\n\
- name1 (String): Customer Name\n\
- ort01 (String): City\n\
- land1 (String): Country Code (e.g., 'US', 'DE')\n";

use std::time::Duration;

#[derive(Clone, Debug)]
pub struct OllamaClient {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl OllamaClient {
    /// Initializes an HTTP connection-pooled client from environment variables with fallback defaults
    pub fn init_from_env_or_default() -> Result<Self> {
        let base_url = std::env::var("OLLAMA_HOST")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        let timeout_secs: u64 = std::env::var("OLLAMA_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        let model = std::env::var("OLLAMA_MODEL")
            .unwrap_or_else(|_| "llama3.2:latest".to_string());

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| anyhow!("Failed to build HTTP client: {}", e))?;

        Ok(Self {
            client,
            base_url,
            model,
        })
    }

    /// Fast health probe returning true if the Ollama daemon is reachable
    pub async fn is_healthy(&self) -> bool {
        let url = format!("{}/api/tags", self.base_url);
        match self.client.get(&url).timeout(Duration::from_secs(2)).send().await {
            Ok(res) if res.status().is_success() => true,
            _ => false,
        }
    }

    /// Sends a prompt to the Ollama model with exponential backoff retry
    pub async fn prompt_model(&self, prompt: &str) -> Result<String> {
        let req_body = OllamaRequest {
            model: self.model.clone(),
            prompt: prompt.to_string(),
            stream: false,
        };

        let url = format!("{}/api/generate", self.base_url);
        let mut last_err = None;

        for attempt in 0..2 {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_millis(500)).await;
            }

            match self.client.post(&url).json(&req_body).send().await {
                Ok(res) => {
                    let parsed = res.json::<OllamaResponse>().await?;
                    return Ok(parsed.response);
                }
                Err(e) => {
                    last_err = Some(e);
                }
            }
        }

        Err(anyhow!(
            "Failed to reach LLM at {} after retry: {}",
            url,
            last_err.map(|e| e.to_string()).unwrap_or_default()
        ))
    }
}

pub fn build_routing_prompt(user_question: &str) -> String {
    format!(
        "You are an expert SAP data engineer. Read the user's question and decide if it requires exact SQL or SEMANTIC search.\n\n{}\nRULES:\n1. You must ONLY output raw JSON. Do not wrap it in markdown. Do not add conversational text.\n2. The JSON must have one key: \"route\" (either \"SQL\" or \"SEMANTIC\").\n\nExamples:\nQ: \"How many customers are in Berlin?\"\nA: {{\"route\": \"SQL\"}}\n\nQ: \"Show me the names of 5 customers in the US.\"\nA: {{\"route\": \"SQL\"}}\n\nQ: \"Find customers who are large tech manufacturers.\"\nA: {{\"route\": \"SEMANTIC\"}}\n\nUser Question: \"{}\"\nA: ",
        SAP_KNA1_SCHEMA, user_question
    )
}

pub fn build_sql_prompt(user_question: &str) -> String {
    format!(
        "You are an expert SAP data engineer. Read the user's question and write the exact SQL query required.\n\n{}\nRULES:\n1. You must ONLY output raw JSON. Do not wrap it in markdown. Do not add conversational text.\n2. The JSON must have one key: \"query\" containing the generated SQL string.\n\nUser Question: \"{}\"\nA: ",
        SAP_KNA1_SCHEMA, user_question
    )
}

/// Strips hallucinated markdown wrappers from LLM outputs and safely parses
/// the remaining JSON into any strongly typed Rust struct.
pub fn parse_llm_json<T: DeserializeOwned>(raw_output: &str) -> Result<T> {
    let mut clean_json = raw_output.trim();

    if let Some(stripped) = clean_json.strip_prefix("```json\n") {
        clean_json = stripped;
    } else if let Some(stripped) = clean_json.strip_prefix("```json") {
        clean_json = stripped;
    } else if let Some(stripped) = clean_json.strip_prefix("```\n") {
        clean_json = stripped;
    } else if let Some(stripped) = clean_json.strip_prefix("```") {
        clean_json = stripped;
    }

    clean_json = clean_json.trim();

    if let Some(stripped) = clean_json.strip_suffix("```") {
        clean_json = stripped;
    }

    clean_json = clean_json.trim();

    serde_json::from_str(clean_json)
        .map_err(|e| anyhow!("Failed to parse JSON: {}. Raw text: {}", e, clean_json))
}

pub fn verify_and_parse_llm_generation(
    raw_output: &str,
    retrieved_chunks: &HashMap<String, std::sync::Arc<str>>,
) -> Result<SemanticResponse> {
    let response: SemanticResponse = parse_llm_json(raw_output)?;

    if response.answer_found {
        let source_text = retrieved_chunks
            .get(&response.source_chunk_id)
            .ok_or_else(|| {
                anyhow!(
                    "SECURITY VIOLATION: LLM cited a non-existent chunk ID: {}",
                    response.source_chunk_id
                )
            })?;

        if !source_text.contains(&response.exact_quote) {
            bail!(
                "SECURITY VIOLATION: Hallucination detected. The exact quote does not exist in the source document. Quote: '{}'",
                response.exact_quote
            );
        }
    }

    Ok(response)
}

pub fn build_semantic_prompt(
    user_question: &str,
    chunks: &HashMap<String, std::sync::Arc<str>>,
) -> String {
    let mut context = String::new();
    for (id, text) in chunks {
        context.push_str(&format!("CHUNK ID: {}\nTEXT: {}\n\n", id, text));
    }
    //this prompt is  there to ensure the ai doenst jsut answer but is forced to answer the other
    //context which increases then accuracy
    format!(
        "You are an expert SAP data engineer. Answer the user's question using ONLY the provided chunks.\n\n\
        CONTEXT:\n{}\n\n\
        RULES:\n\
        1. You must ONLY output raw JSON. Do not wrap it in markdown. Do not add conversational text.\n\
        2. The JSON must exactly match this schema:\n\
           {{\n\
             \"answer_found\": boolean,\n\
             \"answer\": \"your detailed answer based ONLY on the text\",\n\
             \"exact_quote\": \"a verbatim, exact substring from the chunk that proves your answer\",\n\
             \"source_chunk_id\": \"the specific CHUNK ID you extracted the quote from\"\n\
           }}\n\
        3. If the answer is not found in the chunks, set `answer_found` to false, and leave the other fields blank.\n\n\
        User Question: \"{}\"\nA: ",
        context, user_question
    )
}

/// Resolves Research Area 4 (Raw Query Embedding) by intercepting the user's raw question
/// before it reaches the tensor embedding pipeline.
///
/// Why: Dense vector math is excellent at semantic similarity but terrible at
/// boolean logic (like "NOT") and heavily over-indexes exact alphanumeric strings.
/// If a user asks "NOT KUNNR 1000", passing this directly to LanceDB will likely
/// retrieve customer 1000 because the vector mathematically aligns with the specific number.
///
/// This prompt forces a local LLM to parse the query into:
/// 1. `intent`: The pure semantic meaning (safe for vector math).
/// 2. `filters`: Explicit IDs and logical exclusions (used later for deterministic Hybrid Search).
pub fn build_question_parser_prompt(user_question: &str) -> String {
    format!(
        "You are an expert SAP data engineer. Your task is to parse the user's question into pure semantic intent and explicit filters.\n\n\
        RULES:\n\
        1. You must ONLY output raw JSON. Do not wrap it in markdown. Do not add conversational text.\n\
        2. The JSON must exactly match this schema:\n\
           {{\n\
             \"intent\": \"The pure semantic meaning, without hard IDs or strict exclusions\",\n\
             \"filters\": [\"EXACT_ID_1\", \"NOT EXACT_ID_2\"]\n\
           }}\n\
        3. If there are no explicit IDs or exclusions, leave the `filters` array empty.\n\n\
        Examples:\n\
        Q: \"Which customer in Berlin is NOT KUNNR 1000?\"\n\
        A: {{\"intent\": \"customer in Berlin\", \"filters\": [\"NOT KUNNR 1000\"]}}\n\n\
        Q: \"Find SAP customer 00001042.\"\n\
        A: {{\"intent\": \"Find SAP customer\", \"filters\": [\"00001042\"]}}\n\n\
        User Question: \"{}\"\nA: ",
        user_question
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn test_parse_clean_json() {
        let raw = r#"{"route": "SQL", "query": "SELECT * FROM kna1"}"#;
        let parsed: Value = parse_llm_json(raw).unwrap();
        assert_eq!(parsed["route"], "SQL");
    }

    #[test]
    fn test_parse_markdown_wrapped_json() {
        let raw = "```json\n{\"route\": \"SQL\", \"query\": \"\"}\n```";
        let parsed: Value = parse_llm_json(raw).unwrap();
        assert_eq!(parsed["route"], "SQL");
    }

    #[test]
    fn test_parse_triple_backtick_json() {
        let raw = "```\n{\"route\": \"SEMANTIC\", \"query\": \"\"}\n```";
        let parsed: Value = parse_llm_json(raw).unwrap();
        assert_eq!(parsed["route"], "SEMANTIC");
    }

    #[test]
    fn test_parse_garbage_fails() {
        let raw = "This is just conversational text.";
        let parsed: Result<Value> = parse_llm_json(raw);
        assert!(parsed.is_err());
    }

    #[test]
    fn test_verify_valid_generation() {
        let mut chunks = HashMap::new();
        chunks.insert(
            "chunk_1".to_string(),
            std::sync::Arc::from("The customer is Acme Corp in Berlin."),
        );

        let raw = r#"{
            "answer_found": true,
            "answer": "The customer is Acme Corp.",
            "exact_quote": "Acme Corp in Berlin",
            "source_chunk_id": "chunk_1"
        }"#;

        let result = verify_and_parse_llm_generation(raw, &chunks);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_hallucinated_quote_blocked() {
        let mut chunks = HashMap::new();
        chunks.insert(
            "chunk_1".to_string(),
            std::sync::Arc::from("The customer is Acme Corp in Berlin."),
        );

        let raw = r#"{
            "answer_found": true,
            "answer": "The customer is in Munich.",
            "exact_quote": "Acme Corp in Munich",
            "source_chunk_id": "chunk_1"
        }"#;

        let result = verify_and_parse_llm_generation(raw, &chunks);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Hallucination detected"));
    }

    #[test]
    fn test_verify_fake_chunk_id_blocked() {
        let mut chunks = HashMap::new();
        chunks.insert(
            "chunk_1".to_string(),
            std::sync::Arc::from("The customer is Acme Corp in Berlin."),
        );

        let raw = r#"{
            "answer_found": true,
            "answer": "The customer is Acme Corp.",
            "exact_quote": "Acme Corp",
            "source_chunk_id": "chunk_999"
        }"#;

        let result = verify_and_parse_llm_generation(raw, &chunks);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("non-existent chunk ID"));
    }

    #[test]
    fn test_verify_answer_not_found_passes() {
        let chunks = HashMap::new();

        let raw = r#"{
            "answer_found": false,
            "answer": "",
            "exact_quote": "",
            "source_chunk_id": ""
        }"#;

        let result = verify_and_parse_llm_generation(raw, &chunks);
        assert!(result.is_ok());
    }
}
