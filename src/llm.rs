use crate::models::{OllamaRequest, OllamaResponse, SemanticResponse};
use std::collections::HashMap;
use anyhow::{anyhow, bail, Result};
use serde::de::DeserializeOwned;

const SAP_KNA1_SCHEMA: &str = "\
Database Schema for table `kna1`:\n\
- kunnr (String): Customer Number / ID\n\
- name1 (String): Customer Name\n\
- ort01 (String): City\n\
- land1 (String): Country Code (e.g., 'US', 'DE')\n";

/// Sends a prompt to the local Ollama server running Llama 3.2
pub async fn ask_llm(prompt: &str) -> Result<String> {
    let req_body = OllamaRequest {
        model: "llama3.2:latest".to_string(),
        prompt: prompt.to_string(),
        stream: false,
    };

    let host = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
    let url = format!("{}/api/generate", host);

    let client = reqwest::Client::new();
    let res = client
        .post(&url)
        .json(&req_body)
        .send()
        .await?
        .json::<OllamaResponse>()
        .await?;
    Ok(res.response)
}

pub fn build_routing_prompt(user_question: &str) -> String {
    format!(
        "You are an expert SAP data engineer. Read the user's question and decide if it requires exact SQL or SEMANTIC search.\n\n{}\nRULES:\n1. You must ONLY output raw JSON. Do not wrap it in markdown. Do not add conversational text.\n2. The JSON must have two keys: \"route\" (either \"SQL\" or \"SEMANTIC\") and \"query\" (the generated SQL string, or blank).\n\nExamples:\nQ: \"How many customers are in Berlin?\"\nA: {{\"route\": \"SQL\", \"query\": \"SELECT count(*) FROM kna1 WHERE ort01 = 'Berlin'\"}}\n\nQ: \"Show me the names of 5 customers in the US.\"\nA: {{\"route\": \"SQL\", \"query\": \"SELECT name1 FROM kna1 WHERE land1 = 'US' LIMIT 5\"}}\n\nQ: \"Find customers who are large tech manufacturers.\"\nA: {{\"route\": \"SEMANTIC\", \"query\": \"\"}}\n\nUser Question: \"{}\"\nA: ",
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

    if clean_json.starts_with("```json") {
        clean_json = clean_json.strip_prefix("```json").unwrap();
    } else if clean_json.starts_with("```") {
        clean_json = clean_json.strip_prefix("```").unwrap();
    }

    clean_json = clean_json.trim();

    if clean_json.ends_with("```") {
        clean_json = clean_json.strip_suffix("```").unwrap();
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
        let source_text = retrieved_chunks.get(&response.source_chunk_id).ok_or_else(|| {
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

pub fn build_semantic_prompt(user_question: &str, chunks: &HashMap<String, std::sync::Arc<str>>) -> String {
    let mut context = String::new();
    for (id, text) in chunks {
        context.push_str(&format!("CHUNK ID: {}\nTEXT: {}\n\n", id, text));
    }

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
        chunks.insert("chunk_1".to_string(), std::sync::Arc::from("The customer is Acme Corp in Berlin."));
        
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
        chunks.insert("chunk_1".to_string(), std::sync::Arc::from("The customer is Acme Corp in Berlin."));
        
        let raw = r#"{
            "answer_found": true,
            "answer": "The customer is in Munich.",
            "exact_quote": "Acme Corp in Munich",
            "source_chunk_id": "chunk_1"
        }"#;

        let result = verify_and_parse_llm_generation(raw, &chunks);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Hallucination detected"));
    }

    #[test]
    fn test_verify_fake_chunk_id_blocked() {
        let mut chunks = HashMap::new();
        chunks.insert("chunk_1".to_string(), std::sync::Arc::from("The customer is Acme Corp in Berlin."));
        
        let raw = r#"{
            "answer_found": true,
            "answer": "The customer is Acme Corp.",
            "exact_quote": "Acme Corp",
            "source_chunk_id": "chunk_999"
        }"#;

        let result = verify_and_parse_llm_generation(raw, &chunks);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("non-existent chunk ID"));
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
