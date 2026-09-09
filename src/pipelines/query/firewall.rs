use anyhow::{anyhow, bail, Result};
use serde::de::DeserializeOwned;
use std::collections::HashMap;
use std::sync::Arc;

use crate::common::types::SemanticResponse;

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

/// Anti-hallucination verification firewall: enforces that any positive claim
/// made by the LLM contains a verifiable, verbatim substring quote from retrieved chunks.
pub fn verify_and_parse_llm_generation(
    raw_output: &str,
    retrieved_chunks: &HashMap<String, Arc<str>>,
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
            Arc::from("The customer is Acme Corp in Berlin."),
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
            Arc::from("The customer is Acme Corp in Berlin."),
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
            Arc::from("The customer is Acme Corp in Berlin."),
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
