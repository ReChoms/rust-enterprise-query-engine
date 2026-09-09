use anyhow::Result;
use tracing::info;

use crate::common::types::ParsedQuestion;
use super::firewall::parse_llm_json;
use super::ollama::OllamaClient;
use super::prompts::build_question_parser_prompt;

/// Parses a user's natural language question into semantic intent and alphanumeric filter constraints.
pub async fn parse_question_intent(llm: &OllamaClient, user_question: &str) -> Result<ParsedQuestion> {
    info!("Parsing natural language intent and alphanumeric filter constraints...");
    let prompt = build_question_parser_prompt(user_question);
    let raw_response = llm.prompt_model(&prompt).await?;
    let parsed: ParsedQuestion = parse_llm_json(&raw_response)?;
    Ok(parsed)
}
