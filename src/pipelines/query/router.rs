use anyhow::Result;
use tracing::info;

use crate::common::types::RouterDecision;
use super::firewall::parse_llm_json;
use super::ollama::OllamaClient;
use super::prompts::build_routing_prompt;

/// Classifies a natural language query into either "SQL" or "SEMANTIC" using local LLM.
pub async fn classify_intent(llm: &OllamaClient, user_question: &str) -> Result<RouterDecision> {
    info!("Determining execution strategy (SQL vs Semantic)...");
    let prompt = build_routing_prompt(user_question);
    let raw_response = llm.prompt_model(&prompt).await?;
    let decision: RouterDecision = parse_llm_json(&raw_response)?;
    Ok(decision)
}
