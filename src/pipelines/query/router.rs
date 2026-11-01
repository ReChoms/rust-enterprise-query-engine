use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::common::types::RouterDecision;
use super::firewall::parse_llm_json;
use super::ollama::OllamaClient;
use super::prompts::build_routing_prompt;
use super::{AppState, SemanticQueryResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "route")]
pub enum RoutedQueryResult {
    #[serde(rename = "SQL")]
    Sql {
        generated_sql: String,
        results: Vec<serde_json::Value>,
    },
    #[serde(rename = "SEMANTIC")]
    Semantic { response: SemanticQueryResult },
    #[serde(rename = "SEMANTIC_FALLBACK")]
    SemanticFallback { response: SemanticQueryResult },
}

/// Executes a routed natural language query workflow.
pub async fn run_routed_pipeline(state: &AppState, query: &str) -> Result<RoutedQueryResult> {
    let prompt = build_routing_prompt(query);
    let Ok(raw_json) = state.llm_client.prompt_model(&prompt).await else {
        let response = super::run_semantic_pipeline(state, query).await?;
        return Ok(RoutedQueryResult::SemanticFallback { response });
    };

    let decision: RouterDecision = parse_llm_json(&raw_json)
        .unwrap_or_else(|_| RouterDecision { route: "SEMANTIC".to_string() });

    if decision.route == "SQL" {
        return execute_routed_sql(state, query).await;
    }
    let response = super::run_semantic_pipeline(state, query).await?;
    Ok(RoutedQueryResult::Semantic { response })
}

async fn execute_routed_sql(state: &AppState, query: &str) -> Result<RoutedQueryResult> {
    let raw_sql = state.llm_client.prompt_model(&super::prompts::build_sql_prompt(query)).await?;
    let sql_resp: crate::common::types::SqlResponse = parse_llm_json(&raw_sql)?;
    let batches = crate::engines::sql::execute_sql_query(&state.sql_engine, &sql_resp.query).await?;
    let lines = crate::engines::sql::record_batches_to_json_lines(&batches)?;
    let results = lines.iter().map(|l| serde_json::from_str(l)).collect::<Result<Vec<_>, _>>()?;
    Ok(RoutedQueryResult::Sql { generated_sql: sql_resp.query, results })
}

/// Classifies a natural language query into either "SQL" or "SEMANTIC" using local LLM.
pub async fn classify_intent(llm: &OllamaClient, user_question: &str) -> Result<RouterDecision> {
    info!("Determining execution strategy (SQL vs Semantic)...");
    let prompt = build_routing_prompt(user_question);
    let raw_response = llm.prompt_model(&prompt).await?;
    let decision: RouterDecision = parse_llm_json(&raw_response)?;
    Ok(decision)
}
