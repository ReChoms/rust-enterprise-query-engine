mod cli;
mod db;
mod debug;
mod embeddings;
mod ingest;
mod llm;
mod types;

#[cfg(test)]
mod tests;

use clap::Parser;
use anyhow::{bail, Result};
use std::sync::Arc;
use tracing::{info, warn};

use crate::cli::{Cli, Commands};
use crate::db::{check_lancedb_health, execute_fallback_search, execute_semantic_search, execute_sql_query, init_datafusion};
use crate::embeddings::load_model;
use crate::ingest::execute_ingestion;
use crate::llm::{build_question_parser_prompt, build_routing_prompt, build_semantic_prompt, build_sql_prompt, parse_llm_json, verify_and_parse_llm_generation, OllamaClient};
use crate::types::{DegradedChunk, DegradedResponse, HealthResponse, ParsedQuestion, RouterDecision, SqlResponse};

#[tokio::main]
async fn main() -> Result<()> {
    debug::init_logger();

    let cli = Cli::parse();
    let llm_client = OllamaClient::init_from_env_or_default()?;

    match &cli.command {
        Commands::Ingest { file, overwrite, batch_size } => {
            info!("Loading embedding model (BAAI/bge-base-en-v1.5)...");
            let (model, tokenizer) = load_model().await?;
            execute_ingestion(file, "data/sap_vectors", *overwrite, *batch_size, Arc::clone(&model), Arc::clone(&tokenizer)).await?;
        }
        Commands::AskSemantic { query } => {
            info!(">>> Executing ASK-SEMANTIC command");
            run_semantic_pipeline(&llm_client, query).await?;
        }
        Commands::Ask { query } => {
            info!(">>> Executing ASK (ROUTER) command");
            let full_prompt = build_routing_prompt(query);
            match llm_client.prompt_model(&full_prompt).await {
                Ok(raw_json) => {
                    let decision: RouterDecision = parse_llm_json(&raw_json).unwrap_or_else(|err| {
                        warn!("Failed to parse router output '{}': {}. Defaulting to SEMANTIC route.", raw_json, err);
                        RouterDecision {
                            route: "SEMANTIC".to_string(),
                        }
                    });

                    if decision.route == "SQL" {
                        run_sql_pipeline(&llm_client, query).await?;
                    } else {
                        run_semantic_pipeline(&llm_client, query).await?;
                    }
                }
                Err(err) => {
                    warn!("Ollama router unreachable ({}). Falling back to degraded semantic search.", err);
                    run_semantic_pipeline(&llm_client, query).await?;
                }
            }
        }
        Commands::ExecuteSql { query } => {
            info!(">>> Executing EXECUTE-SQL command");
            let sql_engine = init_datafusion().await?;
            execute_sql_query(&sql_engine, query).await?;
        }
        Commands::AskAiSql { query } => {
            info!(">>> Executing ASK-AISQL command");
            run_sql_pipeline(&llm_client, query).await?;
        }
        Commands::Health => {
            info!(">>> Executing HEALTH command");
            let llm_online = llm_client.is_healthy().await;
            let lancedb_stats = check_lancedb_health("data/sap_vectors").await;

            let (vector_db_connected, total_records) = match lancedb_stats {
                Ok(count) => (true, count),
                Err(_) => (false, 0),
            };

            let status = if llm_online && vector_db_connected {
                "healthy".to_string()
            } else if vector_db_connected {
                "degraded".to_string()
            } else {
                "unhealthy".to_string()
            };

            let report = HealthResponse {
                status,
                llm_connected: llm_online,
                vector_db_connected,
                total_records,
            };

            println!("{}", serde_json::to_string(&report)?);
        }
    }

    Ok(())
}

async fn run_sql_pipeline(llm_client: &OllamaClient, query: &str) -> Result<()> {
    let full_prompt = build_sql_prompt(query);

    let raw_json = match llm_client.prompt_model(&full_prompt).await {
        Ok(json) => json,
        Err(err) => bail!("Cannot execute AI SQL query: Ollama is offline or timed out: {}", err),
    };
    let response: SqlResponse = parse_llm_json(&raw_json)?;
    let sql_engine = init_datafusion().await?;
    execute_sql_query(&sql_engine, &response.query).await?;
    Ok(())
}

async fn run_semantic_pipeline(llm_client: &OllamaClient, query: &str) -> Result<()> {
    info!("Parsing raw query to separate semantic intent from exact filters...");
    let parser_prompt = build_question_parser_prompt(query);
    let parsed_query = match llm_client.prompt_model(&parser_prompt).await {
        Ok(raw_parser_output) => parse_llm_json::<ParsedQuestion>(&raw_parser_output).unwrap_or_else(|_| {
            ParsedQuestion {
                intent: query.to_string(),
                filters: vec![],
            }
        }),
        Err(err) => {
            warn!("LLM parser offline ({}). Using raw query as intent without filters.", err);
            ParsedQuestion {
                intent: query.to_string(),
                filters: vec![],
            }
        }
    };

    info!("Loading embedding model (BAAI/bge-base-en-v1.5)...");
    let (model, tokenizer) = load_model().await?;
    
    let chunks = execute_semantic_search(&parsed_query.intent, "data/sap_vectors", &parsed_query.filters, Arc::clone(&model), Arc::clone(&tokenizer)).await?;
    
    info!("Passing retrieved chunks to LLM for deterministic generation...");
    let semantic_prompt = build_semantic_prompt(query, &chunks);
    match llm_client.prompt_model(&semantic_prompt).await {
        Ok(raw_llm_output) => {
            let mut final_payload = verify_and_parse_llm_generation(&raw_llm_output, &chunks)?;
            
            if !final_payload.answer_found {
                info!("Vector search failed. Triggering deterministic fallback (Absence Proof)...");
                let fallback_chunks = execute_fallback_search(&parsed_query.intent, "data/sap_vectors").await?;
                
                if !fallback_chunks.is_empty() {
                    info!("Fallback search found missing chunks. Re-querying LLM...");
                    let fallback_prompt = build_semantic_prompt(query, &fallback_chunks);
                    if let Ok(raw_fallback_output) = llm_client.prompt_model(&fallback_prompt).await {
                        final_payload = verify_and_parse_llm_generation(&raw_fallback_output, &fallback_chunks)?;
                    }
                } else {
                    info!("Absence mathematically proven. The data does not exist in the corpus.");
                }
            }
            
            println!("{}", serde_json::to_string(&final_payload)?);
        }
        Err(err) => {
            warn!("LLM generation offline ({}). Returning raw vector chunks (degraded mode).", err);
            let degraded_chunks = chunks
                .into_iter()
                .map(|(chunk_id, content)| DegradedChunk {
                    chunk_id,
                    content: content.to_string(),
                })
                .collect();
            let degraded_payload = DegradedResponse {
                degraded: true,
                message: format!("LLM is offline or timed out. Raw LanceDB vector search results returned: {}", err),
                retrieved_chunks: degraded_chunks,
            };
            println!("{}", serde_json::to_string(&degraded_payload)?);
        }
    }

    Ok(())
}
