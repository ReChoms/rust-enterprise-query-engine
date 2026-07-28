mod cli;
mod db;
mod debug;
mod embeddings;
mod ingest;
mod llm;
mod models;

#[cfg(test)]
mod tests;

use clap::Parser;
use anyhow::Result;
use std::sync::Arc;
use tracing::info;

use crate::cli::{Cli, Commands};
use crate::db::{execute_fallback_search, execute_semantic_search, execute_sql_query, init_datafusion};
use crate::embeddings::load_model;
use crate::ingest::execute_ingestion;
use crate::llm::{ask_llm, build_question_parser_prompt, build_routing_prompt, build_sql_prompt, build_semantic_prompt, verify_and_parse_llm_generation, parse_llm_json};
use crate::models::{ParsedQuestion, RouterDecision, SqlResponse};

#[tokio::main]
async fn main() -> Result<()> {
    debug::init_logger();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Ingest { file, overwrite, batch_size } => {
            info!("Loading embedding model (BAAI/bge-base-en-v1.5)...");
            let (model, tokenizer) = load_model().await?;
            execute_ingestion(file, "data/sap_vectors", *overwrite, *batch_size, Arc::clone(&model), Arc::clone(&tokenizer)).await?;
        }
        Commands::AskSemantic { query } => {
            info!(">>> Executing ASK-SEMANTIC command");
            
            info!("Parsing raw query to separate semantic intent from exact filters...");
            let parser_prompt = build_question_parser_prompt(&query);
            let raw_parser_output = ask_llm(&parser_prompt).await?;
            let parsed_query: ParsedQuestion = parse_llm_json(&raw_parser_output)?;

            info!("Loading embedding model (BAAI/bge-base-en-v1.5)...");
            let (model, tokenizer) = load_model().await?;
            
            // We execute the vector math strictly on the `intent`, 
            // and pass `filters` for the exact SQL string match.
            let chunks = execute_semantic_search(&parsed_query.intent, "data/sap_vectors", &parsed_query.filters, Arc::clone(&model), Arc::clone(&tokenizer)).await?;
            
            info!("Passing retrieved chunks to LLM for deterministic generation...");
            // We still pass the original full `query` to the final answering LLM so it knows 
            // about the user's constraints (e.g. "NOT 1000") when generating the final answer.
            let semantic_prompt = build_semantic_prompt(&query, &chunks);
            let raw_llm_output = ask_llm(&semantic_prompt).await?;
            let mut final_payload = verify_and_parse_llm_generation(&raw_llm_output, &chunks)?;
            
            if !final_payload.answer_found {
                info!("Vector search failed. Triggering deterministic fallback (Absence Proof)...");
                let fallback_chunks = execute_fallback_search(&parsed_query.intent, "data/sap_vectors").await?;
                
                if !fallback_chunks.is_empty() {
                    info!("Fallback search found missing chunks. Re-querying LLM...");
                    let fallback_prompt = build_semantic_prompt(&query, &fallback_chunks);
                    let raw_fallback_output = ask_llm(&fallback_prompt).await?;
                    final_payload = verify_and_parse_llm_generation(&raw_fallback_output, &fallback_chunks)?;
                } else {
                    info!("Absence mathematically proven. The data does not exist in the corpus.");
                }
            }
            
            println!("{}", serde_json::to_string(&final_payload)?);
        }
        Commands::Ask { query } => {
            info!(">>> Executing ASK (ROUTER) command");
            let full_prompt = build_routing_prompt(&query);
            let raw_json = ask_llm(&full_prompt).await?;
            let decision: RouterDecision = serde_json::from_str(&raw_json)?;

            if decision.route == "SQL" {
                let sql_engine = init_datafusion().await?;
                execute_sql_query(&sql_engine, &decision.query).await?;
            } else {
                info!("Parsing raw query to separate semantic intent from exact filters...");
                let parser_prompt = build_question_parser_prompt(&query);
                let raw_parser_output = ask_llm(&parser_prompt).await?;
                let parsed_query: ParsedQuestion = parse_llm_json(&raw_parser_output)?;

                info!("Loading embedding model (BAAI/bge-base-en-v1.5)...");
                let (model, tokenizer) = load_model().await?;
                
                // Pass parsed intent and exact filters instead of raw query
                let chunks = execute_semantic_search(&parsed_query.intent, "data/sap_vectors", &parsed_query.filters, Arc::clone(&model), Arc::clone(&tokenizer)).await?;
                
                info!("Passing retrieved chunks to LLM for deterministic generation...");
                let semantic_prompt = build_semantic_prompt(&query, &chunks);
                let raw_llm_output = ask_llm(&semantic_prompt).await?;
                let mut final_payload = verify_and_parse_llm_generation(&raw_llm_output, &chunks)?;
                
                if !final_payload.answer_found {
                    info!("Vector search failed. Triggering deterministic fallback (Absence Proof)...");
                    let fallback_chunks = execute_fallback_search(&parsed_query.intent, "data/sap_vectors").await?;
                    
                    if !fallback_chunks.is_empty() {
                        info!("Fallback search found missing chunks. Re-querying LLM...");
                        let fallback_prompt = build_semantic_prompt(&query, &fallback_chunks);
                        let raw_fallback_output = ask_llm(&fallback_prompt).await?;
                        final_payload = verify_and_parse_llm_generation(&raw_fallback_output, &fallback_chunks)?;
                    } else {
                        info!("Absence mathematically proven. The data does not exist in the corpus.");
                    }
                }
                
                println!("{}", serde_json::to_string(&final_payload)?);
            }
        }
        Commands::AskSql { query } => {
            info!(">>> Executing ASK-SQL command");
            let sql_engine = init_datafusion().await?;
            execute_sql_query(&sql_engine, &query).await?;
        }
        Commands::AskAiSql { query } => {
            info!(">>> Executing ASK-AISQL command");
            let full_prompt = build_sql_prompt(&query);

            let raw_json = ask_llm(&full_prompt).await?;
            let response: SqlResponse = parse_llm_json(&raw_json)?;
            let sql_engine = init_datafusion().await?;
            execute_sql_query(&sql_engine, &response.query).await?;
        }
    }

    Ok(())
}
