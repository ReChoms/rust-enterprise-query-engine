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
use anyhow::Result;
use std::sync::Arc;
use tracing::info;

use crate::cli::{Cli, Commands};
use crate::db::{execute_fallback_search, execute_semantic_search, execute_sql_query, init_datafusion};
use crate::embeddings::load_model;
use crate::ingest::execute_ingestion;
use crate::llm::{ask_llm, build_question_parser_prompt, build_routing_prompt, build_sql_prompt, build_semantic_prompt, verify_and_parse_llm_generation, parse_llm_json};
use crate::types::{ParsedQuestion, RouterDecision, SqlResponse};

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
            run_semantic_pipeline(query).await?;
        }
        Commands::Ask { query } => {
            info!(">>> Executing ASK (ROUTER) command");
            let full_prompt = build_routing_prompt(&query);
            let raw_json = ask_llm(&full_prompt).await?;
            let decision: RouterDecision = serde_json::from_str(&raw_json)?;

            if decision.route == "SQL" {
                run_sql_pipeline(query).await?;
            } else {
                run_semantic_pipeline(query).await?;
            }
        }
        Commands::ExecuteSql { query } => {
            info!(">>> Executing EXECUTE-SQL command");
            let sql_engine = init_datafusion().await?;
            execute_sql_query(&sql_engine, &query).await?;
        }
        Commands::AskAiSql { query } => {
            info!(">>> Executing ASK-AISQL command");
            run_sql_pipeline(query).await?;
        }
    }

    Ok(())
}

async fn run_sql_pipeline(query: &str) -> Result<()> {
    let full_prompt = build_sql_prompt(query);

    let raw_json = ask_llm(&full_prompt).await?;
    let response: SqlResponse = parse_llm_json(&raw_json)?;
    let sql_engine = init_datafusion().await?;
    execute_sql_query(&sql_engine, &response.query).await?;
    Ok(())
}

async fn run_semantic_pipeline(query: &str) -> Result<()> {
    info!("Parsing raw query to separate semantic intent from exact filters...");
    let parser_prompt = build_question_parser_prompt(query);
    let raw_parser_output = ask_llm(&parser_prompt).await?;
    let parsed_query: ParsedQuestion = parse_llm_json(&raw_parser_output)?;

    info!("Loading embedding model (BAAI/bge-base-en-v1.5)...");
    let (model, tokenizer) = load_model().await?;
    
    let chunks = execute_semantic_search(&parsed_query.intent, "data/sap_vectors", &parsed_query.filters, Arc::clone(&model), Arc::clone(&tokenizer)).await?;
    
    info!("Passing retrieved chunks to LLM for deterministic generation...");
    let semantic_prompt = build_semantic_prompt(query, &chunks);
    let raw_llm_output = ask_llm(&semantic_prompt).await?;
    let mut final_payload = verify_and_parse_llm_generation(&raw_llm_output, &chunks)?;
    
    if !final_payload.answer_found {
        info!("Vector search failed. Triggering deterministic fallback (Absence Proof)...");
        let fallback_chunks = execute_fallback_search(&parsed_query.intent, "data/sap_vectors").await?;
        
        if !fallback_chunks.is_empty() {
            info!("Fallback search found missing chunks. Re-querying LLM...");
            let fallback_prompt = build_semantic_prompt(query, &fallback_chunks);
            let raw_fallback_output = ask_llm(&fallback_prompt).await?;
            final_payload = verify_and_parse_llm_generation(&raw_fallback_output, &fallback_chunks)?;
        } else {
            info!("Absence mathematically proven. The data does not exist in the corpus.");
        }
    }
    
    println!("{}", serde_json::to_string(&final_payload)?);
    Ok(())
}
