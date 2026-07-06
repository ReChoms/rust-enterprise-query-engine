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
use std::error::Error;
use tracing::info;

use crate::cli::{Cli, Commands};
use crate::db::{execute_semantic_search, execute_sql_query};
use crate::embeddings::load_model;
use crate::ingest::execute_ingestion;
use crate::llm::{ask_llm, build_routing_prompt, build_sql_prompt, build_semantic_prompt, verify_and_parse_llm_generation, verify_and_parse_sql_generation};
use crate::models::RouterDecision;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    debug::init_logger();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Ingest { file, overwrite, batch_size } => {
            info!("Loading embedding model (BAAI/bge-base-en-v1.5)...");
            let (model, tokenizer) = load_model()?;
            execute_ingestion(file, *overwrite, *batch_size, &model, &tokenizer).await?;
        }
        Commands::AskSemantic { query } => {
            info!(">>> Executing ASK-SEMANTIC command");
            info!("Loading embedding model (BAAI/bge-base-en-v1.5)...");
            let (model, tokenizer) = load_model()?;
            let chunks = execute_semantic_search(&query, &model, &tokenizer).await?;
            
            info!("Passing retrieved chunks to LLM for deterministic generation...");
            let semantic_prompt = build_semantic_prompt(&query, &chunks);
            let raw_llm_output = ask_llm(&semantic_prompt).await?;
            let final_payload = verify_and_parse_llm_generation(&raw_llm_output, &chunks)?;
            
            println!("{}", serde_json::to_string(&final_payload)?);
        }
        Commands::Ask { query } => {
            info!(">>> Executing ASK (ROUTER) command");
            let full_prompt = build_routing_prompt(&query);
            let raw_json = ask_llm(&full_prompt).await?;
            let decision: RouterDecision = serde_json::from_str(&raw_json)?;

            if decision.route == "SQL" {
                execute_sql_query(&decision.query).await?;
            } else {
                info!("Loading embedding model (BAAI/bge-base-en-v1.5)...");
                let (model, tokenizer) = load_model()?;
                let chunks = execute_semantic_search(&query, &model, &tokenizer).await?;
                
                info!("Passing retrieved chunks to LLM for deterministic generation...");
                let semantic_prompt = build_semantic_prompt(&query, &chunks);
                let raw_llm_output = ask_llm(&semantic_prompt).await?;
                let final_payload = verify_and_parse_llm_generation(&raw_llm_output, &chunks)?;
                
                println!("{}", serde_json::to_string(&final_payload)?);
            }
        }
        Commands::AskSql { query } => {
            info!(">>> Executing ASK-SQL command");
            execute_sql_query(&query).await?;
        }
        Commands::AskAiSql { query } => {
            info!(">>> Executing ASK-AISQL command");
            let full_prompt = build_sql_prompt(&query);

            let raw_json = ask_llm(&full_prompt).await?;
            let response = verify_and_parse_sql_generation(&raw_json)?;
            execute_sql_query(&response.query).await?;
        }
    }

    Ok(())
}
