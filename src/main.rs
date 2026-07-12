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
use crate::llm::{ask_llm, build_question_parser_prompt, build_routing_prompt, build_sql_prompt, build_semantic_prompt, verify_and_parse_llm_generation, parse_llm_json};
use crate::models::{ParsedQuestion, RouterDecision, SqlResponse};

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
            
            info!("Parsing raw query to separate semantic intent from exact filters...");
            let parser_prompt = build_question_parser_prompt(&query);
            let raw_parser_output = ask_llm(&parser_prompt).await?;
            let parsed_query: ParsedQuestion = parse_llm_json(&raw_parser_output)?;

            info!("Loading embedding model (BAAI/bge-base-en-v1.5)...");
            let (model, tokenizer) = load_model()?;
            
            // CRITICAL FIX: We execute the vector math strictly on the `intent`, 
            // and pass `filters` for the exact SQL string match.
            let chunks = execute_semantic_search(&parsed_query.intent, &parsed_query.filters, &model, &tokenizer).await?;
            
            info!("Passing retrieved chunks to LLM for deterministic generation...");
            // We still pass the original full `query` to the final answering LLM so it knows 
            // about the user's constraints (e.g. "NOT 1000") when generating the final answer.
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
                info!("Parsing raw query to separate semantic intent from exact filters...");
                let parser_prompt = build_question_parser_prompt(&query);
                let raw_parser_output = ask_llm(&parser_prompt).await?;
                let parsed_query: ParsedQuestion = parse_llm_json(&raw_parser_output)?;

                info!("Loading embedding model (BAAI/bge-base-en-v1.5)...");
                let (model, tokenizer) = load_model()?;
                
                // Pass parsed intent and exact filters instead of raw query
                let chunks = execute_semantic_search(&parsed_query.intent, &parsed_query.filters, &model, &tokenizer).await?;
                
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
            let response: SqlResponse = parse_llm_json(&raw_json)?;
            execute_sql_query(&response.query).await?;
        }
    }

    Ok(())
}
