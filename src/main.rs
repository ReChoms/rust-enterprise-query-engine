mod cli;
mod debug;
mod embeddings;
mod ingest;
mod llm;
mod sql_engine;
mod types;
mod vector_db;
mod server;

#[cfg(test)]
mod tests;

use clap::Parser;
use anyhow::{bail, Result};
use std::io::{self, BufRead, IsTerminal};
use std::sync::Arc;
use tracing::{info, warn};

use crate::cli::{Cli, Commands};
use crate::embeddings::load_model;
use crate::ingest::execute_ingestion;
use crate::llm::{build_question_parser_prompt, build_routing_prompt, build_semantic_prompt, build_sql_prompt, parse_llm_json, verify_and_parse_llm_generation, OllamaClient};
use crate::sql_engine::{execute_sql_query, init_datafusion, write_record_batches_as_json_lines};
use crate::types::{DegradedChunk, DegradedResponse, HealthResponse, ParsedQuestion, RouterDecision, SqlResponse};
use crate::vector_db::{check_lancedb_health, execute_fallback_search, execute_semantic_search};
use datafusion::prelude::SessionContext;

fn resolve_query_inputs(cli_arg: &Option<String>) -> Result<Box<dyn Iterator<Item = Result<String>>>> {
    if let Some(q) = cli_arg {
        let trimmed = q.trim();
        if trimmed.is_empty() {
            bail!("Provided query is empty.");
        }
        return Ok(Box::new(std::iter::once(Ok(trimmed.to_string()))));
    }

    if io::stdin().is_terminal() {
        bail!("No query provided. Pass a query argument or stream via STDIN (e.g. echo '...' | bridge ask).");
    }

    let stdin = io::stdin();
    let iter = stdin
        .lock()
        .lines()
        .map(|res| res.map_err(|e| anyhow::anyhow!("Failed reading from STDIN: {}", e)))
        .filter_map(|res| match res {
            Ok(line) => {
                let trimmed = line.trim().to_string();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(Ok(trimmed))
                }
            }
            Err(e) => Some(Err(e)),
        });

    Ok(Box::new(iter))
}

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
            info!("Loading embedding model (BAAI/bge-base-en-v1.5)...");
            let (model, tokenizer) = load_model().await?;
            let queries = resolve_query_inputs(query)?;
            for q_res in queries {
                let q = q_res?;
                run_semantic_pipeline(&llm_client, Arc::clone(&model), Arc::clone(&tokenizer), &q).await?;
            }
        }
        Commands::Ask { query } => {
            info!(">>> Executing ASK (ROUTER) command");
            let queries = resolve_query_inputs(query)?;
            let mut sql_engine: Option<SessionContext> = None;
            let mut model_bundle: Option<(Arc<candle_transformers::models::bert::BertModel>, Arc<tokenizers::Tokenizer>)> = None;
            for q_res in queries {
                let q = q_res?;
                let full_prompt = build_routing_prompt(&q);
                match llm_client.prompt_model(&full_prompt).await {
                    Ok(raw_json) => {
                        let decision: RouterDecision = parse_llm_json(&raw_json).unwrap_or_else(|err| {
                            warn!("Failed to parse router output '{}': {}. Defaulting to SEMANTIC route.", raw_json, err);
                            RouterDecision {
                                route: "SEMANTIC".to_string(),
                            }
                        });

                        if decision.route == "SQL" {
                            if sql_engine.is_none() {
                                sql_engine = Some(init_datafusion().await?);
                            }
                            if let Some(engine) = sql_engine.as_ref() {
                                run_sql_pipeline(&llm_client, engine, &q).await?;
                            }
                        } else {
                            if model_bundle.is_none() {
                                info!("Loading embedding model (BAAI/bge-base-en-v1.5)...");
                                model_bundle = Some(load_model().await?);
                            }
                            if let Some((model, tokenizer)) = model_bundle.as_ref() {
                                run_semantic_pipeline(&llm_client, Arc::clone(model), Arc::clone(tokenizer), &q).await?;
                            }
                        }
                    }
                    Err(err) => {
                        warn!("Ollama router unreachable ({}). Falling back to degraded semantic search.", err);
                        if model_bundle.is_none() {
                            info!("Loading embedding model (BAAI/bge-base-en-v1.5)...");
                            model_bundle = Some(load_model().await?);
                        }
                        if let Some((model, tokenizer)) = model_bundle.as_ref() {
                            run_semantic_pipeline(&llm_client, Arc::clone(model), Arc::clone(tokenizer), &q).await?;
                        }
                    }
                }
            }
        }
        Commands::ExecuteSql { query } => {
            info!(">>> Executing EXECUTE-SQL command");
            let queries = resolve_query_inputs(query)?;
            let sql_engine = init_datafusion().await?;
            let mut stdout = io::stdout();
            for q_res in queries {
                let q = q_res?;
                let batches = execute_sql_query(&sql_engine, &q).await?;
                write_record_batches_as_json_lines(&batches, &mut stdout)?;
            }
        }
        Commands::AskAiSql { query } => {
            info!(">>> Executing ASK-AISQL command");
            let queries = resolve_query_inputs(query)?;
            let sql_engine = init_datafusion().await?;
            for q_res in queries {
                let q = q_res?;
                run_sql_pipeline(&llm_client, &sql_engine, &q).await?;
            }
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
        Commands::Serve { host, port } => {
            let addr_str = format!("{}:{}", host, port);
            let addr: std::net::SocketAddr = addr_str.parse()?;
            info!("Booting Axum REST API microservice on {}", addr);
            info!("Loading embedding model (BAAI/bge-base-en-v1.5)...");
            let (model, tokenizer) = load_model().await?;
            let sql_engine = init_datafusion().await?;
            let vector_db_uri = std::env::var("VECTOR_DB_URI").unwrap_or_else(|_| "data/sap_vectors".to_string());
            info!("Connecting to LanceDB vector storage at '{}'...", vector_db_uri);
            let state = server::AppState {
                llm_client: Arc::new(llm_client),
                sql_engine: Arc::new(sql_engine),
                embedding_model: Arc::clone(&model),
                tokenizer: Arc::clone(&tokenizer),
                vector_db_uri,
            };
            server::start_server(state, addr).await?;
        }
    }

    Ok(())
}

async fn run_sql_pipeline(llm_client: &OllamaClient, sql_engine: &SessionContext, query: &str) -> Result<()> {
    let full_prompt = build_sql_prompt(query);

    let raw_json = match llm_client.prompt_model(&full_prompt).await {
        Ok(json) => json,
        Err(err) => bail!("Cannot execute AI SQL query: Ollama is offline or timed out: {}", err),
    };
    let response: SqlResponse = parse_llm_json(&raw_json)?;
    let batches = execute_sql_query(sql_engine, &response.query).await?;
    let mut stdout = io::stdout();
    write_record_batches_as_json_lines(&batches, &mut stdout)?;
    Ok(())
}

async fn run_semantic_pipeline(
    llm_client: &OllamaClient,
    model: Arc<candle_transformers::models::bert::BertModel>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    query: &str,
) -> Result<()> {
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

    let chunks = execute_semantic_search(&parsed_query.intent, "data/sap_vectors", &parsed_query.filters, model, tokenizer).await?;
    
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
