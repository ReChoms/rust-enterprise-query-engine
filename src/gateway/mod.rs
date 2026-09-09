//! Altitude 3: Ingress Gateway
//!
//! Entry points for the application:
//! - Terminal CLI & Piped STDIN (`gateway::cli`)
//! - Axum REST Microservice (`gateway::http`)

pub mod cli;
pub mod http;

pub use cli::{resolve_query_inputs, Cli, Commands};
pub use http::{create_router, create_tuned_tcp_listener, start_server};

use anyhow::Result;
use clap::Parser;
use datafusion::prelude::SessionContext;
use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::{info, warn};

use crate::common::types::{HealthResponse, RouterDecision};
use crate::engines::embeddings::load_model;
use crate::engines::sql::{execute_sql_query, init_datafusion, write_record_batches_as_json_lines};
use crate::engines::vector::check_lancedb_health;
use crate::pipelines::ingest::execute_ingestion;
use crate::pipelines::query::{
    build_routing_prompt, parse_llm_json, run_semantic_pipeline, run_sql_query, AppState,
    OllamaClient,
};

/// Main application dispatcher for CLI commands and HTTP server.
pub async fn run() -> Result<()> {
    let cli = Cli::parse();
    let vector_db_uri = std::env::var("VECTOR_DB_URI")
        .unwrap_or_else(|_| "data/sap_vectors".to_string());
    let llm_client = OllamaClient::init_from_env_or_default()?;

    match &cli.command {
        Commands::Ingest {
            file,
            overwrite,
            batch_size,
        } => {
            info!("Loading embedding model (BAAI/bge-base-en-v1.5)...");
            let (model, tokenizer) = load_model().await?;
            execute_ingestion(
                file,
                &vector_db_uri,
                *overwrite,
                *batch_size,
                Arc::clone(&model),
                Arc::clone(&tokenizer),
            )
            .await?;
        }
        Commands::AskSemantic { query } => {
            info!(">>> Executing ASK-SEMANTIC command");
            info!("Loading embedding model (BAAI/bge-base-en-v1.5)...");
            let (model, tokenizer) = load_model().await?;
            let state = AppState {
                llm_client: Arc::new(llm_client),
                sql_engine: Arc::new(SessionContext::new()),
                embedding_model: model,
                tokenizer,
                vector_db_uri,
            };
            let queries = resolve_query_inputs(query)?;
            for q_res in queries {
                let q = q_res?;
                let res = run_semantic_pipeline(&state, &q).await?;
                println!("{}", serde_json::to_string(&res)?);
            }
        }
        Commands::Ask { query } => {
            info!(">>> Executing ASK (ROUTER) command");
            let queries = resolve_query_inputs(query)?;
            let mut sql_engine: Option<SessionContext> = None;
            let mut model_bundle: Option<(
                Arc<candle_transformers::models::bert::BertModel>,
                Arc<tokenizers::Tokenizer>,
            )> = None;

            for q_res in queries {
                let q = q_res?;
                let full_prompt = build_routing_prompt(&q);
                match llm_client.prompt_model(&full_prompt).await {
                    Ok(raw_json) => {
                        let decision: RouterDecision =
                            parse_llm_json(&raw_json).unwrap_or_else(|err| {
                                warn!(
                                    "Failed to parse router output '{}': {}. Defaulting to SEMANTIC route.",
                                    raw_json, err
                                );
                                RouterDecision {
                                    route: "SEMANTIC".to_string(),
                                }
                            });

                        if decision.route == "SQL" {
                            if sql_engine.is_none() {
                                sql_engine = Some(init_datafusion().await?);
                            }
                            if let Some(engine) = sql_engine.as_ref() {
                                let batches = run_sql_query(&llm_client, engine, &q).await?;
                                let mut stdout = io::stdout();
                                write_record_batches_as_json_lines(&batches, &mut stdout)?;
                            }
                        } else {
                            if model_bundle.is_none() {
                                info!("Loading embedding model (BAAI/bge-base-en-v1.5)...");
                                model_bundle = Some(load_model().await?);
                            }
                            if let Some((model, tokenizer)) = model_bundle.as_ref() {
                                let state = AppState {
                                    llm_client: Arc::new(llm_client.clone()),
                                    sql_engine: Arc::new(SessionContext::new()),
                                    embedding_model: Arc::clone(model),
                                    tokenizer: Arc::clone(tokenizer),
                                    vector_db_uri: vector_db_uri.clone(),
                                };
                                let res = run_semantic_pipeline(&state, &q).await?;
                                println!("{}", serde_json::to_string(&res)?);
                            }
                        }
                    }
                    Err(err) => {
                        warn!(
                            "Ollama router unreachable ({}). Falling back to degraded semantic search.",
                            err
                        );
                        if model_bundle.is_none() {
                            info!("Loading embedding model (BAAI/bge-base-en-v1.5)...");
                            model_bundle = Some(load_model().await?);
                        }
                        if let Some((model, tokenizer)) = model_bundle.as_ref() {
                            let state = AppState {
                                llm_client: Arc::new(llm_client.clone()),
                                sql_engine: Arc::new(SessionContext::new()),
                                embedding_model: Arc::clone(model),
                                tokenizer: Arc::clone(tokenizer),
                                vector_db_uri: vector_db_uri.clone(),
                            };
                            let res = run_semantic_pipeline(&state, &q).await?;
                            println!("{}", serde_json::to_string(&res)?);
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
                let batches = run_sql_query(&llm_client, &sql_engine, &q).await?;
                let mut stdout = io::stdout();
                write_record_batches_as_json_lines(&batches, &mut stdout)?;
            }
        }
        Commands::Health => {
            info!(">>> Executing HEALTH command");
            let llm_online = llm_client.is_healthy().await;
            let lancedb_stats = check_lancedb_health(&vector_db_uri).await;

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
            let addr: SocketAddr = addr_str.parse()?;
            info!("Booting Axum REST API microservice on {}", addr);
            info!("Loading embedding model (BAAI/bge-base-en-v1.5)...");
            let (model, tokenizer) = load_model().await?;
            let sql_engine = init_datafusion().await?;
            info!("Connecting to LanceDB vector storage at '{}'...", vector_db_uri);
            let state = AppState {
                llm_client: Arc::new(llm_client),
                sql_engine: Arc::new(sql_engine),
                embedding_model: model,
                tokenizer,
                vector_db_uri,
            };
            start_server(state, addr).await?;
        }
    }

    Ok(())
}
