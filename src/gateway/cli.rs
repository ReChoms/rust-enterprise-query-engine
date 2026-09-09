use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use std::io::{self, BufRead, IsTerminal};

#[derive(Parser)]
#[command(name = "rust-enterprise-query-engine")]
#[command(about = "Bridging SAP ERP data with Semantic AI Search", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Commands {
    /// Ingest a CSV file from SAP into the Vector Database
    Ingest {
        /// The path to the CSV file (e.g. data/kna1.csv)
        #[arg(short, long)]
        file: String,
        /// Overwrite the existing database instead of appending
        #[arg(short, long)]
        overwrite: bool,
        /// The dynamic batch size for ingestion chunking (default: 256)
        #[arg(short, long, default_value_t = 256)]
        batch_size: usize,
    },
    /// The Primary Router: dynamically chooses Semantic or SQL
    Ask {
        query: Option<String>,
    },
    /// Force a Semantic Vector Search
    AskSemantic {
        query: Option<String>,
    },
    /// Execute a raw SQL query against the SAP data
    ExecuteSql {
        query: Option<String>,
    },
    /// Force the LLM to generate and run a SQL query
    AskAiSql {
        query: Option<String>,
    },
    /// Check the health and connectivity of the LLM and Vector Database
    Health,
    /// Start the high-performance Axum REST API server
    Serve {
        /// The host address (CLI flag `--host` or env var `HOST`, default: "0.0.0.0")
        #[arg(short = 'H', long, env = "HOST", default_value = "0.0.0.0")]
        host: String,
        /// The port number (CLI flag `--port` or env var `PORT`, default_value_t = 8080)
        #[arg(short, long, env = "PORT", default_value_t = 8080)]
        port: u16,
    },
}

/// Resolves queries from CLI flag arguments or piped STDIN streams.
pub fn resolve_query_inputs(cli_arg: &Option<String>) -> Result<Box<dyn Iterator<Item = Result<String>>>> {
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
