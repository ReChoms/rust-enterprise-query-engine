//! Ingestion Pipeline (Altitude 2: Write River)
//!
//! Orchestrates CSV reading, JIT deduplication, Arrow schema validation,
//! dense vector embedding computation, and merge insertion into LanceDB.

pub mod chunker;
pub mod dedup;

use anyhow::{anyhow, Result};
use candle_transformers::models::bert::BertModel;
use std::fs::File;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::info;

pub use chunker::{define_schema, process_csv_in_batches};
pub use dedup::find_existing_in_db;

/// Main entry point for the ingestion write pipeline.
pub async fn execute_ingestion(
    csv_path: &str,
    db_uri: &str,
    overwrite: bool,
    batch_size: usize,
    model: Arc<BertModel>,
    tokenizer: Arc<Tokenizer>,
) -> Result<()> {
    info!(">>> Executing INGEST command on file: {}", csv_path);

    info!("Connecting to LanceDB...");
    let vector_db = lancedb::connect(db_uri).execute().await?;

    if overwrite {
        info!("Overwrite flag detected. Dropping existing 'customers' table...");
        let _ = vector_db.drop_table("customers").await;
    }

    let schema = define_schema();

    info!("Reading {} with dynamic chunk size {}...", csv_path, batch_size);
    let file_stream = File::open(csv_path)
        .map_err(|e| anyhow!("File load went wrong. Error: {}", e))?;
    let mut csv_reader = csv::Reader::from_reader(file_stream);

    let total_inserted = process_csv_in_batches(
        &mut csv_reader,
        &vector_db,
        schema,
        model,
        tokenizer,
        batch_size,
    )
    .await?;

    info!(
        "Successfully ingested {} total new records into LanceDB 'customers' table!",
        total_inserted
    );
    Ok(())
}
