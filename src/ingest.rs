use arrow_array::Array;
use arrow_schema::{DataType, Field, Schema};
use lancedb::query::ExecutableQuery;
use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use std::sync::Arc;
use tracing::info;

use crate::db::insert_batch;
use crate::embeddings::get_embeddings;
use crate::models::Kna1Row;

/// Main entry point for the ingestion pipeline
pub async fn execute_ingestion(
    csv_path: &str,
    overwrite: bool,
    batch_size: usize,
    model: &candle_transformers::models::bert::BertModel,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<(), Box<dyn Error>> {
    info!(">>> Executing INGEST command on file: {}", csv_path);

    info!("Connecting to LanceDB...");
    let db = lancedb::connect("data/sap_vectors")
        .execute()
        .await
        .map_err(|e| Box::<dyn Error>::from(e.to_string()))?;

    if overwrite {
        info!("Overwrite flag detected. Dropping existing 'customers' table...");
        let _ = db.drop_table("customers").await;
    }

    let schema = define_schema();

    info!("Reading {} with dynamic chunk size {}...", csv_path, batch_size);
    let file_stream = File::open(csv_path)
        .map_err(|e| format!("File load went wrong. Rust shows the following error: {}", e))?;
    let mut rdr = csv::Reader::from_reader(file_stream);

    process_csv_in_batches(&mut rdr, &db, schema, &model, &tokenizer, batch_size).await?;

    Ok(())
}



/// Helper function defining the strict Apache Arrow Memory Schema
fn define_schema() -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("kunnr", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("city", DataType::Utf8, false),
        Field::new("sentence", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                768,
            ),
            false,
        ),
    ]));
    info!("LanceDB schema defined successfully.");
    schema
}

/// The outer loop that only accumulates raw CSV rows into an array
async fn process_csv_in_batches(
    rdr: &mut csv::Reader<File>,
    db: &lancedb::Connection,
    schema: Arc<Schema>,
    model: &candle_transformers::models::bert::BertModel,
    tokenizer: &tokenizers::Tokenizer,
    batch_size: usize,
) -> Result<(), Box<dyn Error>> {
    let mut batch_records: Vec<Kna1Row> = Vec::new();
    let mut total_inserted = 0;

    for result in rdr.deserialize() {
        batch_records.push(result?);
        if batch_records.len() >= batch_size {
            total_inserted += process_current_batch(db, schema.clone(), model, tokenizer, &batch_records).await?;
            batch_records.clear();
        }
    }

    if !batch_records.is_empty() {
        total_inserted += process_current_batch(db, schema.clone(), model, tokenizer, &batch_records).await?;
    }

    info!("Successfully ingested {} total new records into LanceDB 'customers' table!", total_inserted);
    Ok(())
}

/// Helper function to perform Just-In-Time SQL lookup on a specific batch of IDs
async fn find_existing_in_db(db: &lancedb::Connection, ids_for_sql: &[String]) -> HashSet<String> {
    let mut existing = HashSet::new();
    if ids_for_sql.is_empty() { return existing; }

    if let Ok(table) = db.open_table("customers").execute().await {
        let filter = format!("kunnr IN ({})", ids_for_sql.join(", "));
        use futures::StreamExt;
        if let Ok(mut stream) = table.query().filter(filter).execute().await {
            while let Some(batch) = stream.next().await {
                if let Ok(batch) = batch {
                    if let Some(col) = batch.column_by_name("kunnr") {
                        if let Some(k_arr) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
                            for i in 0..k_arr.len() {
                                existing.insert(k_arr.value(i).to_string());
                            }
                        }
                    }
                }
            }
        }
    }
    existing
}

/// The core logic that filters duplicates and runs the heavy embedding math strictly on the delta
async fn process_current_batch(
    db: &lancedb::Connection,
    schema: Arc<Schema>,
    model: &candle_transformers::models::bert::BertModel,
    tokenizer: &tokenizers::Tokenizer,
    batch_records: &[Kna1Row],
) -> Result<usize, Box<dyn Error>> {
    let mut ids_for_sql = Vec::new();
    for row in batch_records {
        if let Some(k) = &row.kunnr { ids_for_sql.push(format!("'{}'", k)); }
    }

    let existing_in_db = find_existing_in_db(db, &ids_for_sql).await;
    
    let mut documents = Vec::new();
    let mut records = Vec::new();
    let mut local_seen = HashSet::new();

    for row in batch_records {
        let kunnr = row.kunnr.clone().unwrap_or_else(|| "Unknown".to_string());
        if !local_seen.insert(kunnr.clone()) || existing_in_db.contains(&kunnr) { continue; }

        let name = row.name1.clone().unwrap_or_else(|| "Unknown".to_string());
        let city = row.ort01.clone().unwrap_or_else(|| "Unknown".to_string());
        let country = row.land1.clone().unwrap_or_else(|| "Unknown".to_string());

        documents.push(format!("Customer {} is named {} and is located in {}, {}.", kunnr, name, city, country));
        records.push((name, city, kunnr));
    }

    if documents.is_empty() { return Ok(0); }

    let embeddings = get_embeddings(&documents, tokenizer, model)?;
    insert_batch(db, schema, &records, &documents, embeddings).await?;

    Ok(documents.len())
}
