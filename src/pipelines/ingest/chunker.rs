use anyhow::Result;
use arrow_schema::{DataType, Field, Schema};
use candle_transformers::models::bert::BertModel;
use std::collections::HashSet;
use std::fs::File;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::info;

use crate::common::types::Kna1Row;
use crate::engines::embeddings::get_embeddings;
use crate::engines::vector::insert_batch;
use super::dedup::find_existing_in_db;

/// Defines the strict Apache Arrow Memory Schema for LanceDB.
pub fn define_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
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
    ]))
}

/// Ingests accumulated CSV rows in batches, deduplicates, and embeds delta records.
pub async fn process_csv_in_batches(
    csv_reader: &mut csv::Reader<File>,
    vector_db: &lancedb::Connection,
    schema: Arc<Schema>,
    model: Arc<BertModel>,
    tokenizer: Arc<Tokenizer>,
    batch_size: usize,
) -> Result<usize> {
    let mut batch_records: Vec<Kna1Row> = Vec::new();
    let mut total_inserted = 0;

    for result in csv_reader.deserialize() {
        batch_records.push(result?);
        if batch_records.len() >= batch_size {
            total_inserted += process_current_batch(
                vector_db,
                schema.clone(),
                Arc::clone(&model),
                Arc::clone(&tokenizer),
                &batch_records,
            )
            .await?;
            batch_records.clear();
        }
    }

    if !batch_records.is_empty() {
        total_inserted += process_current_batch(
            vector_db,
            schema.clone(),
            Arc::clone(&model),
            Arc::clone(&tokenizer),
            &batch_records,
        )
        .await?;
    }

    Ok(total_inserted)
}

async fn process_current_batch(
    vector_db: &lancedb::Connection,
    schema: Arc<Schema>,
    model: Arc<BertModel>,
    tokenizer: Arc<Tokenizer>,
    batch_records: &[Kna1Row],
) -> Result<usize> {
    let mut ids_for_sql = Vec::new();
    for row in batch_records {
        if let Some(customer_id) = &row.kunnr {
            ids_for_sql.push(format!("'{}'", customer_id));
        }
    }

    let existing_in_db = find_existing_in_db(vector_db, &ids_for_sql).await;

    let mut documents = Vec::new();
    let mut records = Vec::new();
    let mut local_seen = HashSet::new();

    for row in batch_records {
        let kunnr = row.kunnr.clone().unwrap_or_else(|| "Unknown".to_string());
        if !local_seen.insert(kunnr.clone()) || existing_in_db.contains(&kunnr) {
            continue;
        }

        let name = row.name1.clone().unwrap_or_else(|| "Unknown".to_string());
        let city = row.ort01.clone().unwrap_or_else(|| "Unknown".to_string());
        let country = row.land1.clone().unwrap_or_else(|| "Unknown".to_string());

        documents.push(format!(
            "Customer {} is named {} and is located in {}, {}.",
            kunnr, name, city, country
        ));
        records.push((name, city, kunnr));
    }

    if documents.is_empty() {
        return Ok(0);
    }

    info!("Embedding and inserting batch of {} records...", documents.len());
    let embeddings = get_embeddings(documents.clone(), tokenizer, model).await?;
    insert_batch(vector_db, schema, &records, &documents, embeddings).await?;

    Ok(documents.len())
}
