use arrow_array::builder::{FixedSizeListBuilder, PrimitiveBuilder};
use arrow_array::types::Float32Type;
use arrow_array::{Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::Schema;
use datafusion::prelude::*;
use lancedb::query::{ExecutableQuery, QueryBase};
use sqlparser::ast::Statement;
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use std::error::Error;
use std::sync::Arc;
use tracing::info;

use crate::embeddings::get_embeddings;

pub async fn insert_batch(
    db: &lancedb::Connection,
    schema: Arc<Schema>,
    records: &[(String, String, String)],
    documents: &[String],
    embeddings: Vec<Vec<f32>>,
) -> Result<(), Box<dyn Error>> {
    let kunnr_array = StringArray::from(
        records
            .iter()
            .map(|r| Some(r.2.clone()))
            .collect::<Vec<_>>(),
    );
    let name_array = StringArray::from(
        records
            .iter()
            .map(|r| Some(r.0.clone()))
            .collect::<Vec<_>>(),
    );
    let city_array = StringArray::from(
        records
            .iter()
            .map(|r| Some(r.1.clone()))
            .collect::<Vec<_>>(),
    );
    let sentence_array = StringArray::from(documents.to_vec());

    let mut vector_builder =
        FixedSizeListBuilder::new(PrimitiveBuilder::<Float32Type>::new(), 768);
    for emb in embeddings {
        vector_builder.values().append_slice(&emb);
        vector_builder.append(true);
    }
    let vector_array = vector_builder.finish();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(kunnr_array),
            Arc::new(name_array),
            Arc::new(city_array),
            Arc::new(sentence_array),
            Arc::new(vector_array),
        ],
    )
    .map_err(|e| Box::<dyn Error>::from(e.to_string()))?;

    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

    match db.open_table("customers").execute().await {
        Ok(table) => {
            let mut builder = table.merge_insert(&["kunnr"]);
            builder.when_not_matched_insert_all();
            builder
                .execute(Box::new(batches))
                .await
                .map_err(|e| Box::<dyn Error>::from(e.to_string()))?;
        }
        Err(_) => {
            db.create_table("customers", batches)
                .execute()
                .await
                .map_err(|e| Box::<dyn Error>::from(e.to_string()))?;
        }
    }
    Ok(())
}

fn validate_sql_is_safe(query: &str) -> Result<(), Box<dyn Error>> {
    let dialect = GenericDialect {};
    let ast = Parser::parse_sql(&dialect, query)
        .map_err(|e| format!("Failed to parse SQL: {}", e))?;

    if ast.is_empty() {
        return Err("No SQL statements found.".into());
    }

    if ast.len() > 1 {
        return Err("SECURITY VIOLATION: Multiple SQL statements detected. Only a single query is allowed.".into());
    }

    match &ast[0] {
        Statement::Query(_) => Ok(()),
        _ => Err("SECURITY VIOLATION: Only pure SELECT queries are permitted in automated systems.".into()),
    }
}

pub async fn execute_sql_query(query: &str) -> Result<(), Box<dyn Error>> {
    info!("Validating SQL query safety via AST parser...");
    validate_sql_is_safe(query)?;

    info!("Spinning up Apache DataFusion Engine (Zero-Copy)...");
    let ctx = SessionContext::new();

    info!("Registering data/kna1.csv as virtual SQL table...");
    ctx.register_csv("kna1", "data/kna1.csv", CsvReadOptions::new())
        .await?;

    info!("Executing SQL: {}", query);
    let df = ctx.sql(query).await?;
    df.show().await?;

    Ok(())
}

async fn extract_chunks_from_stream(
    mut stream: lancedb::arrow::SendableRecordBatchStream,
    retrieved_chunks: &mut std::collections::HashMap<String, String>,
) -> Result<(), Box<dyn Error>> {
    use futures::StreamExt;
    
    while let Some(result) = stream.next().await {
        let batch = result.map_err(|e| Box::<dyn Error>::from(e.to_string()))?;

        let name_arr = batch.column_by_name("name").and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
        let city_arr = batch.column_by_name("city").and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
        let kunnr_arr = batch.column_by_name("kunnr").and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());

        if let (Some(names), Some(cities), Some(kunnrs)) = (name_arr, city_arr, kunnr_arr) {
            for i in 0..batch.num_rows() {
                let kunnr = kunnrs.value(i).to_string();
                let name = names.value(i).to_string();
                let city = cities.value(i).to_string();
                
                // Reconstruct the exact semantic string
                let chunk_text = format!("Customer ID {}: {} located in {}", kunnr, name, city);
                retrieved_chunks.insert(kunnr, chunk_text);
            }
        }
    }
    Ok(())
}

pub async fn execute_semantic_search(
    query: &str,
    filters: &[String],
    model: &candle_transformers::models::bert::BertModel,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<std::collections::HashMap<String, String>, Box<dyn Error>> {
    info!("Embedding search query...");
    let embeddings = get_embeddings(&[query.to_string()], &tokenizer, &model)?;
    let query_vector = embeddings.into_iter().next().ok_or("Failed to generate embedding")?;

    info!("Connecting to LanceDB...");
    let db = lancedb::connect("data/sap_vectors").execute().await.map_err(|e| Box::<dyn Error>::from(e.to_string()))?;
    let table = db.open_table("customers").execute().await.map_err(|e| Box::<dyn Error>::from(e.to_string()))?;

    let mut retrieved_chunks = std::collections::HashMap::new();

    info!("Executing semantic search (Fuzzy Pass)...");
    let vector_stream = table.query().nearest_to(query_vector).unwrap().limit(5).execute().await.map_err(|e| Box::<dyn Error>::from(e.to_string()))?;
    extract_chunks_from_stream(vector_stream, &mut retrieved_chunks).await?;

    info!("Executing exact keyword search (Deterministic Pass)...");
    for filter in filters {
        let is_not = filter.starts_with("NOT ");
        let term = if is_not { filter.strip_prefix("NOT ").unwrap() } else { filter.as_str() };

        if is_not {
            // Programmatically destroy chunks containing negative constraints
            retrieved_chunks.retain(|_, v| !v.contains(term));
        } else {
            // Deterministically retrieve exactly matching chunks
            let sql = format!("sentence LIKE '%{}%'", term);
            let exact_stream = table.query().only_if(sql).limit(5).execute().await.map_err(|e| Box::<dyn Error>::from(e.to_string()))?;
            extract_chunks_from_stream(exact_stream, &mut retrieved_chunks).await?;
        }
    }

    Ok(retrieved_chunks)
}
