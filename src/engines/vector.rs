use anyhow::{anyhow, Result};
use arrow_array::builder::{FixedSizeListBuilder, PrimitiveBuilder};
use arrow_array::types::Float32Type;
use arrow_array::{Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::Schema;
use futures::StreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;

use super::embeddings::get_embeddings;

pub async fn insert_batch(
    vector_db: &lancedb::Connection,
    schema: Arc<Schema>,
    records: &[(String, String, String)],
    documents: &[String],
    embeddings: Vec<Vec<f32>>,
) -> Result<()> {
    let kunnr_array = StringArray::from(
        records.iter().map(|r| Some(r.2.clone())).collect::<Vec<_>>(),
    );
    let name_array = StringArray::from(
        records.iter().map(|r| Some(r.0.clone())).collect::<Vec<_>>(),
    );
    let city_array = StringArray::from(
        records.iter().map(|r| Some(r.1.clone())).collect::<Vec<_>>(),
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
    )?;

    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

    match vector_db.open_table("customers").execute().await {
        Ok(target_table) => {
            let mut builder = target_table.merge_insert(&["kunnr"]);
            builder.when_not_matched_insert_all();
            builder.execute(Box::new(batches)).await?;
        }
        Err(_) => {
            vector_db.create_table("customers", batches).execute().await?;
        }
    }
    Ok(())
}

async fn extract_chunks_from_stream(
    mut stream: lancedb::arrow::SendableRecordBatchStream,
    retrieved_chunks: &mut HashMap<String, Arc<str>>,
) -> Result<Vec<String>> {
    let mut ids = Vec::new();
    while let Some(result) = stream.next().await {
        let batch = result?;

        let name_arr = batch.column_by_name("name").and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let city_arr = batch.column_by_name("city").and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let kunnr_arr = batch.column_by_name("kunnr").and_then(|c| c.as_any().downcast_ref::<StringArray>());

        if let (Some(names), Some(cities), Some(kunnrs)) = (name_arr, city_arr, kunnr_arr) {
            for i in 0..batch.num_rows() {
                let kunnr = kunnrs.value(i).to_string();
                let name = names.value(i).to_string();
                let city = cities.value(i).to_string();

                let chunk_json = serde_json::json!({
                    "kunnr": kunnr,
                    "name": name,
                    "city": city
                }).to_string();
                ids.push(kunnr.clone());
                retrieved_chunks.insert(kunnr, Arc::from(chunk_json));
            }
        }
    }
    Ok(ids)
}

pub async fn execute_semantic_search(
    query: &str,
    db_uri: &str,
    filters: &[String],
    model: Arc<candle_transformers::models::bert::BertModel>,
    tokenizer: Arc<tokenizers::Tokenizer>,
) -> Result<HashMap<String, Arc<str>>> {
    info!("Embedding search query...");
    let embeddings = get_embeddings(vec![query.to_string()], tokenizer, model).await?;
    let query_vector = embeddings.into_iter().next().ok_or_else(|| anyhow!("Failed to generate embedding"))?;
    execute_semantic_search_vector(query_vector, db_uri, filters).await
}

#[tracing::instrument(name = "engine_lancedb_hybrid_search", skip(query_vector))]
pub async fn execute_hybrid_search(
    query_vector: Vec<f32>,
    db_uri: &str,
    filters: &[String],
) -> Result<Vec<(String, Arc<str>)>> {
    info!("Connecting to LanceDB...");
    let vector_db = lancedb::connect(db_uri).execute().await?;
    let target_table = vector_db.open_table("customers").execute().await?;

    let mut retrieved_chunks = HashMap::new();

    info!("Executing semantic search (Fuzzy Pass)...");
    let vector_stream = target_table.query().nearest_to(query_vector)?.limit(5).execute().await?;
    let dense_ids = extract_chunks_from_stream(vector_stream, &mut retrieved_chunks).await?;

    info!("Executing exact keyword search (Deterministic Pass)...");
    let mut lexical_ids = Vec::new();
    for filter in filters {
        let is_not = filter.starts_with("NOT ");
        let term = if is_not { filter.strip_prefix("NOT ").unwrap_or(filter.as_str()) } else { filter.as_str() };

        if is_not {
            retrieved_chunks.retain(|_, v| !v.contains(term));
        } else {
            let sql = format!("sentence LIKE '%{}%'", term);
            let exact_stream = target_table.query().only_if(sql).limit(5).execute().await?;
            let mut ids = extract_chunks_from_stream(exact_stream, &mut retrieved_chunks).await?;
            lexical_ids.append(&mut ids);
        }
    }

    let dense_refs: Vec<&str> = dense_ids.iter().map(|s| s.as_str()).collect();
    let lexical_refs: Vec<&str> = lexical_ids.iter().map(|s| s.as_str()).collect();
    let fused = crate::engines::rank_fusion::fuse_ranked_candidates(&dense_refs, &lexical_refs, 60.0);
    let mut ordered = Vec::new();
    for (id, _) in fused.into_iter().take(5) {
        if let Some(chunk) = retrieved_chunks.remove(&id) {
            ordered.push((id, chunk));
        }
    }
    Ok(ordered)
}

pub async fn execute_semantic_search_vector(
    query_vector: Vec<f32>,
    db_uri: &str,
    filters: &[String],
) -> Result<HashMap<String, Arc<str>>> {
    let ordered = execute_hybrid_search(query_vector, db_uri, filters).await?;
    Ok(ordered.into_iter().collect())
}

pub fn build_fallback_sql(intent: &str) -> String {
    let mut conditions = Vec::new();
    for word in intent.split_whitespace() {
        let escaped_word = word.replace("'", "''");
        conditions.push(format!("sentence LIKE '%{}%'", escaped_word));
    }
    if conditions.is_empty() {
        return "1=0".to_string();
    }
    conditions.join(" AND ")
}

pub async fn execute_fallback_search(
    intent: &str,
    db_uri: &str,
) -> Result<HashMap<String, Arc<str>>> {
    info!("Vector search failed. Executing deterministic fallback search (Absence Proof) for: {}", intent);
    let vector_db = lancedb::connect(db_uri).execute().await?;
    let target_table = vector_db.open_table("customers").execute().await?;

    let mut retrieved_chunks = HashMap::new();
    let sql = build_fallback_sql(intent);
    let exact_stream = target_table.query().only_if(sql).limit(5).execute().await?;
    extract_chunks_from_stream(exact_stream, &mut retrieved_chunks).await?;

    Ok(retrieved_chunks)
}

pub async fn check_lancedb_health(db_uri: &str) -> Result<usize> {
    let vector_db = lancedb::connect(db_uri).execute().await?;
    let target_table = vector_db.open_table("customers").execute().await?;
    let total_rows = target_table.count_rows(None).await?;
    Ok(total_rows)
}

pub async fn is_vector_indexed(db_uri: &str) -> Result<bool> {
    let vector_db = lancedb::connect(db_uri).execute().await?;
    let target_table = vector_db.open_table("customers").execute().await?;
    let indices = target_table.list_indices().await?;
    Ok(indices.iter().any(|idx| idx.columns.iter().any(|c| c == "vector")))
}

pub async fn create_scalar_index(table: &lancedb::Table, column: &str) -> Result<()> {
    table
        .create_index(&[column], lancedb::index::Index::BTree(lancedb::index::scalar::BTreeIndexBuilder::default()))
        .execute()
        .await?;
    Ok(())
}

pub async fn create_vector_index(table: &lancedb::Table, column: &str) -> Result<()> {
    let rows = table.count_rows(None).await?;
    if rows < 256 {
        info!("Skipping vector index on '{}': table has {} rows (IVF-PQ requires >= 256)", column, rows);
        return Ok(());
    }
    table
        .create_index(&[column], lancedb::index::Index::Auto)
        .execute()
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_sql_sanitization() {
        let intent = "find customer 'O'Connor'";
        let sql = build_fallback_sql(intent);
        assert_eq!(sql, "sentence LIKE '%find%' AND sentence LIKE '%customer%' AND sentence LIKE '%''O''Connor''%'");
    }
}
