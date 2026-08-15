use arrow_array::builder::{FixedSizeListBuilder, PrimitiveBuilder};
use arrow_array::types::Float32Type;
use arrow_array::{Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::Schema;
use datafusion::prelude::*;
use lancedb::query::{ExecutableQuery, QueryBase};
use sqlparser::ast::Statement;
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use anyhow::{anyhow, bail, Result};
use std::sync::Arc;
use tracing::info;

use crate::embeddings::get_embeddings;

pub async fn insert_batch(
    vector_db: &lancedb::Connection,
    schema: Arc<Schema>,
    records: &[(String, String, String)],
    documents: &[String],
    embeddings: Vec<Vec<f32>>,
) -> Result<()> {
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

fn validate_sql_is_safe(query: &str) -> Result<()> {
    let dialect = GenericDialect {};
    let ast = Parser::parse_sql(&dialect, query)
        .map_err(|e| anyhow!("Failed to parse SQL: {}", e))?;

    if ast.is_empty() {
        bail!("No SQL statements found.");
    }

    if ast.len() > 1 {
        bail!("SECURITY VIOLATION: Multiple SQL statements detected. Only a single query is allowed.");
    }

    match &ast[0] {
        Statement::Query(_) => Ok(()),
        _ => bail!("SECURITY VIOLATION: Only pure SELECT queries are permitted in automated systems."),
    }
}

pub async fn init_datafusion() -> Result<SessionContext> {
    info!("Spinning up Apache DataFusion Engine (Zero-Copy)...");
    let sql_engine = SessionContext::new();

    info!("Registering data/kna1.csv as virtual SQL table...");
    sql_engine.register_csv("kna1", "data/kna1.csv", CsvReadOptions::new())
        .await?;

    Ok(sql_engine)
}

pub async fn execute_sql_query(sql_engine: &SessionContext, query: &str) -> Result<()> {
    info!("Validating SQL query safety via AST parser...");
    validate_sql_is_safe(query)?;

    info!("Executing SQL: {}", query);
    let df = sql_engine.sql(query).await?;
    df.show().await?;

    Ok(())
}

async fn extract_chunks_from_stream(
    mut stream: lancedb::arrow::SendableRecordBatchStream,
    retrieved_chunks: &mut std::collections::HashMap<String, std::sync::Arc<str>>,
) -> Result<()> {
    use futures::StreamExt;
    
    while let Some(result) = stream.next().await {
        let batch = result?;

        let name_arr = batch.column_by_name("name").and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
        let city_arr = batch.column_by_name("city").and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
        let kunnr_arr = batch.column_by_name("kunnr").and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());

        if let (Some(names), Some(cities), Some(kunnrs)) = (name_arr, city_arr, kunnr_arr) {
            for i in 0..batch.num_rows() {
                let kunnr = kunnrs.value(i).to_string();
                let name = names.value(i).to_string();
                let city = cities.value(i).to_string();
                
                // Maintain structured relational boundaries instead of flattening
                let chunk_json = serde_json::json!({
                    "kunnr": kunnr,
                    "name": name,
                    "city": city
                }).to_string();
                retrieved_chunks.insert(kunnr, std::sync::Arc::from(chunk_json));
            }
        }
    }
    Ok(())
}

pub async fn execute_semantic_search(
    query: &str,
    db_uri: &str,
    filters: &[String],
    model: Arc<candle_transformers::models::bert::BertModel>,
    tokenizer: Arc<tokenizers::Tokenizer>,
) -> Result<std::collections::HashMap<String, std::sync::Arc<str>>> {
    info!("Embedding search query...");
    let embeddings = get_embeddings(vec![query.to_string()], tokenizer, model).await?;
    let query_vector = embeddings.into_iter().next().ok_or_else(|| anyhow!("Failed to generate embedding"))?;

    info!("Connecting to LanceDB...");
    let vector_db = lancedb::connect(db_uri).execute().await?;
    let target_table = vector_db.open_table("customers").execute().await?;

    let mut retrieved_chunks = std::collections::HashMap::new();

    info!("Executing semantic search (Fuzzy Pass)...");
    let vector_stream = target_table.query().nearest_to(query_vector).unwrap().limit(5).execute().await?;
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
            let exact_stream = target_table.query().only_if(sql).limit(5).execute().await?;
            extract_chunks_from_stream(exact_stream, &mut retrieved_chunks).await?;
        }
    }

    Ok(retrieved_chunks)
}

pub fn build_fallback_sql(intent: &str) -> String {
    let mut conditions = Vec::new();
    
    for word in intent.split_whitespace() {
        // Standard SQL escaping: double up single quotes to prevent string literal breakout
        let escaped_word = word.replace("'", "''");
        conditions.push(format!("sentence LIKE '%{}%'", escaped_word));
    }
    
    if conditions.is_empty() {
        return "1=0".to_string(); // Fail-safe for empty inputs prevents syntax errors
    }
    
    conditions.join(" AND ")
}

pub async fn execute_fallback_search(
    intent: &str,
    db_uri: &str,
) -> Result<std::collections::HashMap<String, std::sync::Arc<str>>> {
    info!("Vector search failed. Executing deterministic fallback search (Absence Proof) for: {}", intent);
    
    let vector_db = lancedb::connect(db_uri).execute().await?;
    let target_table = vector_db.open_table("customers").execute().await?;

    let mut retrieved_chunks = std::collections::HashMap::new();
    
    let sql = build_fallback_sql(intent);
    
    let exact_stream = target_table.query()
        .only_if(sql)
        .limit(5)
        .execute()
        .await?;
        
    extract_chunks_from_stream(exact_stream, &mut retrieved_chunks).await?;

    Ok(retrieved_chunks)
}

/// Probes LanceDB connectivity and returns the total count of indexed vectors
pub async fn check_lancedb_health(db_uri: &str) -> Result<usize> {
    let vector_db = lancedb::connect(db_uri).execute().await?;
    let target_table = vector_db.open_table("customers").execute().await?;
    let total_rows = target_table.count_rows(None).await?;
    Ok(total_rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_select_passes() {
        assert!(validate_sql_is_safe("SELECT * FROM kna1 WHERE ort01 = 'Berlin'").is_ok());
    }

    #[test]
    fn test_drop_table_blocked() {
        let result = validate_sql_is_safe("DROP TABLE kna1");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("SECURITY VIOLATION"));
    }

    #[test]
    fn test_multi_statement_blocked() {
        let result = validate_sql_is_safe("SELECT 1; DROP TABLE kna1;");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Multiple SQL statements detected"));
    }

    #[test]
    fn test_insert_blocked() {
        let result = validate_sql_is_safe("INSERT INTO kna1 VALUES ('1000')");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("pure SELECT queries are permitted"));
    }

    #[test]
    fn test_fallback_sql_sanitization() {
        let intent = "find customer 'O'Connor'";
        let sql = build_fallback_sql(intent);
        // Should safely escape single quotes by doubling them, preserving data integrity
        assert_eq!(sql, "sentence LIKE '%find%' AND sentence LIKE '%customer%' AND sentence LIKE '%''O''Connor''%'");
    }

    // --- GOLDEN TESTS: ADVANCED ANALYTICS (MUST PASS) ---

    #[test]
    fn test_golden_heavy_aggregation() {
        assert!(validate_sql_is_safe("SELECT land1, sum(length(name1)), count(*) FROM kna1 GROUP BY land1 HAVING count(*) > 5 ORDER BY land1 DESC LIMIT 5 OFFSET 10").is_ok());
    }

    #[test]
    fn test_golden_cte_with_clause() {
        assert!(validate_sql_is_safe("WITH regional_counts AS (SELECT ort01, count(*) as c FROM kna1 GROUP BY ort01) SELECT * FROM regional_counts WHERE c > 10").is_ok());
    }

    #[test]
    fn test_golden_window_function() {
        assert!(validate_sql_is_safe("SELECT kunnr, name1, row_number() OVER (PARTITION BY land1 ORDER BY kunnr) as rank FROM kna1").is_ok());
    }

    #[test]
    fn test_golden_complex_self_join() {
        assert!(validate_sql_is_safe("SELECT a.kunnr, b.name1 FROM kna1 a INNER JOIN kna1 b ON a.ort01 = b.ort01 WHERE a.kunnr != b.kunnr").is_ok());
    }

    #[test]
    fn test_golden_casting_and_coalesce() {
        assert!(validate_sql_is_safe("SELECT coalesce(kunnr, 'UNKNOWN'), cast(length(name1) as INT) FROM kna1 WHERE ort01 IS NOT NULL").is_ok());
    }

    // --- GOLDEN TESTS: ADVERSARIAL ATTACKS (MUST FAIL) ---

    #[test]
    fn test_golden_attack_truncate() {
        let result = validate_sql_is_safe("TRUNCATE TABLE kna1;");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("SECURITY VIOLATION"));
    }

    #[test]
    fn test_golden_attack_alter_schema() {
        let result = validate_sql_is_safe("ALTER TABLE kna1 ADD COLUMN secret VARCHAR(255);");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("SECURITY VIOLATION"));
    }

    #[test]
    fn test_golden_attack_exfiltration_copy() {
        let result = validate_sql_is_safe("COPY kna1 TO '/tmp/kna1_stolen.csv';");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("SECURITY VIOLATION"));
    }

    #[test]
    fn test_golden_attack_transaction_block() {
        let result = validate_sql_is_safe("BEGIN TRANSACTION; DROP TABLE kna1; COMMIT;");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Multiple SQL statements"));
    }

    #[test]
    fn test_golden_attack_comment_obfuscated_injection() {
        let result = validate_sql_is_safe("SELECT * FROM kna1; /* bypass filter */ DROP TABLE kna1;");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Multiple SQL statements"));
    }
}
