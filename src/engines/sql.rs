use anyhow::Result;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::json::LineDelimitedWriter;
use datafusion::prelude::*;
use tracing::info;

use super::ast_guard::validate_sql_is_safe;

/// Initializes the Apache DataFusion execution engine with in-memory zero-copy Arrow structures.
pub async fn init_datafusion() -> Result<SessionContext> {
    info!("Spinning up Apache DataFusion Engine (Zero-Copy)...");
    let sql_engine = SessionContext::new();

    let csv_path = std::env::var("CSV_PATH").unwrap_or_else(|_| "data/kna1.csv".to_string());
    info!("Registering {} as virtual SQL table 'kna1'...", csv_path);
    sql_engine.register_csv("kna1", &csv_path, CsvReadOptions::new())
        .await?;

    Ok(sql_engine)
}

/// Executes a SQL query against DataFusion after passing it through the AST security firewall.
pub async fn execute_sql_query(sql_engine: &SessionContext, query: &str) -> Result<Vec<RecordBatch>> {
    info!("Validating SQL query safety via AST parser...");
    validate_sql_is_safe(query)?;

    info!("Executing SQL: {}", query);
    let df = sql_engine.sql(query).await?;
    let batches = df.collect().await?;

    Ok(batches)
}

/// Streams Apache Arrow RecordBatches directly into any destination implementing `std::io::Write`.
pub fn write_record_batches_as_json_lines<W: std::io::Write>(
    batches: &[RecordBatch],
    writer: &mut W,
) -> Result<()> {
    let mut json_writer = LineDelimitedWriter::new(writer);
    for batch in batches {
        json_writer.write(batch)?;
    }
    json_writer.finish()?;
    Ok(())
}

/// Convenience helper: serializes Apache Arrow RecordBatches into a vector of JSON strings.
pub fn record_batches_to_json_lines(batches: &[RecordBatch]) -> Result<Vec<String>> {
    let mut buffer = Vec::new();
    write_record_batches_as_json_lines(batches, &mut buffer)?;

    let raw_text = String::from_utf8(buffer)?;
    let lines = raw_text
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{Int32Array, StringArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn test_record_batches_to_json_lines_formatting() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("kunnr", DataType::Utf8, false),
            Field::new("count", DataType::Int32, false),
        ]));

        let kunnr_array = Arc::new(StringArray::from(vec!["1000", "2000"]));
        let count_array = Arc::new(Int32Array::from(vec![42, 99]));

        let batch = RecordBatch::try_new(schema, vec![kunnr_array, count_array]).unwrap();
        let lines = record_batches_to_json_lines(&[batch]).unwrap();

        assert_eq!(lines.len(), 2);
        let parsed_1: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(parsed_1["kunnr"], "1000");
        assert_eq!(parsed_1["count"], 42);
        let parsed_2: serde_json::Value = serde_json::from_str(&lines[1]).unwrap();
        assert_eq!(parsed_2["kunnr"], "2000");
        assert_eq!(parsed_2["count"], 99);
    }
}
