use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::json::LineDelimitedWriter;
use datafusion::prelude::*;
use sqlparser::ast::Statement;
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use anyhow::{anyhow, bail, Result};
use tracing::info;

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

pub async fn execute_sql_query(sql_engine: &SessionContext, query: &str) -> Result<Vec<RecordBatch>> {
    info!("Validating SQL query safety via AST parser...");
    validate_sql_is_safe(query)?;

    info!("Executing SQL: {}", query);
    let df = sql_engine.sql(query).await?;
    let batches = df.collect().await?;

    Ok(batches)
}

/// Streams Apache Arrow RecordBatches directly into any destination implementing `std::io::Write` (e.g. STDOUT, socket, buffer)
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

/// Convenience helper: serializes Apache Arrow RecordBatches into a vector of JSON strings
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

    #[test]
    fn test_record_batches_to_json_lines_formatting() {
        use datafusion::arrow::array::{Int32Array, StringArray};
        use datafusion::arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

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
