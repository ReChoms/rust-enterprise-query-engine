use anyhow::{anyhow, bail, Result};
use sqlparser::ast::Statement;
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

/// Validates that an incoming SQL query contains strictly one pure SELECT query.
///
/// Prevents SQL injection, destructive statements (DROP/TRUNCATE/ALTER/INSERT),
/// and transaction blocks before queries reach Apache DataFusion.
pub fn validate_sql_is_safe(query: &str) -> Result<()> {
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
