use rust_enterprise_query_engine::engines::sql::{
    execute_sql_query, init_datafusion, record_batches_to_json_lines,
};

#[tokio::test]
async fn test_sql_engine_json_lines_e2e() {
    let sql_engine = init_datafusion().await.expect("Failed to init DataFusion");
    let batches = execute_sql_query(&sql_engine, "SELECT kunnr, name1, ort01 FROM kna1 LIMIT 2")
        .await
        .expect("SQL execution failed");
    let lines = record_batches_to_json_lines(&batches).expect("JSON serialization failed");

    assert_eq!(lines.len(), 2, "Expected 2 JSON-Lines output");
    for line in lines {
        let parsed: serde_json::Value =
            serde_json::from_str(&line).expect("Output line was not valid JSON");
        assert!(parsed.get("kunnr").is_some(), "Missing kunnr field");
        assert!(parsed.get("name1").is_some(), "Missing name1 field");
        assert!(parsed.get("ort01").is_some(), "Missing ort01 field");
    }
}

#[test]
fn test_record_batches_to_json_lines_empty() {
    let lines = record_batches_to_json_lines(&[]).expect("Empty batch serialization failed");
    assert!(lines.is_empty());
}
