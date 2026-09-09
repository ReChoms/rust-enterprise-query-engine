use rust_enterprise_query_engine::engines::embeddings::load_model;
use rust_enterprise_query_engine::engines::vector::{
    check_lancedb_health, execute_fallback_search, execute_semantic_search,
};
use rust_enterprise_query_engine::pipelines::ingest::execute_ingestion;
use std::io::Write;
use std::sync::Arc;
use tempfile::{tempdir, NamedTempFile};

#[tokio::test]
async fn test_dummy_data_pipeline() {
    let db_dir = tempdir().unwrap();
    let db_uri = db_dir.path().to_str().unwrap();

    let mut dummy_csv = NamedTempFile::new().unwrap();
    writeln!(dummy_csv, "kunnr,name1,ort01,land1").unwrap();
    writeln!(dummy_csv, "DUMMY001,Acme Corp,Berlin,DE").unwrap();
    writeln!(dummy_csv, "DUMMY002,Global Tech,Munich,DE").unwrap();

    let csv_path = dummy_csv.path().to_str().unwrap();
    let (model, tokenizer) = load_model().await.expect("Failed to load model");

    let result = execute_ingestion(
        csv_path,
        db_uri,
        false,
        2,
        Arc::clone(&model),
        Arc::clone(&tokenizer),
    )
    .await;
    assert!(result.is_ok(), "Ingestion pipeline panicked");

    let db_path = db_dir.path().join("customers.lance");
    assert!(db_path.exists(), "LanceDB failed to write physical files");
}

#[tokio::test]
#[ignore = "Heavy E2E Test: Ingests 20,210 rows from scratch due to test isolation"]
async fn test_real_sap_pipeline() {
    let db_dir = tempdir().unwrap();
    let db_uri = db_dir.path().to_str().unwrap();
    let csv_path = "data/kna1.csv";

    let (model, tokenizer) = load_model().await.expect("Failed to load model");

    let result = execute_ingestion(
        csv_path,
        db_uri,
        false,
        128,
        Arc::clone(&model),
        Arc::clone(&tokenizer),
    )
    .await;
    assert!(result.is_ok(), "Ingestion pipeline panicked");

    let filters: Vec<String> = vec![];
    let search_result = execute_semantic_search(
        "technology companies in Berlin",
        db_uri,
        &filters,
        Arc::clone(&model),
        Arc::clone(&tokenizer),
    )
    .await;
    assert!(search_result.is_ok(), "Semantic search execution failed");
}

#[tokio::test]
async fn test_isolated_vector_retrieval() {
    let db_dir = tempdir().unwrap();
    let db_uri = db_dir.path().to_str().unwrap();

    let mut dummy_csv = NamedTempFile::new().unwrap();
    writeln!(dummy_csv, "kunnr,name1,ort01,land1").unwrap();
    writeln!(dummy_csv, "CUST01,Alpha Tech,Berlin,DE").unwrap();
    writeln!(dummy_csv, "CUST02,Beta Tech,Munich,DE").unwrap();
    writeln!(dummy_csv, "CUST03,Gamma Bakery,Paris,FR").unwrap();

    let csv_path = dummy_csv.path().to_str().unwrap();
    let (model, tokenizer) = load_model().await.expect("Failed to load model");

    execute_ingestion(
        csv_path,
        db_uri,
        false,
        3,
        Arc::clone(&model),
        Arc::clone(&tokenizer),
    )
    .await
    .unwrap();

    let empty_filters: Vec<String> = vec![];
    let chunks_basic = execute_semantic_search(
        "Find technology companies",
        db_uri,
        &empty_filters,
        Arc::clone(&model),
        Arc::clone(&tokenizer),
    )
    .await
    .unwrap();
    assert!(
        !chunks_basic.is_empty(),
        "Failed to retrieve basic semantic chunks"
    );

    let has_tech = chunks_basic.values().any(|v| v.contains("Tech"));
    assert!(has_tech, "Did not retrieve a tech company");

    let negative_filters = vec!["NOT Munich".to_string()];
    let chunks_filtered = execute_semantic_search(
        "Find technology companies",
        db_uri,
        &negative_filters,
        Arc::clone(&model),
        Arc::clone(&tokenizer),
    )
    .await
    .unwrap();

    assert!(
        !chunks_filtered.is_empty(),
        "Filtered chunks should not be entirely empty"
    );
    for (_, chunk_text) in chunks_filtered.iter() {
        assert!(
            !chunk_text.contains("Munich"),
            "Filter exclusion failed: Munich was retrieved"
        );
    }
}

#[tokio::test]
async fn test_fallback_search() {
    let db_dir = tempdir().unwrap();
    let db_uri = db_dir.path().to_str().unwrap();

    let mut dummy_csv = NamedTempFile::new().unwrap();
    writeln!(dummy_csv, "kunnr,name1,ort01,land1").unwrap();
    writeln!(dummy_csv, "CUST99,Omega Corp,Berlin,DE").unwrap();

    let csv_path = dummy_csv.path().to_str().unwrap();
    let (model, tokenizer) = load_model().await.expect("Failed to load model");

    execute_ingestion(
        csv_path,
        db_uri,
        false,
        1,
        Arc::clone(&model),
        Arc::clone(&tokenizer),
    )
    .await
    .unwrap();

    let chunks_fallback = execute_fallback_search("Omega Corp", db_uri)
        .await
        .unwrap();
    assert!(
        !chunks_fallback.is_empty(),
        "Fallback search failed to retrieve exact match"
    );

    let retrieved_text = chunks_fallback.values().next().unwrap().to_string();
    assert!(retrieved_text.contains("Omega Corp"));

    let empty_fallback = execute_fallback_search("Delta Corp", db_uri)
        .await
        .unwrap();
    assert!(
        empty_fallback.is_empty(),
        "Fallback found data that does not exist"
    );
}

#[tokio::test]
async fn test_check_lancedb_health() {
    let db_dir = tempdir().unwrap();
    let db_uri = db_dir.path().to_str().unwrap();

    let mut dummy_csv = NamedTempFile::new().unwrap();
    writeln!(dummy_csv, "kunnr,name1,ort01,land1").unwrap();
    writeln!(dummy_csv, "CUST01,Alpha Tech,Berlin,DE").unwrap();
    writeln!(dummy_csv, "CUST02,Beta Tech,Munich,DE").unwrap();

    let csv_path = dummy_csv.path().to_str().unwrap();
    let (model, tokenizer) = load_model().await.expect("Failed to load model");
    execute_ingestion(
        csv_path,
        db_uri,
        false,
        2,
        Arc::clone(&model),
        Arc::clone(&tokenizer),
    )
    .await
    .unwrap();

    let total_records = check_lancedb_health(db_uri)
        .await
        .expect("Health check failed");
    assert_eq!(total_records, 2, "LanceDB health check reported wrong row count");
}
