#[cfg(test)]
mod integration_tests {
    use crate::db::execute_semantic_search;
    use crate::ingest::execute_ingestion;
    use crate::embeddings::load_model;
    use crate::llm::{ask_llm, build_routing_prompt};
    use crate::models::RouterDecision;
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

        // Trigger Ingestion safely into the isolated temp directory
        let result = execute_ingestion(csv_path, db_uri, false, 2, Arc::clone(&model), Arc::clone(&tokenizer)).await;
        assert!(result.is_ok(), "Ingestion pipeline panicked");

        // Verify LanceDB files exist inside the temp directory
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

        let result = execute_ingestion(csv_path, db_uri, false, 128, Arc::clone(&model), Arc::clone(&tokenizer)).await;
        assert!(result.is_ok(), "Ingestion pipeline panicked");

        let filters: Vec<String> = vec![];
        let search_result = execute_semantic_search("technology companies in Berlin", db_uri, &filters, Arc::clone(&model), Arc::clone(&tokenizer)).await;
        assert!(search_result.is_ok(), "Semantic search execution failed");
    }

    #[tokio::test]
    #[ignore = "Requires Ollama daemon running on localhost:11434"]
    async fn test_llm_router_determinism() {
        // Test 1: Exact SQL match
        let sql_prompt = build_routing_prompt("How many customers are in Berlin?");
        let sql_json = ask_llm(&sql_prompt).await.unwrap();
        let sql_decision: RouterDecision = serde_json::from_str(&sql_json).unwrap();
        assert_eq!(sql_decision.route, "SQL", "LLM Hallucinated on SQL query");

        // Test 2: Fuzzy Semantic match
        let fuzzy_prompt = build_routing_prompt("Find companies that bake bread.");
        let fuzzy_json = ask_llm(&fuzzy_prompt).await.unwrap();
        let fuzzy_decision: RouterDecision = serde_json::from_str(&fuzzy_json).unwrap();
        assert_eq!(fuzzy_decision.route, "SEMANTIC", "LLM Hallucinated on Semantic query");
    }
}
