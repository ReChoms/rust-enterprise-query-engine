#[cfg(test)]
mod integration_tests {
    use crate::db::execute_semantic_search;
    use crate::ingest::execute_ingestion;
    use crate::embeddings::load_model;
    use crate::llm::{build_routing_prompt, OllamaClient};
    use crate::types::RouterDecision;
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
        let client = OllamaClient::init_from_env_or_default().unwrap();

        // Test 1: Exact SQL match
        let sql_prompt = build_routing_prompt("How many customers are in Berlin?");
        let sql_json = client.prompt_model(&sql_prompt).await.unwrap();
        let sql_decision: RouterDecision = serde_json::from_str(&sql_json).unwrap();
        assert_eq!(sql_decision.route, "SQL", "LLM Hallucinated on SQL query");

        // Test 2: Fuzzy Semantic match
        let fuzzy_prompt = build_routing_prompt("Find companies that bake bread.");
        let fuzzy_json = client.prompt_model(&fuzzy_prompt).await.unwrap();
        let fuzzy_decision: RouterDecision = serde_json::from_str(&fuzzy_json).unwrap();
        assert_eq!(fuzzy_decision.route, "SEMANTIC", "LLM Hallucinated on Semantic query");
    }

    #[test]
    fn test_intent_parsing_logic() {
        use crate::llm::{build_question_parser_prompt, parse_llm_json};
        use crate::types::ParsedQuestion;

        // Test 1: Basic Intent Parsing
        let basic_prompt = build_question_parser_prompt("Find companies in Berlin");
        assert!(basic_prompt.contains("Find companies in Berlin"));
        let basic_json = r#"{"intent": "Find companies in Berlin", "filters": []}"#;
        let parsed_basic: ParsedQuestion = parse_llm_json(basic_json).unwrap();
        assert_eq!(parsed_basic.intent, "Find companies in Berlin");
        assert!(parsed_basic.filters.is_empty());

        // Test 2: Explicit Negative Filters
        let neg_prompt = build_question_parser_prompt("Find tech companies but NOT in Berlin");
        assert!(neg_prompt.contains("NOT in Berlin"));
        let neg_json = r#"{"intent": "Find tech companies", "filters": ["NOT in Berlin"]}"#;
        let parsed_neg: ParsedQuestion = parse_llm_json(neg_json).unwrap();
        assert_eq!(parsed_neg.intent, "Find tech companies");
        assert_eq!(parsed_neg.filters, vec!["NOT in Berlin"]);

        // Test 3: Complex Boolean Filters
        let complex_json = r#"{"intent": "manufacturing companies in Germany", "filters": ["NOT in Munich", "NOT in Berlin"]}"#;
        let parsed_complex: ParsedQuestion = parse_llm_json(complex_json).unwrap();
        assert_eq!(parsed_complex.intent, "manufacturing companies in Germany");
        assert_eq!(parsed_complex.filters, vec!["NOT in Munich", "NOT in Berlin"]);
    }

    #[tokio::test]
    async fn test_isolated_vector_retrieval() {
        use crate::db::execute_semantic_search;
        use crate::ingest::execute_ingestion;
        use crate::embeddings::load_model;
        use std::io::Write;
        use std::sync::Arc;
        use tempfile::{tempdir, NamedTempFile};

        let db_dir = tempdir().unwrap();
        let db_uri = db_dir.path().to_str().unwrap();

        let mut dummy_csv = NamedTempFile::new().unwrap();
        writeln!(dummy_csv, "kunnr,name1,ort01,land1").unwrap();
        writeln!(dummy_csv, "CUST01,Alpha Tech,Berlin,DE").unwrap();
        writeln!(dummy_csv, "CUST02,Beta Tech,Munich,DE").unwrap();
        writeln!(dummy_csv, "CUST03,Gamma Bakery,Paris,FR").unwrap();
        
        let csv_path = dummy_csv.path().to_str().unwrap();
        let (model, tokenizer) = load_model().await.expect("Failed to load model");

        execute_ingestion(csv_path, db_uri, false, 3, Arc::clone(&model), Arc::clone(&tokenizer)).await.unwrap();

        // Test 4: Basic Semantic Retrieval
        let empty_filters: Vec<String> = vec![];
        let chunks_basic = execute_semantic_search("Find technology companies", db_uri, &empty_filters, Arc::clone(&model), Arc::clone(&tokenizer)).await.unwrap();
        assert!(!chunks_basic.is_empty(), "Failed to retrieve basic semantic chunks");
        
        let has_tech = chunks_basic.values().any(|v| v.contains("Tech"));
        assert!(has_tech, "Did not retrieve a tech company");

        // Test 5: Semantic Retrieval with Hard Filter Exclusion
        let negative_filters = vec!["NOT Munich".to_string()];
        let chunks_filtered = execute_semantic_search("Find technology companies", db_uri, &negative_filters, Arc::clone(&model), Arc::clone(&tokenizer)).await.unwrap();
        
        assert!(!chunks_filtered.is_empty(), "Filtered chunks should not be entirely empty");
        for (_, chunk_text) in chunks_filtered.iter() {
            assert!(!chunk_text.contains("Munich"), "Filter exclusion failed: Munich was retrieved");
        }
    }

    #[test]
    fn test_llm_generation_and_verification() {
        use crate::llm::{build_semantic_prompt, verify_and_parse_llm_generation};
        use std::collections::HashMap;

        // Mock retrieved chunks from the vector database
        let mut chunks = HashMap::new();
        chunks.insert("CUST_88".to_string(), std::sync::Arc::from("Gamma Bakery in Paris"));

        // Test 6: LLM Deterministic Generation (Answer Present)
        let sem_prompt = build_semantic_prompt("Where is Gamma Bakery?", &chunks);
        assert!(sem_prompt.contains("Gamma Bakery in Paris"));
        
        let valid_json = r#"{
            "answer_found": true,
            "answer": "Gamma Bakery is located in Paris.",
            "exact_quote": "Gamma Bakery in Paris",
            "source_chunk_id": "CUST_88"
        }"#;
        
        let payload = verify_and_parse_llm_generation(valid_json, &chunks).unwrap();
        assert!(payload.answer_found);

        // Test 7: LLM Hallucination Rejection (Answer Missing but LLM fakes it)
        let hallucinated_json = r#"{
            "answer_found": true,
            "answer": "Gamma Bakery is located in London.",
            "exact_quote": "Gamma Bakery in London",
            "source_chunk_id": "CUST_88"
        }"#;
        
        let rejected = verify_and_parse_llm_generation(hallucinated_json, &chunks);
        assert!(rejected.is_err());
        assert!(rejected.unwrap_err().to_string().contains("Hallucination detected"));
    }

    #[tokio::test]
    async fn test_fallback_search() {
        use crate::db::execute_fallback_search;
        use crate::ingest::execute_ingestion;
        use crate::embeddings::load_model;
        use std::io::Write;
        use std::sync::Arc;
        use tempfile::{tempdir, NamedTempFile};

        let db_dir = tempdir().unwrap();
        let db_uri = db_dir.path().to_str().unwrap();

        let mut dummy_csv = NamedTempFile::new().unwrap();
        writeln!(dummy_csv, "kunnr,name1,ort01,land1").unwrap();
        writeln!(dummy_csv, "CUST99,Omega Corp,Berlin,DE").unwrap();
        
        let csv_path = dummy_csv.path().to_str().unwrap();
        let (model, tokenizer) = load_model().await.expect("Failed to load model");

        execute_ingestion(csv_path, db_uri, false, 1, Arc::clone(&model), Arc::clone(&tokenizer)).await.unwrap();

        // Test 8: Fallback Search (Absence Proof)
        let chunks_fallback = execute_fallback_search("Omega Corp", db_uri).await.unwrap();
        assert!(!chunks_fallback.is_empty(), "Fallback search failed to retrieve exact match");
        
        let retrieved_text = chunks_fallback.values().next().unwrap().to_string();
        assert!(retrieved_text.contains("Omega Corp"));
        
        let empty_fallback = execute_fallback_search("Delta Corp", db_uri).await.unwrap();
        assert!(empty_fallback.is_empty(), "Fallback found data that does not exist");
    }

    #[test]
    fn test_mocked_e2e_ask_semantic() {
        use crate::llm::{parse_llm_json, verify_and_parse_llm_generation};
        use crate::types::ParsedQuestion;
        use std::collections::HashMap;

        // Test 9: Simulated E2E Flow with Mocked LLM and Vector DB
        // 1. LLM parses the user question
        let mock_intent_json = r#"{"intent": "Find Gamma Bakery", "filters": []}"#;
        let parsed: ParsedQuestion = parse_llm_json(mock_intent_json).unwrap();
        assert_eq!(parsed.intent, "Find Gamma Bakery");

        // 2. Vector DB retrieves the chunk
        let mut mock_chunks = HashMap::new();
        mock_chunks.insert("CUST_88".to_string(), std::sync::Arc::from("Gamma Bakery in Paris"));

        // 3. LLM verifies the chunk contains the answer
        let mock_answer_json = r#"{
            "answer_found": true,
            "answer": "Gamma Bakery is in Paris.",
            "exact_quote": "Gamma Bakery in Paris",
            "source_chunk_id": "CUST_88"
        }"#;
        
        let final_payload = verify_and_parse_llm_generation(mock_answer_json, &mock_chunks).unwrap();
        assert!(final_payload.answer_found);
        assert_eq!(final_payload.answer, "Gamma Bakery is in Paris.");
    }

    #[tokio::test]
    #[ignore = "Heavy E2E Test: Runs full AskSemantic pipeline requiring Live Ollama and BAAI models"]
    async fn test_heavy_e2e_ask_semantic_pipeline() {
        use crate::llm::{build_question_parser_prompt, build_semantic_prompt, parse_llm_json, verify_and_parse_llm_generation, OllamaClient};
        use crate::types::ParsedQuestion;
        use crate::db::execute_semantic_search;
        use crate::ingest::execute_ingestion;
        use crate::embeddings::load_model;
        use std::io::Write;
        use std::sync::Arc;
        use tempfile::{tempdir, NamedTempFile};

        // Setup Isolated Physical DB
        let db_dir = tempdir().unwrap();
        let db_uri = db_dir.path().to_str().unwrap();

        let mut dummy_csv = NamedTempFile::new().unwrap();
        writeln!(dummy_csv, "kunnr,name1,ort01,land1").unwrap();
        writeln!(dummy_csv, "CUST01,Alpha Tech,Berlin,DE").unwrap();
        
        let csv_path = dummy_csv.path().to_str().unwrap();
        let (model, tokenizer) = load_model().await.expect("Failed to load model");
        execute_ingestion(csv_path, db_uri, false, 1, Arc::clone(&model), Arc::clone(&tokenizer)).await.unwrap();

        // --- Execute Real AskSemantic Flow ---
        let query = "Who is the technology company in Berlin?";
        let client = OllamaClient::init_from_env_or_default().unwrap();

        // 1. LLM Parses Intent (Live Ollama call)
        let parser_prompt = build_question_parser_prompt(query);
        let raw_parser_output = client.prompt_model(&parser_prompt).await.unwrap();
        let parsed_query: ParsedQuestion = parse_llm_json(&raw_parser_output).unwrap();
        
        // 2. Vector DB Searches (Live Embeddings + LanceDB)
        let chunks = execute_semantic_search(&parsed_query.intent, db_uri, &parsed_query.filters, Arc::clone(&model), Arc::clone(&tokenizer)).await.unwrap();
        assert!(!chunks.is_empty(), "E2E Search returned no chunks");

        // 3. LLM Generates Final Output (Live Ollama call)
        let semantic_prompt = build_semantic_prompt(query, &chunks);
        let raw_llm_output = client.prompt_model(&semantic_prompt).await.unwrap();
        let final_payload = verify_and_parse_llm_generation(&raw_llm_output, &chunks).unwrap();

        assert!(final_payload.answer_found, "Live LLM failed to generate answer from physical DB");
        assert!(final_payload.answer.contains("Alpha Tech"), "LLM generated wrong answer");
    }

    #[tokio::test]
    async fn test_check_lancedb_health() {
        use crate::db::check_lancedb_health;
        use crate::ingest::execute_ingestion;
        use crate::embeddings::load_model;
        use std::io::Write;
        use std::sync::Arc;
        use tempfile::{tempdir, NamedTempFile};

        let db_dir = tempdir().unwrap();
        let db_uri = db_dir.path().to_str().unwrap();

        let mut dummy_csv = NamedTempFile::new().unwrap();
        writeln!(dummy_csv, "kunnr,name1,ort01,land1").unwrap();
        writeln!(dummy_csv, "CUST01,Alpha Tech,Berlin,DE").unwrap();
        writeln!(dummy_csv, "CUST02,Beta Tech,Munich,DE").unwrap();

        let csv_path = dummy_csv.path().to_str().unwrap();
        let (model, tokenizer) = load_model().await.expect("Failed to load model");
        execute_ingestion(csv_path, db_uri, false, 2, Arc::clone(&model), Arc::clone(&tokenizer)).await.unwrap();

        let total_records = check_lancedb_health(db_uri).await.expect("Health check failed");
        assert_eq!(total_records, 2, "LanceDB health check reported wrong row count");
    }

    #[tokio::test]
    async fn test_ollama_client_offline_probe_does_not_panic() {
        use crate::llm::OllamaClient;
        let client = OllamaClient::init_from_env_or_default().expect("Failed to init client");
        // Fast probe should return false (or true if Ollama happens to be live) without hanging or panicking
        let _online = client.is_healthy().await;
    }

    #[test]
    fn test_degraded_response_serialization() {
        use crate::types::{DegradedChunk, DegradedResponse};

        let degraded = DegradedResponse {
            degraded: true,
            message: "LLM offline. Degraded results.".to_string(),
            retrieved_chunks: vec![DegradedChunk {
                chunk_id: "chunk_1".to_string(),
                content: "Customer Acme in Berlin".to_string(),
            }],
        };

        let json = serde_json::to_string(&degraded).unwrap();
        assert!(json.contains("\"degraded\":true"));
        assert!(json.contains("Customer Acme in Berlin"));
    }
}
