use rust_enterprise_query_engine::common::types::{
    DegradedChunk, DegradedResponse, ParsedQuestion, RouterDecision,
};
use rust_enterprise_query_engine::engines::embeddings::load_model;
use rust_enterprise_query_engine::engines::vector::execute_semantic_search;
use rust_enterprise_query_engine::pipelines::ingest::execute_ingestion;
use rust_enterprise_query_engine::pipelines::query::{
    build_question_parser_prompt, build_routing_prompt, build_semantic_prompt, parse_llm_json,
    verify_and_parse_llm_generation, OllamaClient,
};
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use tempfile::{tempdir, NamedTempFile};

#[tokio::test]
#[ignore = "Requires Ollama daemon running on localhost:11434"]
async fn test_llm_router_determinism() {
    let client = OllamaClient::init_from_env_or_default().unwrap();

    let sql_prompt = build_routing_prompt("How many customers are in Berlin?");
    let sql_json = client.prompt_model(&sql_prompt).await.unwrap();
    let sql_decision: RouterDecision = serde_json::from_str(&sql_json).unwrap();
    assert_eq!(sql_decision.route, "SQL", "LLM Hallucinated on SQL query");

    let fuzzy_prompt = build_routing_prompt("Find companies that bake bread.");
    let fuzzy_json = client.prompt_model(&fuzzy_prompt).await.unwrap();
    let fuzzy_decision: RouterDecision = serde_json::from_str(&fuzzy_json).unwrap();
    assert_eq!(
        fuzzy_decision.route, "SEMANTIC",
        "LLM Hallucinated on Semantic query"
    );
}

#[test]
fn test_intent_parsing_logic() {
    let basic_prompt = build_question_parser_prompt("Find companies in Berlin");
    assert!(basic_prompt.contains("Find companies in Berlin"));
    let basic_json = r#"{"intent": "Find companies in Berlin", "filters": []}"#;
    let parsed_basic: ParsedQuestion = parse_llm_json(basic_json).unwrap();
    assert_eq!(parsed_basic.intent, "Find companies in Berlin");
    assert!(parsed_basic.filters.is_empty());

    let neg_prompt = build_question_parser_prompt("Find tech companies but NOT in Berlin");
    assert!(neg_prompt.contains("NOT in Berlin"));
    let neg_json = r#"{"intent": "Find tech companies", "filters": ["NOT in Berlin"]}"#;
    let parsed_neg: ParsedQuestion = parse_llm_json(neg_json).unwrap();
    assert_eq!(parsed_neg.intent, "Find tech companies");
    assert_eq!(parsed_neg.filters, vec!["NOT in Berlin"]);

    let complex_json = r#"{"intent": "manufacturing companies in Germany", "filters": ["NOT in Munich", "NOT in Berlin"]}"#;
    let parsed_complex: ParsedQuestion = parse_llm_json(complex_json).unwrap();
    assert_eq!(parsed_complex.intent, "manufacturing companies in Germany");
    assert_eq!(parsed_complex.filters, vec!["NOT in Munich", "NOT in Berlin"]);
}

#[test]
fn test_llm_generation_and_verification() {
    let mut chunks = HashMap::new();
    chunks.insert("CUST_88".to_string(), Arc::from("Gamma Bakery in Paris"));

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

    let hallucinated_json = r#"{
        "answer_found": true,
        "answer": "Gamma Bakery is located in London.",
        "exact_quote": "Gamma Bakery in London",
        "source_chunk_id": "CUST_88"
    }"#;

    let rejected = verify_and_parse_llm_generation(hallucinated_json, &chunks);
    assert!(rejected.is_err());
    assert!(rejected
        .unwrap_err()
        .to_string()
        .contains("Hallucination detected"));
}

#[test]
fn test_mocked_e2e_ask_semantic() {
    let mock_intent_json = r#"{"intent": "Find Gamma Bakery", "filters": []}"#;
    let parsed: ParsedQuestion = parse_llm_json(mock_intent_json).unwrap();
    assert_eq!(parsed.intent, "Find Gamma Bakery");

    let mut mock_chunks = HashMap::new();
    mock_chunks.insert("CUST_88".to_string(), Arc::from("Gamma Bakery in Paris"));

    let mock_answer_json = r#"{
        "answer_found": true,
        "answer": "Gamma Bakery is in Paris.",
        "exact_quote": "Gamma Bakery in Paris",
        "source_chunk_id": "CUST_88"
    }"#;

    let final_payload =
        verify_and_parse_llm_generation(mock_answer_json, &mock_chunks).unwrap();
    assert!(final_payload.answer_found);
    assert_eq!(final_payload.answer, "Gamma Bakery is in Paris.");
}

#[tokio::test]
#[ignore = "Heavy E2E Test: Runs full AskSemantic pipeline requiring Live Ollama and BAAI models"]
async fn test_heavy_e2e_ask_semantic_pipeline() {
    let db_dir = tempdir().unwrap();
    let db_uri = db_dir.path().to_str().unwrap();

    let mut dummy_csv = NamedTempFile::new().unwrap();
    writeln!(dummy_csv, "kunnr,name1,ort01,land1").unwrap();
    writeln!(dummy_csv, "CUST01,Alpha Tech,Berlin,DE").unwrap();

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

    let query = "Who is the technology company in Berlin?";
    let client = OllamaClient::init_from_env_or_default().unwrap();

    let parser_prompt = build_question_parser_prompt(query);
    let raw_parser_output = client.prompt_model(&parser_prompt).await.unwrap();
    let parsed_query: ParsedQuestion = parse_llm_json(&raw_parser_output).unwrap();

    let chunks = execute_semantic_search(
        &parsed_query.intent,
        db_uri,
        &parsed_query.filters,
        Arc::clone(&model),
        Arc::clone(&tokenizer),
    )
    .await
    .unwrap();
    assert!(!chunks.is_empty(), "E2E Search returned no chunks");

    let semantic_prompt = build_semantic_prompt(query, &chunks);
    let raw_llm_output = client.prompt_model(&semantic_prompt).await.unwrap();
    let final_payload = verify_and_parse_llm_generation(&raw_llm_output, &chunks).unwrap();

    assert!(
        final_payload.answer_found,
        "Live LLM failed to generate answer from physical DB"
    );
    assert!(
        final_payload.answer.contains("Alpha Tech"),
        "LLM generated wrong answer"
    );
}

#[tokio::test]
async fn test_ollama_client_offline_probe_does_not_panic() {
    let client = OllamaClient::init_from_env_or_default().expect("Failed to init client");
    let _online = client.is_healthy().await;
}

#[test]
fn test_degraded_response_serialization() {
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
