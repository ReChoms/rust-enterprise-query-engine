use std::sync::Arc;
use tonic::Request;
use rust_enterprise_query_engine::engines::embeddings::load_model;
use rust_enterprise_query_engine::engines::sql::init_datafusion;
use rust_enterprise_query_engine::gateway::grpc::{
    HealthCheckRequest, QueryService, QueryServiceImpl,
};
use rust_enterprise_query_engine::pipelines::query::{AppState, OllamaClient};

async fn setup_test_service() -> QueryServiceImpl {
    let (model, tokenizer) = load_model().await.expect("Failed to load model");
    let sql_engine = init_datafusion().await.expect("Failed to init DataFusion");
    let llm_client = OllamaClient::init_from_env_or_default().expect("Failed init client");
    let state = AppState {
        llm_client: Arc::new(llm_client),
        sql_engine: Arc::new(sql_engine),
        embedding_model: Arc::clone(&model),
        tokenizer: Arc::clone(&tokenizer),
        vector_db_uri: "data/sap_vectors".to_string(),
    };
    QueryServiceImpl::new(state)
}

#[tokio::test]
async fn test_grpc_check_health() {
    let service = setup_test_service().await;
    let resp = service.check_health(Request::new(HealthCheckRequest {})).await.unwrap().into_inner();
    assert!(resp.status == "healthy" || resp.status == "degraded" || resp.status == "unhealthy");
}

#[tokio::test]
async fn test_grpc_execute_sql() {
    use rust_enterprise_query_engine::gateway::grpc::SqlQueryRequest;
    let service = setup_test_service().await;
    let req = Request::new(SqlQueryRequest {
        query: "SELECT kunnr, name1 FROM kna1 LIMIT 2".to_string(),
    });
    let resp = service.execute_sql(req).await.unwrap().into_inner();
    assert_eq!(resp.row_count, 2);
    assert!(!resp.json_lines.is_empty());

    let attack_req = Request::new(SqlQueryRequest {
        query: "DROP TABLE kna1".to_string(),
    });
    let err = service.execute_sql(attack_req).await.unwrap_err();
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}

#[tokio::test]
async fn test_grpc_execute_semantic_validation() {
    use rust_enterprise_query_engine::gateway::grpc::SemanticQueryRequest;
    let service = setup_test_service().await;

    // 1. Empty query rejection
    let empty_req = Request::new(SemanticQueryRequest {
        query: "   ".to_string(),
        top_k: 5,
    });
    let err = service.execute_semantic(empty_req).await.unwrap_err();
    assert_eq!(err.code(), tonic::Code::InvalidArgument);

    // 2. Oversized query ceiling
    let long_req = Request::new(SemanticQueryRequest {
        query: "a".repeat(5_000),
        top_k: 5,
    });
    let err = service.execute_semantic(long_req).await.unwrap_err();
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}
