use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use rust_enterprise_query_engine::engines::embeddings::load_model;
use rust_enterprise_query_engine::engines::sql::init_datafusion;
use rust_enterprise_query_engine::gateway::http::{
    create_router, create_tuned_tcp_listener, handle_load_shed_error,
};
use rust_enterprise_query_engine::pipelines::query::{AppState, OllamaClient};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tower::ServiceExt;

#[tokio::test]
async fn test_http_health_endpoint() {
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

    let app = create_router(state);
    let response = app
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn test_http_sql_query_endpoint() {
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

    let app = create_router(state);
    let req_body = serde_json::json!({
        "query": "SELECT kunnr, name1 FROM kna1 LIMIT 2"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/query/sql")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let rows: Vec<serde_json::Value> = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(rows.len(), 2);
    assert!(rows[0].get("kunnr").is_some());
}

#[tokio::test]
async fn test_http_sql_query_injection_blocked() {
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

    let app = create_router(state);
    let req_body = serde_json::json!({
        "query": "DROP TABLE kna1"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/query/sql")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_create_tuned_tcp_listener_binds_and_accepts() {
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let listener = create_tuned_tcp_listener(addr).expect("Failed to create tuned listener");
    let local_addr = listener.local_addr().expect("Failed to get local addr");

    let client_task = tokio::spawn(async move {
        let mut stream = tokio::net::TcpStream::connect(local_addr)
            .await
            .expect("Failed to connect");
        stream
            .write_all(b"PING")
            .await
            .expect("Failed to write to tuned socket");
    });

    let (mut server_stream, _) = listener.accept().await.expect("Failed to accept connection");
    let mut buf = [0u8; 4];
    server_stream
        .read_exact(&mut buf)
        .await
        .expect("Failed to read from server stream");
    assert_eq!(&buf, b"PING");

    client_task.await.expect("Client task failed");
}

#[tokio::test]
async fn test_handle_load_shed_error_mappings() {
    let overloaded_err = Box::new(tower::load_shed::error::Overloaded::new());
    let (status, body) = handle_load_shed_error(overloaded_err).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        body.get("error").and_then(|v| v.as_str()),
        Some("Server overloaded, request shed to protect memory stability")
    );
    assert_eq!(body.get("status_code").and_then(|v| v.as_i64()), Some(503));

    let generic_err = Box::new(std::io::Error::other("unexpected middleware fault"));
    let (status_500, body_500) = handle_load_shed_error(generic_err).await;
    assert_eq!(status_500, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        body_500.get("error").and_then(|v| v.as_str()),
        Some("Unhandled middleware error: unexpected middleware fault")
    );
    assert_eq!(body_500.get("status_code").and_then(|v| v.as_i64()), Some(500));
}
