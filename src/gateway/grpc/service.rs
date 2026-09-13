use tonic::{Request, Response, Status};
use crate::pipelines::query::AppState;
use super::proto::query_service_server::QueryService;
use super::proto::*;

#[derive(Clone)]
pub struct QueryServiceImpl {
    pub state: AppState,
}

impl QueryServiceImpl {
    pub fn new(state: AppState) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl QueryService for QueryServiceImpl {
    async fn check_health(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let llm_online = self.state.llm_client.is_healthy().await;
        let (vector_db_connected, total_records) =
            match crate::engines::vector::check_lancedb_health(&self.state.vector_db_uri).await {
                Ok(count) => (true, count as u64),
                Err(_) => (false, 0),
            };
        let status = if llm_online && vector_db_connected {
            "healthy"
        } else if vector_db_connected {
            "degraded"
        } else {
            "unhealthy"
        };
        Ok(Response::new(HealthCheckResponse {
            status: status.to_string(),
            llm_connected: llm_online,
            vector_db_connected,
            total_records,
        }))
    }

    async fn execute_sql(
        &self,
        req: Request<SqlQueryRequest>,
    ) -> Result<Response<SqlQueryResponse>, Status> {
        let query = req.into_inner().query;
        let max_chars = std::env::var("MAX_QUERY_CHARS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4_096);
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Err(Status::invalid_argument("SQL query cannot be empty"));
        }
        if trimmed.len() > max_chars {
            return Err(Status::invalid_argument(format!(
                "SQL query exceeds maximum length of {} characters",
                max_chars
            )));
        }

        let batches = crate::engines::sql::execute_sql_query(&self.state.sql_engine, trimmed)
            .await
            .map_err(|e| Status::invalid_argument(format!("SQL Execution Error: {}", e)))?;

        let mut raw_lines = Vec::new();
        crate::engines::sql::write_record_batches_as_json_lines(&batches, &mut raw_lines)
            .map_err(|e| Status::internal(format!("Serialization Error: {}", e)))?;
        let row_count = batches.iter().map(|b| b.num_rows()).sum::<usize>() as u32;

        Ok(Response::new(SqlQueryResponse {
            json_lines: raw_lines.into(),
            row_count,
        }))
    }

    async fn execute_semantic(
        &self,
        req: Request<SemanticQueryRequest>,
    ) -> Result<Response<SemanticQueryResponse>, Status> {
        let inner = req.into_inner();
        let max_chars = std::env::var("MAX_QUERY_CHARS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4_096);
        let trimmed = inner.query.trim();
        if trimmed.is_empty() {
            return Err(Status::invalid_argument("Semantic query cannot be empty"));
        }
        if trimmed.len() > max_chars {
            return Err(Status::invalid_argument(format!(
                "Semantic query exceeds maximum length of {} characters",
                max_chars
            )));
        }

        let result = crate::pipelines::query::run_semantic_pipeline(&self.state, trimmed)
            .await
            .map_err(|e| Status::internal(format!("Semantic pipeline error: {}", e)))?;

        let (answer, records) = match result {
            crate::pipelines::query::SemanticQueryResult::Verified(v) => (
                v.answer,
                vec![RetrievedRecord {
                    kunnr: v.source_chunk_id,
                    sentence: v.exact_quote,
                    score: if v.answer_found { 1.0 } else { 0.0 },
                }],
            ),
            crate::pipelines::query::SemanticQueryResult::Degraded(d) => (
                d.message,
                d.retrieved_chunks
                    .into_iter()
                    .map(|c| RetrievedRecord {
                        kunnr: c.chunk_id,
                        sentence: c.content,
                        score: 0.0,
                    })
                    .collect(),
            ),
        };

        Ok(Response::new(SemanticQueryResponse { answer, records }))
    }
}
