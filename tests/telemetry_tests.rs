use datafusion::prelude::SessionContext;
use rust_enterprise_query_engine::engines::sql::{execute_sql_query, write_record_batches_as_json_lines};
use std::sync::{Arc, Mutex};
use tracing_subscriber::fmt::format::FmtSpan;

#[derive(Clone)]
struct SharedBuf(Arc<Mutex<Vec<u8>>>);
impl std::io::Write for SharedBuf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> { self.0.lock().unwrap().extend_from_slice(buf); Ok(buf.len()) }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

#[tokio::test]
async fn test_telemetry_spans_and_unpolluted_stdout() {
    let trace_buf = Arc::new(Mutex::new(Vec::new()));
    let writer = SharedBuf(trace_buf.clone());
    let subscriber = tracing_subscriber::fmt()
        .with_writer(move || writer.clone())
        .with_span_events(FmtSpan::ENTER)
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);

    let ctx = SessionContext::new();
    let batches = execute_sql_query(&ctx, "SELECT 42 AS val").await.expect("query failed");

    let mut data_stdout = Vec::new();
    write_record_batches_as_json_lines(&batches, &mut data_stdout).expect("serialization failed");
    let stdout_str = String::from_utf8(data_stdout).expect("valid utf8");
    let logs = String::from_utf8(trace_buf.lock().unwrap().clone()).expect("valid utf8");

    assert!(logs.contains("engine_datafusion_sql"), "Expected engine span in telemetry stream");
    assert_eq!(stdout_str.trim(), r#"{"val":42}"#, "Data stream must be pure JSON lines");
}
