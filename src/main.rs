use anyhow::Result;
use rust_enterprise_query_engine::{common::telemetry, gateway};

#[tokio::main]
async fn main() -> Result<()> {
    telemetry::init_logger();

    if let Err(e) = gateway::run().await {
        eprintln!("Execution error: {e}");
        std::process::exit(1);
    }

    Ok(())
}
