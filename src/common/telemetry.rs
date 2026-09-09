use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Initializes the application-wide tracing subscriber routing logs strictly to STDERR.
///
/// Why: Routing logs to STDERR prevents polluting STDOUT, enabling clean Unix pipeline
/// composition (e.g., `cargo run | jq .`) without log artifacts interfering with data.
pub fn init_logger() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,tower_http=debug,axum::rejection=trace"));

    let stderr_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_target(true);

    tracing_subscriber::registry()
        .with(filter)
        .with(stderr_layer)
        .init();
}
