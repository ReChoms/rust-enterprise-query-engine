//! Axum HTTP Gateway (Altitude 3: Ingress)
//!
//! Exposes REST endpoints, configures Linux kernel socket parameters,
//! and applies Layer 7 load-shedding to protect against concurrency saturation.

pub mod middleware;
pub mod routes;
pub mod socket;

use axum::error_handling::HandleErrorLayer;
use axum::routing::{get, post};
use axum::Router;
use std::net::SocketAddr;
use tower::limit::ConcurrencyLimitLayer;
use tower::load_shed::LoadShedLayer;
use tower::ServiceBuilder;
use tracing::info;

pub use middleware::handle_load_shed_error;
pub use routes::{health_handler, router_query_handler, semantic_query_handler, sql_query_handler};
pub use socket::create_tuned_tcp_listener;

use crate::pipelines::query::AppState;

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/query/sql", post(sql_query_handler))
        .route("/query/semantic", post(semantic_query_handler))
        .route("/query", post(router_query_handler))
        .with_state(state)
        // Apply Layer 7 saturation protection: 5µs HTTP 503 shedding above 128 concurrent queries
        .layer(
            ServiceBuilder::new()
                .layer(HandleErrorLayer::new(handle_load_shed_error))
                .layer(LoadShedLayer::new())
                .layer(ConcurrencyLimitLayer::new(128)),
        )
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C signal. Initiating graceful drain & shutdown...");
        },
        _ = terminate => {
            info!("Received SIGTERM signal. Initiating graceful drain & shutdown...");
        },
    }
}

pub async fn start_server(state: AppState, addr: SocketAddr) -> anyhow::Result<()> {
    let app = create_router(state);
    let listener = create_tuned_tcp_listener(addr)?;
    info!("Axum REST API listening on http://{}", addr);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    info!("Axum REST API server shutdown completed gracefully.");
    Ok(())
}
