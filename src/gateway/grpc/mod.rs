//! Altitude 3: Ingress Gateway (Door 2: Tonic gRPC)
//!
//! Exposes typed binary RPC endpoints generated from Protobuf contracts.

pub mod proto {
    tonic::include_proto!("query_engine");
}

pub use proto::query_service_server::{QueryService, QueryServiceServer};
pub use proto::*;

pub mod service;
pub use service::QueryServiceImpl;
