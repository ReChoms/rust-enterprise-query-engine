//! Altitude 0: Common Bedrock
//!
//! Contains zero-dependency data structures, serializable request/response models,
//! and process-wide logging instrumentation.

pub mod telemetry;
pub mod types;

pub use telemetry::init_logger;
pub use types::*;
