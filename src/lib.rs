//! The River Architecture Engine Root
//!
//! A strict downward-layered data engine enforcing unidirectional gravity:
//! Gateway (Altitude 3) -> Pipelines (Altitude 2) -> Engines (Altitude 1) -> Common (Altitude 0)

pub mod common;
pub mod engines;
pub mod gateway;
pub mod pipelines;
