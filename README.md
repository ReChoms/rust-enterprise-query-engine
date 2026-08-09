# rust-enterprise-query-engine

A high-performance, pure-Rust Hybrid RAG (Retrieval-Augmented Generation) architecture built for structured enterprise data. 

This engine bridges the gap between traditional tabular databases and modern AI semantic search. It dynamically parses natural language questions, routing them either to an exact SQL analytical engine or a fuzzy vector database based on mathematical confidence.

## Core Architecture

This project was built under strict systems engineering constraints to maximize CPU efficiency and minimize memory overhead:

* **100% Pure Rust:** No Python, no C++ bindings, and no massive environment bloat. 
* **Zero-Copy Memory (Apache Arrow):** The system relies on the Apache Arrow memory format. The SQL engine, the Embedding Neural Networks, and the Vector Database all share the exact same RAM pointers, completely eliminating the CPU overhead of translating data between layers (Zero-ETL).
* **Pure-Rust Embeddings:** Uses Hugging Face's `candle` ML framework to execute transformer models locally on the CPU without relying on external C++ ONNX bindings.
* **Hybrid Retrieval (LanceDB + DataFusion):** Merges dense vector similarity search (`LanceDB`) with exact deterministic SQL keyword matching (`Apache DataFusion`) to prevent the LLM from hallucinating alphanumeric identifiers.
* **Local LLM Routing:** Uses local LLMs (via `Ollama`) to securely parse natural language intents into strict JSON schemas and dynamically generate safe AST-validated SQL.

## Current Roadmap / Pending Optimizations

**Status:** The system is currently undergoing a deep-dive **Architectural Code Review**. We are actively mapping out the data flow and hardening the architecture before shipping new features.

The remaining technical milestones and planned future add-ons include:

* **Universal Dynamic Schema:** Ripping out the hardcoded `Kna1Row` structs so the ingestion pipeline can dynamically read the headers of *any* CSV file (MARA, VBAK, etc.) and generate the LanceDB schema on the fly.
* **REST API Migration via Axum:** Transitioning the tool from a short-lived CLI process to a long-running REST API server, drastically improving performance by holding the massive 400MB HuggingFace model permanently in memory across requests.
* **LLM Graceful Degradation:** Implementing strict fail-safes so that if the local AI hallucination gate fails or the LLM server crashes, the system automatically falls back to deterministic, mathematically pure vector/SQL searches without breaking the user experience.
* **Relational Data Extraction:** Transitioning the ingestion pipeline from flat string concatenation to strict JSON schemas to preserve relational boundaries for the LLM.
* **Deterministic Absence Proofs:** Implementing a fallback corpus scan to mathematically prove a record does not exist when vector top-K searches return empty, preventing false negatives.
* **Robustness:** Consolidating networking to the async `reqwest` client, migrating error handling to `anyhow`, and ensuring runtime-relative paths for CI/CD portability.

## Disclaimer

This is a personal free-time project built strictly for educational purposes. While the architecture is designed to handle generic tabular data, I selected a standard SAP dataset from Kaggle (under the MIT license) as the test corpus to demonstrate handling complex, structured enterprise schemas. 

This project is entirely independent and has absolutely no reference to or connection with my actual employer or daily work.
