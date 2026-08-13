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

**Status:** The system is currently undergoing Phase 3 of the **Architectural Code Review**. The ingestion pipeline, hybrid routing, test coverage, and deterministic absence proofs have been completed.

The remaining technical milestones and planned future add-ons include:

* **LLM Graceful Degradation:** Implementing strict fail-safes so that if the Ollama server is unreachable, the system automatically falls back to deterministic vector/SQL searches without crashing.
* **Universal Dynamic Schema:** Ripping out the hardcoded `Kna1Row` structs so the ingestion pipeline can dynamically read the headers of *any* CSV file (MARA, VBAK, etc.) and generate the LanceDB schema on the fly.
* **Enterprise Cloud Orchestration:** Wrapping the core engine in an Axum REST API, containerizing it with Docker, and deploying it alongside an Ollama sidecar on Kubernetes using PVCs.
* **Observability & SLO Definition:** Defining latency targets and wiring OpenTelemetry tracing to visualize bottlenecks.
* **Advanced RAG Architecture:** Implementing two-stage retrieval (Cross-Encoders) and Semantic Caching to eliminate expensive LLM inference for identical queries.
* **Database Indexing:** Adding IVF-PQ vector indexing and scalar indices to ensure query latency stays microsecond-fast at enterprise scale.

## Disclaimer

This is a personal free-time project built strictly for educational purposes. While the architecture is designed to handle generic tabular data, I selected a standard SAP dataset from Kaggle (under the MIT license) as the test corpus to demonstrate handling complex, structured enterprise schemas. 

This project is entirely independent and has absolutely no reference to or connection with my actual employer or daily work.
