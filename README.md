# rust-enterprise-query-engine

A high-performance, pure-Rust Hybrid RAG & Analytical Query Engine bridging structured enterprise SAP data with modern local AI.

The engine parses natural language queries and dynamically routes them either to an exact columnar SQL engine (Apache DataFusion) or a semantic vector database (LanceDB) based on mathematical confidence and AST-validated schema constraints.

---

## What the Engine Does

* **Hybrid Intent Routing**: Dynamically classifies natural language questions using local LLMs (via Ollama) to route analytical aggregations to deterministic SQL and conceptual inquiries to semantic vector search.
* **Deterministic Columnar SQL**: Executes queries against Apache Arrow tables using Apache DataFusion, protected by an AST firewall that blocks dangerous SQL statements (e.g., `DROP`, `ALTER`, `TRUNCATE`, multi-statement injections).
* **Dense Vector Semantic Search**: Ingests structured tabular data (such as SAP `KNA1` customer master tables) into LanceDB, generating normalized dense vector embeddings using Hugging Face's `candle` ML framework (`BAAI/bge-base-en-v1.5`) directly on the CPU.
* **Absence Proofs & Fault Tolerance**: If vector search misses an identifier or the local LLM times out, the engine executes deterministic fallback keyword scans and returns verified fallback chunks or proven absences without panicking.
* **Unix I/O Composability**: Operates as a CLI utility supporting direct argument queries, piped STDIN line-delimited streams (`cat queries.txt | bridge ask`), or a background dual-protocol network daemon.

---

## How It Is Built (Architecture & Systems Physics)

The engine is engineered under strict low-latency systems engineering constraints to maximize CPU cache locality, guarantee bounded memory footprints, and avoid external runtime dependencies:

### 1. Dual-Protocol Hexagonal Microservice
The service exposes a dual-door network interface from a single compiled binary:
* **Door 1 (Axum REST / JSON)**: Public HTTP interface listening on port `8080` for browser clients, curl, and management endpoints.
* **Door 2 (Tonic gRPC / Protobuf v3)**: High-speed typed binary wire protocol listening on port `50051` for internal microservice RPCs.
* **Zero Memory Duplication**: Both protocol doors share the exact same in-memory `AppState` singletons (`Arc<BertModel>`, `Arc<Tokenizer>`, `Arc<SessionContext>`, `Arc<OllamaClient>`) in RAM.
* **100% Pure-Rust Protobuf Pipeline**: Compiles Protobuf definitions using `protox` and `tonic-build` directly in `build.rs`, eliminating host C++ `protoc` build dependencies.

### 2. Zero-Copy Networking & Memory Layout
* **Streaming Arrow JSON-Lines**: Serializes Arrow `RecordBatch` rows directly into network socket buffers via `LineDelimitedWriter`, completely bypassing intermediate `serde_json::Value` DOM heap allocations.
* **Contiguous Embedding Buffers (`EmbeddingBatch`)**: Stores dense vectors in a contiguous `Box<[f32]>` stride buffer, enabling sequential 64-byte L1/L2 CPU cache prefetching and eliminating per-vector pointer indirection.
* **Numerical Safety Firewalls**: Vector normalization uses Blue's Scaled Hypotenuse algorithm with explicit `f32::MIN_POSITIVE` subnormal float guards to prevent float division overflows and CPU microcode trap stalls.

### 3. Layer 4 Socket Tuning & Layer 7 Load Shedding
* **Linux Kernel Socket Tuning**: Configured via `socket2` with `TCP_NODELAY` (disabling Nagle's algorithm and 40ms delayed-ACK stalls), `SO_REUSEADDR`, `SO_REUSEPORT`, and 128KB kernel buffers (`SO_RCVBUF` / `SO_SNDBUF`).
* **Tower Layer 7 Load Shedding**: Bounds concurrent queries to 128 tasks via `ConcurrencyLimitLayer`. Requests exceeding the threshold are shed in ~5µs with HTTP 503 Service Unavailable, preventing out-of-memory (OOM) kernel panics.

### 4. The River Architecture (Downward Layered Gravity)
The codebase enforces strict unidirectional dependency flow across 4 altitudes:
* **Altitude 3: `gateway/` (The Sky / Ingress)**: CLI parsers, STDIN stream handlers, Axum REST routes, Tonic gRPC service, tuned TCP listeners, and Tower middleware.
* **Altitude 2: `pipelines/` (The Flow / Orchestration)**: Ingestion chunking & JIT deduplication, natural language query classification, LLM firewalls, and deterministic absence proofs.
* **Altitude 1: `engines/` (The Bedrock / Hermetic Compute)**: DataFusion SQL engine, LanceDB vector kNN search, Candle BERT embeddings, and SQL AST mutation guards.
* **Altitude 0: `common/` (The Ground / Primitives)**: Zero-dependency domain models, serialized request/response DTOs, and telemetry initialization.

---

## Usage Guide

### 1. Run the Dual-Protocol Daemon
Launch both Axum REST (port 8080) and Tonic gRPC (port 50051) concurrently:
```bash
cargo run -- serve --host 0.0.0.0 --port 8080 --grpc-port 50051
```

### 2. Ingest SAP Enterprise Data
Ingest tabular CSV rows into the local LanceDB vector database:
```bash
cargo run -- ingest --file data/kna1.csv --batch-size 256 --overwrite
```

### 3. Primary AI Query Router
Query through the hybrid router (automatically evaluates whether to execute SQL or semantic vector search):
```bash
cargo run -- ask "Which customer accounts are based in Berlin?"
```

### 4. Unix Pipeline (Piped STDIN)
Stream multiple queries sequentially through Unix pipes:
```bash
cat queries.txt | cargo run -- ask
```

### 5. Direct Columnar SQL
Run AST-validated SQL directly against the in-memory DataFusion catalog:
```bash
cargo run -- execute-sql "SELECT kunnr, name1, ort01 FROM kna1 LIMIT 5"
```

### 6. Cluster Health Probe
Check health status of the LanceDB database and local Ollama model connection:
```bash
cargo run -- health
```

---

## Status

v1 not finished so far.

---

## Disclaimer

This is a personal free-time project built strictly for educational and systems engineering research purposes. While the architecture is designed to handle generic enterprise tabular data, a standard SAP dataset from Kaggle (under the MIT license) was selected as the test corpus to demonstrate handling complex, structured enterprise schemas. 

This project is entirely independent and has absolutely no reference to or connection with my actual employer or daily work.
