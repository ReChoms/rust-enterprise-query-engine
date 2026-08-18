# Rust Enterprise Query Engine — Architecture Specification

## 1. Executive Overview

**Rust Enterprise Query Engine** is a high-throughput, memory-safe analytical and semantic query engine engineered in pure Rust. It bridges enterprise relational ERP data (e.g., SAP KNA1/VBAK) and modern AI systems through a dual-engine architecture:

1. **Analytical SQL Engine (Zero-Copy):** Apache DataFusion execution over Apache Arrow columnar memory with AST-enforced security boundaries.
2. **Neural Semantic Engine (kNN & Hybrid Search):** In-process Candle HuggingFace BERT embeddings (`BAAI/bge-base-en-v1.5`) with LanceDB vector storage, deterministic absence proofing, and grounded quotation verification.

---

## 2. Global Component Topology

```text
               ┌────────────────────────────────────────────────────────┐
               │              Client Ingestion / Query                  │
               │         (CLI / Unix Pipes / Axum REST API)             │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
             ┌────────────────────────────────────────────────────────────┐
             │                     LLM Intent Router                      │
             │           (Ollama / Llama3.2 Query Classification)          │
             └───────────────┬────────────────────────────┬───────────────┘
                             │                            │
                     [Route == SQL]               [Route == SEMANTIC]
                             │                            │
                             ▼                            ▼
              ┌─────────────────────────────┐   ┌───────────────────────────┐
              │      SQL Query Pipeline     │   │  Semantic Query Pipeline  │
              ├─────────────────────────────┤   ├───────────────────────────┤
              │ 1. LLM SQL Synthesizer      │   │ 1. LLM Intent & Filter    │
              │ 2. AST Security Guard       │   │    Parser                 │
              │ 3. DataFusion Engine        │   │ 2. Candle BERT Embeddings │
              │ 4. Zero-Copy Arrow Stream   │   │ 3. LanceDB kNN Search     │
              └──────────────┬──────────────┘   │ 4. Deterministic Fallback │
                             │                  │ 5. LLM Answer Verifier    │
                             │                  └─────────────┬─────────────┘
                             │                                │
                             ▼                                ▼
              ┌─────────────────────────────────────────────────────────────┐
              │                 Zero-Copy JSON-Lines Stream                 │
              │               (STDOUT / HTTP Response Body)                 │
              └─────────────────────────────────────────────────────────────┘
```

---

## 3. Core Engine Subsystems

### 3.1 Analytical SQL Subsystem (`src/sql_engine.rs`)
* **Engine:** Apache DataFusion 38.0 executing over columnar Apache Arrow 51 dataframes.
* **Security Validation:** AST parser (`sqlparser-rs`) inspects SQL prior to execution, rejecting multi-statement scripts, schema alterations (`DROP`, `ALTER`, `TRUNCATE`), data exfiltration copies (`COPY`), and non-SELECT expressions.
* **Streaming Serializer:** `write_record_batches_as_json_lines` serializes Arrow memory directly to output write streams (`std::io::Write`) with zero heap duplication.

### 3.2 Neural Semantic Subsystem (`src/vector_db.rs` & `src/embeddings.rs`)
* **Embedding Model:** Local, CPU-accelerated `BAAI/bge-base-en-v1.5` BERT transformer model executing via HuggingFace `candle-core` / `candle-transformers`.
* **Vector Store:** Embedded LanceDB 0.15.0 with Lance columnar format.
* **Absence Fallback Proof:** If vector kNN search yields insufficient semantic confidence, a deterministic SQL absence search is triggered to mathematically prove non-existence before generating answers.

### 3.3 AI & Verification Subsystem (`src/llm.rs`)
* **Ollama Client:** Connection-pooled HTTP client with exponential backoff and timeout safeguards.
* **Anti-Hallucination Firewall:** LLM-generated citations are verified against retrieved LanceDB text chunks using byte-for-byte substring matching. Hallucinated quotes trigger immediate rejection.

### 3.4 REST Microservice Subsystem (`src/server.rs`)
* **Framework:** Axum 0.7 + Tower HTTP.
* **Shared State:** `AppState` encapsulates `Arc<BertModel>`, `Arc<Tokenizer>`, `Arc<SessionContext>`, and `Arc<OllamaClient>`.
* **Endpoints:**
  * `GET /health` — Diagnostics & Kubernetes liveness/readiness probe.
  * `POST /query` — Dynamic LLM router dispatch.
  * `POST /query/sql` — Direct secured DataFusion SQL execution.
  * `POST /query/semantic` — Semantic RAG search.

---

## 4. Cloud Deployment & Kubernetes Topology

```text
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                            Kubernetes Pod                               │
 │                                                                         │
 │  ┌─────────────────────────────────┐   ┌─────────────────────────────┐  │
 │  │      Query Engine Container     │   │    Ollama Sidecar Container │  │
 │  │        (Axum REST Service)      │   │       (LLM Inference)       │  │
 │  │                                 │   │                             │  │
 │  │  • Port 8080                    │   │  • Port 11434               │  │
 │  │  • Distroless Base (~60MB)      │   │  • Localhost Loopback       │  │
 │  │  • Multi-threaded Tokio Runtime │   │  • GPU/CPU Accelerated      │  │
 │  └────────────────┬────────────────┘   └──────────────┬──────────────┘  │
 │                   │                                   │                 │
 │                   └─────────► localhost:11434 ◄───────┘                 │
 │                                                                         │
 │  ┌───────────────────────────────────────────────────────────────────┐  │
 │  │              Persistent Volume Claim (PVC): /data                 │  │
 │  │        • data/kna1.csv (Raw ERP Table)                            │  │
 │  │        • data/sap_vectors (LanceDB Columnar Vector Store)         │  │
 │  └───────────────────────────────────────────────────────────────────┘  │
 └─────────────────────────────────────────────────────────────────────────┘
```
