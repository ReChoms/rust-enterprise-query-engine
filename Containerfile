# Stage 1: Builder
FROM rust:1.92-slim-bookworm AS builder

# Install OS build dependencies (protobuf-compiler required for LanceDB)
RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler \
    libprotobuf-dev \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/code_app

# Copy dependency manifests for cache layering
COPY Cargo.toml Cargo.lock ./

# Create dummy source to force dependency compilation
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build dependencies using Buildah / BuildKit persistent cache mounts
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/code_app/target \
    cargo build --release --locked

# Copy actual application source code
COPY src ./src

# Compile application and EXTRACT binary from the transient cache mount
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/code_app/target \
    touch src/main.rs && \
    cargo build --release --locked && \
    cp target/release/rust-enterprise-query-engine /app/rust-enterprise-query-engine

# Pre-create data directory with explicit ownership for Distroless 'nonroot' UID 65532
RUN mkdir -p /app/data && chown -R 65532:65532 /app/data

# Stage 2: Minimal Runtime
FROM gcr.io/distroless/cc-debian12

WORKDIR /app

# Transfer the compiled binary
COPY --from=builder --chown=65532:65532 /app/rust-enterprise-query-engine /app/

# Transfer the pre-chowned data directory to prevent OS Error 13 (Permission Denied)
COPY --from=builder --chown=65532:65532 /app/data /app/data

# Drop all root privileges
USER 65532:65532

# Define mount point for Kubernetes PersistentVolumeClaims (PVCs)
VOLUME ["/app/data"]

# 12-Factor App Environment Binding
ENV PORT=8080
ENV HOST=0.0.0.0
ENV VECTOR_DB_URI=/app/data/sap_vectors
ENV CSV_PATH=/app/data/kna1.csv
ENV MODEL_CACHE_DIR=/app/data

EXPOSE 8080

# Exec-form definition guarantees process receives SIGTERM from Kubelet for 0-downtime
ENTRYPOINT ["/app/rust-enterprise-query-engine"]
CMD ["serve"]
