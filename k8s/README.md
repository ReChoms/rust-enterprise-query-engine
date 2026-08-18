# Kubernetes & Podman Deployment Guide

This directory contains the production-grade Kubernetes manifests and Kustomize overlays for orchestrating the **Rust Enterprise Query Engine** alongside an Ollama LLM sidecar.

---

## 1. Local Container Build (Podman / OCI)

Build the minimal Distroless container image using Buildah cache mounts:

```bash
podman build -t rust-enterprise-query-engine:latest -f Containerfile .
```

---

## 2. Standalone Rootless Container Run (Podman)

To run the container locally with volume persistence, ensure you pass the rootless user namespace and SELinux volume relabeling flags:

```bash
podman run -d \
  --name rust-enterprise-query-engine \
  -p 8080:8080 \
  --userns=keep-id:uid=65532,gid=65532 \
  -v ./data:/app/data:Z \
  -e PORT=8080 \
  -e HOST=0.0.0.0 \
  -e VECTOR_DB_URI=/app/data/sap_vectors \
  -e CSV_PATH=/app/data/kna1.csv \
  -e MODEL_CACHE_DIR=/app/data \
  rust-enterprise-query-engine:latest
```

---

## 3. Local Clusterless Validation with `podman play kube`

Podman natively runs Kubernetes Pod manifests locally without requiring Minikube or Kind:

```bash
# Generate rendered dev manifest via Kustomize and execute directly with Podman
kubectl kustomize k8s/overlays/dev | podman play kube -
```

---

## 4. Kubernetes Cluster Deployment (Kustomize)

### Development Overlay
```bash
kubectl apply -k k8s/overlays/dev
```

### Production Overlay (3 Replicas + 8GB Limit)
```bash
kubectl apply -k k8s/overlays/prod
```
