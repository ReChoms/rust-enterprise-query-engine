use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Serialize)]
struct OllamaRequest {
    pub model: String,
    pub prompt: String,
    pub stream: bool,
}

#[derive(Deserialize)]
struct OllamaResponse {
    pub response: String,
}

#[derive(Clone, Debug)]
pub struct OllamaClient {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl OllamaClient {
    /// Initializes an HTTP connection-pooled client from environment variables with fallback defaults.
    pub fn init_from_env_or_default() -> Result<Self> {
        let base_url = std::env::var("OLLAMA_HOST")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        let timeout_secs: u64 = std::env::var("OLLAMA_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        let model = std::env::var("OLLAMA_MODEL")
            .unwrap_or_else(|_| "llama3.2:latest".to_string());

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| anyhow!("Failed to build HTTP client: {}", e))?;

        Ok(Self {
            client,
            base_url,
            model,
        })
    }

    /// Fast health probe returning true if the Ollama daemon is reachable.
    pub async fn is_healthy(&self) -> bool {
        let url = format!("{}/api/tags", self.base_url);
        match self.client.get(&url).timeout(Duration::from_secs(2)).send().await {
            Ok(res) if res.status().is_success() => true,
            _ => false,
        }
    }

    /// Sends a prompt to the Ollama model with exponential backoff retry.
    pub async fn prompt_model(&self, prompt: &str) -> Result<String> {
        let req_body = OllamaRequest {
            model: self.model.clone(),
            prompt: prompt.to_string(),
            stream: false,
        };

        let url = format!("{}/api/generate", self.base_url);
        let mut last_err = None;

        for attempt in 0..2 {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_millis(500)).await;
            }

            match self.client.post(&url).json(&req_body).send().await {
                Ok(res) => {
                    let parsed = res.json::<OllamaResponse>().await?;
                    return Ok(parsed.response);
                }
                Err(e) => {
                    last_err = Some(e);
                }
            }
        }

        Err(anyhow!(
            "Failed to reach LLM at {} after retry: {}",
            url,
            last_err.map(|e| e.to_string()).unwrap_or_default()
        ))
    }
}
