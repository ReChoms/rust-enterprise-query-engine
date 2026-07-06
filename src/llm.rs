use crate::models::{OllamaRequest, OllamaResponse, SemanticResponse, SqlResponse};
use std::collections::HashMap;
use std::error::Error;

/// Sends a prompt to the local Ollama server running Llama 3.2
pub async fn ask_llm(prompt: &str) -> Result<String, Box<dyn Error>> {
    let req_body = OllamaRequest {
        model: "llama3.2:latest".to_string(),
        prompt: prompt.to_string(),
        stream: false,
    };

    let client = reqwest::Client::new();
    let res = client
        .post("http://localhost:11434/api/generate")
        .json(&req_body)
        .send()
        .await?
        .json::<OllamaResponse>()
        .await?;
    Ok(res.response)
}

pub fn build_routing_prompt(user_question: &str) -> String {
    format!(
        "You are an expert SAP data engineer. Read the user's question and decide if it requires exact SQL or SEMANTIC search.\n\nDatabase Schema for table `kna1`:\n- kunnr (String): Customer Number / ID\n- name1 (String): Customer Name\n- ort01 (String): City\n- land1 (String): Country Code (e.g., 'US', 'DE')\n\nRULES:\n1. You must ONLY output raw JSON. Do not wrap it in markdown. Do not add conversational text.\n2. The JSON must have two keys: \"route\" (either \"SQL\" or \"SEMANTIC\") and \"query\" (the generated SQL string, or blank).\n\nExamples:\nQ: \"How many customers are in Berlin?\"\nA: {{\"route\": \"SQL\", \"query\": \"SELECT count(*) FROM kna1 WHERE ort01 = 'Berlin'\"}}\n\nQ: \"Show me the names of 5 customers in the US.\"\nA: {{\"route\": \"SQL\", \"query\": \"SELECT name1 FROM kna1 WHERE land1 = 'US' LIMIT 5\"}}\n\nQ: \"Find customers who are large tech manufacturers.\"\nA: {{\"route\": \"SEMANTIC\", \"query\": \"\"}}\n\nUser Question: \"{}\"\nA: ",
        user_question
    )
}

pub fn build_sql_prompt(user_question: &str) -> String {
    format!(
        "You are an expert SAP data engineer. Read the user's question and write the exact SQL query required.\n\nDatabase Schema for table `kna1`:\n- kunnr (String): Customer Number / ID\n- name1 (String): Customer Name\n- ort01 (String): City\n- land1 (String): Country Code (e.g., 'US', 'DE')\n\nRULES:\n1. You must ONLY output raw JSON. Do not wrap it in markdown. Do not add conversational text.\n2. The JSON must have one key: \"query\" containing the generated SQL string.\n\nUser Question: \"{}\"\nA: ",
        user_question
    )
}

pub fn verify_and_parse_sql_generation(raw_output: &str) -> Result<SqlResponse, Box<dyn Error>> {
    let clean_json = raw_output
        .trim()
        .strip_prefix("```json")
        .unwrap_or(raw_output.trim())
        .strip_prefix("```")
        .unwrap_or(raw_output.trim())
        .strip_suffix("```")
        .unwrap_or(raw_output.trim())
        .trim();

    let response: SqlResponse = serde_json::from_str(clean_json)
        .map_err(|e| format!("Failed to parse SqlResponse: {}. Raw text: {}", e, clean_json))?;
        
    Ok(response)
}

pub fn verify_and_parse_llm_generation(
    raw_output: &str,
    retrieved_chunks: &HashMap<String, String>,
) -> Result<SemanticResponse, Box<dyn Error>> {
    let clean_json = raw_output
        .trim()
        .strip_prefix("```json")
        .unwrap_or(raw_output.trim())
        .strip_prefix("```")
        .unwrap_or(raw_output.trim())
        .strip_suffix("```")
        .unwrap_or(raw_output.trim())
        .trim();

    let response: SemanticResponse = serde_json::from_str(clean_json)
        .map_err(|e| format!("Failed to parse SemanticResponse: {}. Raw text: {}", e, clean_json))?;

    if response.answer_found {
        let source_text = retrieved_chunks.get(&response.source_chunk_id).ok_or_else(|| {
            format!(
                "SECURITY VIOLATION: LLM cited a non-existent chunk ID: {}",
                response.source_chunk_id
            )
        })?;

        if !source_text.contains(&response.exact_quote) {
            return Err(format!(
                "SECURITY VIOLATION: Hallucination detected. The exact quote does not exist in the source document. Quote: '{}'",
                response.exact_quote
            )
            .into());
        }
    }

    Ok(response)
}

pub fn build_semantic_prompt(user_question: &str, chunks: &HashMap<String, String>) -> String {
    let mut context = String::new();
    for (id, text) in chunks {
        context.push_str(&format!("CHUNK ID: {}\nTEXT: {}\n\n", id, text));
    }

    format!(
        "You are an expert SAP data engineer. Answer the user's question using ONLY the provided chunks.\n\n\
        CONTEXT:\n{}\n\n\
        RULES:\n\
        1. You must ONLY output raw JSON. Do not wrap it in markdown. Do not add conversational text.\n\
        2. The JSON must exactly match this schema:\n\
           {{\n\
             \"answer_found\": boolean,\n\
             \"answer\": \"your detailed answer based ONLY on the text\",\n\
             \"exact_quote\": \"a verbatim, exact substring from the chunk that proves your answer\",\n\
             \"source_chunk_id\": \"the specific CHUNK ID you extracted the quote from\"\n\
           }}\n\
        3. If the answer is not found in the chunks, set `answer_found` to false, and leave the other fields blank.\n\n\
        User Question: \"{}\"\nA: ",
        context, user_question
    )
}
