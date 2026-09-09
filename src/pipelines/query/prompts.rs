use std::collections::HashMap;
use std::sync::Arc;

pub const SAP_KNA1_SCHEMA: &str = "\
Database Schema for table `kna1`:\n\
- kunnr (String): Customer Number / ID\n\
- name1 (String): Customer Name\n\
- ort01 (String): City\n\
- land1 (String): Country Code (e.g., 'US', 'DE')\n";

pub fn build_routing_prompt(user_question: &str) -> String {
    format!(
        "You are an expert SAP data engineer. Read the user's question and decide if it requires exact SQL or SEMANTIC search.\n\n{}\nRULES:\n1. You must ONLY output raw JSON. Do not wrap it in markdown. Do not add conversational text.\n2. The JSON must have one key: \"route\" (either \"SQL\" or \"SEMANTIC\").\n\nExamples:\nQ: \"How many customers are in Berlin?\"\nA: {{\"route\": \"SQL\"}}\n\nQ: \"Show me the names of 5 customers in the US.\"\nA: {{\"route\": \"SQL\"}}\n\nQ: \"Find customers who are large tech manufacturers.\"\nA: {{\"route\": \"SEMANTIC\"}}\n\nUser Question: \"{}\"\nA: ",
        SAP_KNA1_SCHEMA, user_question
    )
}

pub fn build_sql_prompt(user_question: &str) -> String {
    format!(
        "You are an expert SAP data engineer. Read the user's question and write the exact SQL query required.\n\n{}\nRULES:\n1. You must ONLY output raw JSON. Do not wrap it in markdown. Do not add conversational text.\n2. The JSON must have one key: \"query\" containing the generated SQL string.\n\nUser Question: \"{}\"\nA: ",
        SAP_KNA1_SCHEMA, user_question
    )
}

pub fn build_semantic_prompt(
    user_question: &str,
    chunks: &HashMap<String, Arc<str>>,
) -> String {
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

pub fn build_question_parser_prompt(user_question: &str) -> String {
    format!(
        "You are an expert SAP data engineer. Your task is to parse the user's question into pure semantic intent and explicit filters.\n\n\
        RULES:\n\
        1. You must ONLY output raw JSON. Do not wrap it in markdown. Do not add conversational text.\n\
        2. The JSON must exactly match this schema:\n\
           {{\n\
             \"intent\": \"The pure semantic meaning, without hard IDs or strict exclusions\",\n\
             \"filters\": [\"EXACT_ID_1\", \"NOT EXACT_ID_2\"]\n\
           }}\n\
        3. If there are no explicit IDs or exclusions, leave the `filters` array empty.\n\n\
        Examples:\n\
        Q: \"Which customer in Berlin is NOT KUNNR 1000?\"\n\
        A: {{\"intent\": \"customer in Berlin\", \"filters\": [\"NOT KUNNR 1000\"]}}\n\n\
        Q: \"Find SAP customer 00001042.\"\n\
        A: {{\"intent\": \"Find SAP customer\", \"filters\": [\"00001042\"]}}\n\n\
        User Question: \"{}\"\nA: ",
        user_question
    )
}
