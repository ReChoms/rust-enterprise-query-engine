use arrow_array::Array;
use futures::StreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::collections::HashSet;

/// Performs a Just-In-Time O(1) primary key lookup against LanceDB for an in-flight batch of IDs.
///
/// Prevents redundant neural vector embedding calculations by identifying already-indexed records.
pub async fn find_existing_in_db(
    vector_db: &lancedb::Connection,
    ids_for_sql: &[String],
) -> HashSet<String> {
    let mut existing = HashSet::new();
    if ids_for_sql.is_empty() {
        return existing;
    }

    if let Ok(target_table) = vector_db.open_table("customers").execute().await {
        let filter = format!("kunnr IN ({})", ids_for_sql.join(", "));
        if let Ok(mut stream) = target_table.query().only_if(filter).execute().await {
            while let Some(batch_result) = stream.next().await {
                let Ok(batch) = batch_result else { continue };
                let Some(col) = batch.column_by_name("kunnr") else { continue };
                let Some(k_arr) = col.as_any().downcast_ref::<arrow_array::StringArray>() else {
                    continue;
                };

                for i in 0..k_arr.len() {
                    existing.insert(k_arr.value(i).to_string());
                }
            }
        }
    }
    existing
}
