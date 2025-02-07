use ebsentinel_core::run_ebsentinel_ebpf;
use rusqlite::{types::{FromSql, FromSqlResult, ToSqlOutput, ValueRef}, Connection, ToSql};
use tokio::signal;
use serde::{Deserialize, Serialize};
use serde_rusqlite::*;
#[derive(Debug, Clone,Serialize,Deserialize)]
struct Syscalls{
    syscalls: Vec<f32>
}

// Custom implementation for `Vec<f32>` serialization to SQLite BLOB
impl ToSql for Syscalls {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput> {
        let serialized = bincode::serialize(self).map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
        Ok(ToSqlOutput::from(serialized))
    }
}

/// Custom implementation for `Vec<f32>` deserialization from SQLite BLOB
impl FromSql for Syscalls {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        match value.as_blob() {
            Ok(blob) => bincode::deserialize(blob).map_err(|e| rusqlite::types::FromSqlError::Other(Box::new(e))),
            Err(e) => Err(e),
        }
    }
}


#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let conn = Connection::open("ebsentinel.db")?;
    conn.execute(
        "create table if not exists train (
             row_id integer primary key,
             syscalls blob not null 
         )",
        [],
    )?;
    conn.execute(
        "create table if not exists test (
             row_id integer primary key,
             syscalls blob not null 
         )",
        [],
    )?;
    for i in 0..10000 {
        let data = Syscalls{syscalls:vec![i as f32 + 1.0;512]};
        // Insert serialized data
        conn.execute("INSERT INTO train (syscalls) VALUES (?)", [&data])?;
    }


    // Retrieve and deserialize
    let retrieved_vec: Syscalls = conn.query_row("SELECT syscalls FROM train", [], |row| row.get(0))?;
    
    println!("Retrieved Vec<f32>: {:?}", retrieved_vec.syscalls);
    let _=run_ebsentinel_ebpf(5222);
    
    println!("Hello, world!");
    let ctrl_c = signal::ctrl_c();
    println!("Waiting for Ctrl-C...");
    ctrl_c.await?;
    println!("Exiting...");

    Ok(())
}
