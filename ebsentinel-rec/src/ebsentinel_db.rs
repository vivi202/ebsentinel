use rusqlite::{types::{FromSql, FromSqlResult, ToSqlOutput, ValueRef}, Connection, ToSql};
use serde::{Deserialize, Serialize};


#[derive(Debug, Clone,Serialize,Deserialize)]
pub struct Syscalls{
    pub syscalls: Vec<f32>
}

impl Syscalls {
    pub fn new(rates: Vec<f32>) -> Self{
        Self { syscalls: rates }
    }
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

pub struct EbsentinelDb{
    conn: Connection
}

impl EbsentinelDb {
    pub fn new(path: String) -> Self{
        let conn = Connection::open(path).unwrap();
        conn.execute(
            "create table if not exists train (
                 row_id integer primary key,
                 syscalls blob not null 
             )",
            [],
        ).unwrap();

        conn.execute(
            "create table if not exists test (
                 row_id integer primary key,
                 syscalls blob not null 
             )",
            [],
        ).unwrap();

        Self { conn }
    }

    pub fn add_train_data(&self,syscalls: &Syscalls){
        self.conn.execute("INSERT INTO train (syscalls) VALUES (?)", [syscalls]).unwrap();
    }
    
    pub fn add_test_data(&self,syscalls: &Syscalls){
        self.conn.execute("INSERT INTO test (syscalls) VALUES (?)", [syscalls]).unwrap();
    }
}