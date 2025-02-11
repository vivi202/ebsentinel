use clap::Parser;

#[derive(Parser)]
pub struct Cli{
    #[arg(value_name = "PID")]
    pub pid: u32,
    #[arg(value_name = "FILE", default_value="ebsentinel.db")]
    pub db_file: String,
    #[arg(short)]
    pub test: bool
}
