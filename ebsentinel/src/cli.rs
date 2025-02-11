use clap::Parser;

#[derive(Parser)]
pub struct Cli{
    #[arg(value_name = "PID")]
    pub pid: u32,
    #[arg(value_name = "THRESH")]
    pub threshold: f32,
}