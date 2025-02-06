use ebsentinel_core::{self, run};
use tokio::signal;
//Main program uses the previusly trained model to detect anomalies
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _=run();
    let ctrl_c = signal::ctrl_c();
    println!("Waiting for Ctrl-C...");
    ctrl_c.await?;
    println!("Exiting...");

    Ok(())
}
