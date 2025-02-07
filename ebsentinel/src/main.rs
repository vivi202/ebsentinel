use ebsentinel_core::{self, run_ebsentinel_ebpf};
use tokio::signal;

//Main program uses the previusly trained model to detect anomalies
#[tokio::main]
async fn main() -> anyhow::Result<()> {

    let _=run_ebsentinel_ebpf(5222);
    let ctrl_c = signal::ctrl_c();
    println!("Waiting for Ctrl-C...");
    ctrl_c.await?;
    println!("Exiting...");

    Ok(())
}
