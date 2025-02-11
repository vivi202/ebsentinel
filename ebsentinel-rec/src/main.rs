use clap::Parser;
use cli::Cli;
use ebsentinel_core::run_ebsentinel_ebpf;
use ebsentinel_db::{EbsentinelDb, Syscalls};
use tokio::signal;
mod ebsentinel_db;
mod cli;



#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    println!("{:?}",cli.db_file);
    println!("{:?}",cli.test);

    let db= EbsentinelDb::new(cli.db_file);

    let mut proc_mon =run_ebsentinel_ebpf(cli.pid)?;
    let mut rx=proc_mon.run()?;
    tokio::spawn(async move {
        loop {
            let rates=rx.recv().await;
            let data= Syscalls::new(rates.unwrap());
            match cli.test {
                true => db.add_test_data(&data),
                false => db.add_train_data(&data),
            }
        }
    });
    
    let ctrl_c = signal::ctrl_c();
    println!("Waiting for Ctrl-C...");
    ctrl_c.await?;
    println!("Exiting...");

    Ok(())
}
