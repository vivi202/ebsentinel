use std::time::Duration;

#[rustfmt::skip]
use proc_mon::ProcMon;
pub mod proc_mon;
pub mod process_data;
pub fn run_ebsentinel_ebpf(pid: u32)-> anyhow::Result<ProcMon>{
    env_logger::init();
    let proc_mon= ProcMon::new(pid, Duration::from_millis(100));
    Ok(proc_mon)
}