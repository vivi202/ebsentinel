use std::{time::Duration, vec};

use aya::{maps::{PerCpuArray, PerCpuValues}, programs::BtfTracePoint, util::nr_cpus, Btf};
use ebsentinel_common::MAX_SYSCALLS;
#[rustfmt::skip]
use log::{debug, warn};
use tokio::{sync::mpsc::UnboundedReceiver, time::sleep};
use tokio::sync::mpsc::unbounded_channel;
pub fn run_ebsentinel_ebpf(pid: u32)-> anyhow::Result<UnboundedReceiver<Vec<f32>>>{
    env_logger::init();

    // Bump the memlock rlimit. This is needed for older kernels that don't use the
    // new memcg based accounting, see https://lwn.net/Articles/837122/
    let rlim = libc::rlimit {
        rlim_cur: libc::RLIM_INFINITY,
        rlim_max: libc::RLIM_INFINITY,
    };
    let ret = unsafe { libc::setrlimit(libc::RLIMIT_MEMLOCK, &rlim) };
    if ret != 0 {
        debug!("remove limit on locked memory failed, ret is: {}", ret);
    }

    //include ebsentinel-ebpf object file as raw bytes 
    let mut ebpf = aya::Ebpf::load(aya::include_bytes_aligned!(concat!(
        env!("OUT_DIR"),
        "/ebsentinel"
    )))?;

    if let Err(e) = aya_log::EbpfLogger::init(&mut ebpf) {
        // This can happen if you remove all log statements from your eBPF program.
        warn!("failed to initialize eBPF logger: {}", e);
    }

    let btf = Btf::from_sys_fs()?;

    let program: &mut BtfTracePoint = ebpf.program_mut("ebsentinel").unwrap().try_into()?;
    program.load("sys_enter",&btf)?;
    program.attach()?;

    let mut monitored_pid: PerCpuArray<&mut aya::maps::MapData, u32> = PerCpuArray::try_from(ebpf.map_mut("MONITORED_PID").unwrap())?;
    
    let nr_cpus = nr_cpus().map_err(|(_, error)| error)?;
    monitored_pid.set(0, PerCpuValues::try_from(vec![pid; nr_cpus])?, 0)?;
    let (tx,rx) = unbounded_channel();
    tokio::spawn(async move {
        let syscall_counts: PerCpuArray<&mut aya::maps::MapData, u64> = PerCpuArray::try_from(ebpf.map_mut("SYSCALLS_COUNTERS").unwrap()).unwrap();
        let mut prev = vec![0.0;MAX_SYSCALLS as usize];
        let interval_ms: u64= 100;
        let mut last_rates=Vec::new();
        loop {
            let mut values =vec![0;MAX_SYSCALLS as usize];
            for i in 0..MAX_SYSCALLS {
                //Get count from all cpus
                if let Ok(counts)= syscall_counts.get(&(i as u32), 0){
                    for cpu_val in counts.iter() {
                        values[i as usize]+=cpu_val;
                    }
                }
            }
            
            //Compute derivative
            let mut rates = vec![0.0; MAX_SYSCALLS as usize];
            let mut max= rates[0];
            for i in 0..MAX_SYSCALLS as usize {
                rates[i] = (values[i] as f32 - prev[i]) / (interval_ms as f32 / 1000.0);
                prev[i] = values[i] as f32;
                max=max.max(rates[i])
            }

            if last_rates != rates{
                let normalized=rates.iter().map(|r| match max {
                    0.0 => 0.0,
                    max => r/max
                }).collect();
                println!("values {:?}",&normalized);

                tx.send(normalized).unwrap();
            }

            last_rates=rates.clone();

            sleep(Duration::from_millis(interval_ms)).await
        }
    });
    Ok(rx)
}