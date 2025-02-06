use std::{time::Duration, vec};

use aya::{maps::{PerCpuArray, PerCpuValues}, programs::BtfTracePoint, util::nr_cpus, Btf};
use ebsentinel_common::MAX_SYSCALLS;
#[rustfmt::skip]
use log::{debug, warn};
use tokio::{signal, time::sleep};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
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

    // This will include your eBPF object file as raw bytes at compile-time and load it at
    // runtime. This approach is recommended for most real-world use cases. If you would
    // like to specify the eBPF program at runtime rather than at compile-time, you can
    // reach for `Bpf::load_file` instead.
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
    monitored_pid.set(0, PerCpuValues::try_from(vec![70370u32; nr_cpus])?, 0)?;

    tokio::spawn(async move {
        let syscall_counts: PerCpuArray<&mut aya::maps::MapData, u64> = PerCpuArray::try_from(ebpf.map_mut("SYSCALLS_COUNTERS").unwrap()).unwrap();

        loop {
            let mut values =[0;MAX_SYSCALLS as usize];
            for i in 0..MAX_SYSCALLS {
                if let Ok(counts)= syscall_counts.get(&(i as u32), 0){
                    for cpu_val in counts.iter() {
                        values[i as usize]+=cpu_val;
                    }
                }
            }
            println!("values: {:?}",values);
            sleep(Duration::from_millis(50)).await
        }
    });

    let ctrl_c = signal::ctrl_c();
    println!("Waiting for Ctrl-C...");
    ctrl_c.await?;
    println!("Exiting...");

    Ok(())
}
