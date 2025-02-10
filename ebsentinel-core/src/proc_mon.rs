use std::time::Duration;

use aya::{
    maps::{PerCpuArray, PerCpuValues},
    programs::BtfTracePoint,
    util::nr_cpus,
    Btf, Ebpf,
};
use ebsentinel_common::MAX_SYSCALLS;
use log::{debug, warn};
use tokio::{
    sync::mpsc::{unbounded_channel, UnboundedReceiver},
    time::sleep,
};

use crate::process_data::{DataProcessor, Differentiator, Normalizer};
pub struct ProcMon {
    //Monitored Pid
    monitored_pid: u32,
    polling_rate: Duration,
    ebpf: Ebpf,
}

impl ProcMon {
    pub fn new(monitored_pid: u32, polling_rate: Duration) -> Self {
        let ebpf = Self::load_epbf().unwrap();
        let proc_mon = Self {
            monitored_pid,
            polling_rate,
            ebpf,
        };
        proc_mon
    }

    pub fn run(&mut self) -> anyhow::Result<UnboundedReceiver<Vec<f32>>> {
        let (tx, rx) = unbounded_channel();
        let mut monitored_pid: PerCpuArray<&mut aya::maps::MapData, u32> =
            PerCpuArray::try_from(self.ebpf.map_mut("MONITORED_PID").unwrap())?;

        let nr_cpus = nr_cpus().map_err(|(_, error)| error)?;

        monitored_pid
            .set(
                0,
                PerCpuValues::try_from(vec![self.monitored_pid; nr_cpus])?,
                0,
            )
            .unwrap();

        let map = self.ebpf.take_map("SYSCALLS_COUNTERS").unwrap();
        let syscall_map: PerCpuArray<aya::maps::MapData, u64> =
            PerCpuArray::try_from(map)?;

        let polling_rate=self.polling_rate.clone();
        
        let mut differentiator= Differentiator::new(&self.polling_rate);

        tokio::spawn(async move {
            let mut prev = Vec::new();
            let mut syscall_counts = vec![0; MAX_SYSCALLS as usize];
            loop {

                for i in 0..MAX_SYSCALLS as usize{
                    //Aggregate Syscalls counts from all cpus
                    syscall_counts[i]=0;
                    if let Ok(counts) = syscall_map.get(&(i as u32), 0) {
                        for cpu_val in counts.iter() {
                            syscall_counts[i] += cpu_val;
                        }
                    }
                }
                
                if syscall_counts != prev{
                                  //Compute derivative
                let rates=differentiator.process(&syscall_counts);
                if let Ok(rates) = rates {
                    let norm=Normalizer.process(&rates).unwrap();
                    tx.send(norm).unwrap();
                }
                prev= syscall_counts.clone();  
                }

                sleep(polling_rate).await
            }
        });
        
        Ok(rx)
    }

    //Load ebsentinel-ebpf program to kernel vm.
    pub fn load_epbf() -> anyhow::Result<Ebpf> {
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
        program.load("sys_enter", &btf)?;
        program.attach()?;

        Ok(ebpf)
    }
}
