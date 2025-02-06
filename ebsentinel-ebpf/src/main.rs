#![no_std]
#![no_main]



use aya_ebpf::{ cty::c_long, macros::{btf_tracepoint, map}, maps::PerCpuArray, programs::BtfTracePointContext, EbpfContext};
use ebsentinel_common::MAX_SYSCALLS;

#[map(name="SYSCALLS_COUNTERS")]
static mut SYSCALLS_COUNTER: PerCpuArray<c_long> = PerCpuArray::with_max_entries(MAX_SYSCALLS, 0);

#[map(name="MONITORED_PID")]
static mut MONITORED_PID: PerCpuArray<u32> = PerCpuArray::with_max_entries(1, 0);

#[btf_tracepoint(function = "sys_enter")]
pub fn ebsentinel(ctx: BtfTracePointContext) -> i32 {
    match try_sys_enter(ctx) {
        Ok(ret) => ret,
        Err(ret) => ret,
    }
}

fn try_sys_enter(ctx: BtfTracePointContext) -> Result<i32, i32> {
        
        let pid= ctx.pid();
        let syscall_id: c_long = unsafe { ctx.arg(1) }; 
        if let Some(monitored_pid) = unsafe { MONITORED_PID.get(0) }{
            if pid != *monitored_pid {
                return Ok(0);
            }
            if let Some(syscall_count) = unsafe {SYSCALLS_COUNTER.get_ptr_mut(syscall_id as u32)}{
                unsafe {
                    *syscall_count+=1; 
                }

            };
        }
    Ok(0)
}

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
