#![no_std]

use dlmalloc::GlobalDlmalloc;

#[global_allocator]
static ALLOCATOR: GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_panic_info: &core::panic::PanicInfo) -> ! {
	unreachable!()
}

#[no_mangle]
pub extern "C" fn mul() -> f64 {
	let m = faer::mat![[2.0, 1.0]];
	let res = m.transpose() * &m;
	*res.get(0, 0)
}
