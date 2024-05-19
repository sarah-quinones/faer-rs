#![no_std]
#![no_main]
#![feature(alloc_error_handler)]
#![feature(panic_info_message)]
#![feature(lang_items)]

use core::alloc::{GlobalAlloc, Layout};

/// The global allocator type.
#[derive(Default)]
pub struct Allocator;

unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let mut mem: *mut libc::c_void = core::ptr::null_mut();
        libc::posix_memalign(&mut mem, layout.align(), layout.size());
        assert!(!mem.is_null());
        mem as *mut u8
    }
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        libc::free(ptr as *mut libc::c_void);
    }
}

#[panic_handler]
fn panic(panic_info: &core::panic::PanicInfo) -> ! {
    if let Some(message) = panic_info.message() {
        libc_print::libc_eprintln!("{message}");
    }

    unsafe { libc::exit(1) };
}

/// If there is an out of memory error, just panic.
#[alloc_error_handler]
fn my_allocator_error(_layout: Layout) -> ! {
    panic!("out of memory");
}

#[cfg(not(tests))]
#[lang = "eh_personality"]
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}

/// The static global allocator.
#[global_allocator]
static GLOBAL_ALLOCATOR: Allocator = Allocator;

#[no_mangle]
pub extern "C" fn main(argc: i32, argv: *const *const u8) -> i32 {
    let _ = (&argc, &argv);

    let m = faer::mat![[2.0, 1.0]];
    libc_print::libc_println!("{m:?}");
    libc_print::libc_println!("{:?}", m.transpose() * &m);
    0
}
