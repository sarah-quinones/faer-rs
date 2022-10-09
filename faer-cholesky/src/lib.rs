pub mod ldlt;

#[track_caller]
#[inline]
unsafe fn unreachable_unchecked() -> ! {
    debug_assert!(false);
    core::hint::unreachable_unchecked()
}
