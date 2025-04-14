use core::fmt::{Debug, Write};

use faer_traits::ComplexConj;
use num_complex::Complex;

/// Obtain a TypeId for T without `T: 'static`
/// credit goes to: <https://github.com/thomcc>
#[inline(always)]
pub fn nonstatic_typeid<T: ?Sized>() -> core::any::TypeId {
	trait NonStaticAny {
		fn type_id(&self) -> core::any::TypeId
		where
			Self: 'static;
	}
	impl<T: ?Sized> NonStaticAny for core::marker::PhantomData<T> {
		#[inline(always)]
		fn type_id(&self) -> core::any::TypeId
		where
			Self: 'static,
		{
			core::any::TypeId::of::<T>()
		}
	}
	let it = core::marker::PhantomData::<T>;
	// There is no excuse for the crimes we have done here, but what jury would convict us?
	unsafe { core::mem::transmute::<&dyn NonStaticAny, &'static dyn NonStaticAny>(&it).type_id() }
}

#[inline(always)]
pub unsafe fn coerce<Src, Dst>(src: Src) -> Dst {
	crate::assert!(nonstatic_typeid::<Src>() == nonstatic_typeid::<Dst>());
	transmute(src)
}

#[inline(always)]
pub unsafe fn transmute<Src, Dst>(src: Src) -> Dst {
	if try_const! {core::mem::size_of::<Src>() !=core::mem::size_of::<Dst>() ||core::mem::align_of::<Src>() !=core::mem::align_of::<Dst>() } {
		panic!();
	} else {
		unsafe { core::mem::transmute_copy(&core::mem::ManuallyDrop::new(src)) }
	}
}

#[repr(C)]
pub struct ComplexDebug<T> {
	re: T,
	im: T,
}

#[repr(C)]
pub struct ComplexConjDebug<T> {
	re: T,
	neg_im: T,
}

#[repr(C)]
pub struct C32 {
	re: f32,
	im: f32,
}
#[repr(C)]
pub struct C64 {
	re: f64,
	im: f64,
}
#[repr(C)]
pub struct C32Conj {
	re: f32,
	neg_im: f32,
}
#[repr(C)]
pub struct C64Conj {
	re: f64,
	neg_im: f64,
}

impl<T: Debug> Debug for ComplexDebug<T> {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		if f.alternate() {
			// don't pretty print
			(unsafe { &*(self as *const _ as *const Complex<T>) }).fmt(f)
		} else {
			let re = &self.re;
			let im = &self.im;
			f.write_str("(")?;
			re.fmt(f)?;
			write!(f, " + ")?;
			im.fmt(f)?;
			f.write_char('i')?;
			f.write_str(")")
		}
	}
}
impl<T: Debug> Debug for ComplexConjDebug<T> {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		if f.alternate() {
			// don't pretty print
			(unsafe { &*(self as *const _ as *const Complex<T>) }).fmt(f)
		} else {
			let re = &self.re;
			let im = &self.neg_im;
			f.write_str("(")?;
			re.fmt(f)?;
			write!(f, " - ")?;
			im.fmt(f)?;
			f.write_char('i')?;
			f.write_str(")")
		}
	}
}

impl Debug for C32 {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		let im_sign = if self.im.is_sign_negative() { '-' } else { '+' };
		let re = self.re;
		let im = self.im.abs();
		re.fmt(f)?;
		write!(f, " {im_sign} ")?;
		im.fmt(f)?;
		f.write_char('i')
	}
}

impl Debug for C64 {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		let im_sign = if self.im.is_sign_negative() { '-' } else { '+' };
		let re = self.re;
		let im = self.im.abs();
		re.fmt(f)?;
		write!(f, " {im_sign} ")?;
		im.fmt(f)?;
		f.write_char('i')
	}
}

impl Debug for C32Conj {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		let re_sign = if self.re.is_sign_negative() { '-' } else { '+' };
		let im_sign = if self.neg_im.is_sign_negative() { '+' } else { '-' };
		let re = self.re.abs();
		let im = self.neg_im.abs();
		write!(f, "{re_sign}")?;
		re.fmt(f)?;
		write!(f, " {im_sign} ")?;
		im.fmt(f)?;
		f.write_char('i')
	}
}

impl Debug for C64Conj {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		let re_sign = if self.re.is_sign_negative() { '-' } else { '+' };
		let im_sign = if self.neg_im.is_sign_negative() { '+' } else { '-' };
		let re = self.re.abs();
		let im = self.neg_im.abs();
		write!(f, "{re_sign}")?;
		re.fmt(f)?;
		write!(f, " {im_sign} ")?;
		im.fmt(f)?;
		f.write_char('i')
	}
}

pub fn hijack_debug<T: Debug>(x: &T) -> &dyn Debug {
	if nonstatic_typeid::<T>() == nonstatic_typeid::<Complex<f32>>() {
		unsafe { &*(x as *const T as *const C32) }
	} else if nonstatic_typeid::<T>() == nonstatic_typeid::<Complex<f64>>() {
		unsafe { &*(x as *const T as *const C64) }
	} else if nonstatic_typeid::<T>() == nonstatic_typeid::<ComplexConj<f32>>() {
		unsafe { &*(x as *const T as *const C32Conj) }
	} else if nonstatic_typeid::<T>() == nonstatic_typeid::<ComplexConj<f64>>() {
		unsafe { &*(x as *const T as *const C64Conj) }
	} else {
		x
	}
}
