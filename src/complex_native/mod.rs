mod c32_conj_impl;
mod c32_impl;
mod c64_conj_impl;
mod c64_impl;

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, PartialEq)]
pub struct c32 {
    pub re: f32,
    pub im: f32,
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, PartialEq)]
pub struct c64 {
    pub re: f64,
    pub im: f64,
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, PartialEq)]
pub struct c32conj {
    pub re: f32,
    pub neg_im: f32,
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, PartialEq)]
pub struct c64conj {
    pub re: f64,
    pub neg_im: f64,
}
