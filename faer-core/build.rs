use std::fmt::Write;
use std::path::Path;
use std::{env, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("zip.rs");
    let mut code = String::new();
    for count in 1..=16 {
        let generics_decl = (0..count)
            .map(|i| format!("M{i}: for<'short> CwiseMat<'short>"))
            .collect::<Vec<_>>()
            .join(",");
        let generics_col_decl = (0..count)
            .map(|i| format!("M{i}: for<'short> CwiseCol<'short>"))
            .collect::<Vec<_>>()
            .join(",");
        let generics_row_decl = (0..count)
            .map(|i| format!("M{i}: for<'short> CwiseRow<'short>"))
            .collect::<Vec<_>>()
            .join(",");

        let generics_use = (0..count)
            .map(|i| format!("M{i}"))
            .collect::<Vec<_>>()
            .join(",");

        let generics_items = (0..count)
            .map(|i| format!("<M{i} as CwiseMat<'_>>::Item"))
            .collect::<Vec<_>>()
            .join(",");
        let generics_col_items = (0..count)
            .map(|i| format!("<M{i} as CwiseCol<'_>>::Item"))
            .collect::<Vec<_>>()
            .join(",");
        let generics_row_items = (0..count)
            .map(|i| format!("<M{i} as CwiseRow<'_>>::Item"))
            .collect::<Vec<_>>()
            .join(",");

        let args_is_col_major = (0..count)
            .map(|i| format!("this.tuple.{i}.is_col_major()"))
            .collect::<Vec<_>>()
            .join("&&");
        let args_is_row_major = (0..count)
            .map(|i| format!("this.tuple.{i}.is_row_major()"))
            .collect::<Vec<_>>()
            .join("&&");
        let args_is_contiguous = (0..count)
            .map(|i| format!("this.tuple.{i}.is_contiguous()"))
            .collect::<Vec<_>>()
            .join("&&");

        let args_get_unchecked = (0..count)
            .map(|i| format!("this.tuple.{i}.get_unchecked(i, j)"))
            .collect::<Vec<_>>()
            .join(",");
        let args_get_col_major_unchecked = (0..count)
            .map(|i| format!("this.tuple.{i}.get_col_major_unchecked(i, j)"))
            .collect::<Vec<_>>()
            .join(",");
        let args_get_row_major_unchecked = (0..count)
            .map(|i| format!("this.tuple.{i}.get_row_major_unchecked(i, j)"))
            .collect::<Vec<_>>()
            .join(",");
        let args_get_contiguous_unchecked = (0..count)
            .map(|i| format!("this.tuple.{i}.get_contiguous_unchecked(i)"))
            .collect::<Vec<_>>()
            .join(",");
        let args_get_col_unchecked = (0..count)
            .map(|i| format!("this.tuple.{i}.get_unchecked(i)"))
            .collect::<Vec<_>>()
            .join(",");
        let args_get_row_unchecked = (0..count)
            .map(|i| format!("this.tuple.{i}.get_unchecked(i)"))
            .collect::<Vec<_>>()
            .join(",");

        let args = (0..count)
            .map(|i| format!("this.tuple.{i}"))
            .collect::<Vec<_>>()
            .join(",");
        let args_transposed = (0..count)
            .map(|i| format!("this.tuple.{i}.transpose()"))
            .collect::<Vec<_>>()
            .join(",");

        write!(
            code,
            r#"
impl<{generics_decl}> ZipMat<({generics_use},)> {{
    #[inline]
    pub fn for_each(self, op: impl FnMut({generics_items})) {{
        let mut this = self;
        let mut op = op;

        let first = &this.tuple.0;
        let nrows = first.nrows();
        let ncols = first.ncols();

        unsafe {{
            if {args_is_col_major} {{
                for j in 0..ncols {{
                    for i in 0..nrows {{
                        op({args_get_col_major_unchecked});
                    }}
                }}
            }} else if {args_is_row_major} {{
                for i in 0..nrows {{
                    for j in 0..ncols {{
                        op({args_get_row_major_unchecked});
                    }}
                }}
            }} else {{
                for j in 0..ncols {{
                    for i in 0..nrows {{
                        op({args_get_unchecked});
                    }}
                }}
            }}
        }}
    }}

    #[inline]
    pub fn for_each_triangular_lower(self, strict: bool, op: impl FnMut({generics_items})) {{
        let mut this = self;
        let mut op = op;

        let first = &this.tuple.0;
        let nrows = first.nrows();
        let ncols = first.ncols();

        unsafe {{
            if {args_is_col_major} {{
                for j in 0..ncols {{
                    for i in j + strict as usize..nrows {{
                        op({args_get_col_major_unchecked});
                    }}
                }}
            }} else if {args_is_row_major} {{
                for i in strict as usize..nrows {{
                    for j in 0..i + !strict as usize {{
                        op({args_get_row_major_unchecked});
                    }}
                }}
            }} else {{
                for j in 0..ncols {{
                    for i in j + strict as usize..nrows {{
                        op({args_get_unchecked});
                    }}
                }}
            }}
        }}
    }}

    #[inline]
    pub fn for_each_triangular_upper(self, strict: bool, op: impl FnMut({generics_items})) {{
        let this = self;
        let this = Self {{ tuple: ({args_transposed},) }};
        this.for_each_triangular_lower(strict, op);
    }}

    #[inline]
    pub fn zip_unchecked<M{count}: for<'short> CwiseMat<'short>>(self, last: M{count}) -> ZipMat<({generics_use}, M{count})> {{
        let this = self;
        fancy_debug_assert!((last.nrows(), last.ncols()) == (this.tuple.0.nrows(), this.tuple.0.ncols()));
        ZipMat {{ tuple: ({args}, last) }}
    }}

    #[inline]
    pub fn zip<M{count}: for<'short> CwiseMat<'short>>(self, last: M{count}) -> ZipMat<({generics_use}, M{count})> {{
        let this = self;
        fancy_assert!((last.nrows(), last.ncols()) == (this.tuple.0.nrows(), this.tuple.0.ncols()));
        ZipMat {{ tuple: ({args}, last) }}
    }}
}}

impl<{generics_col_decl}> ZipCol<({generics_use},)> {{
    #[inline]
    pub fn for_each(self, op: impl FnMut({generics_col_items})) {{
        let mut this = self;
        let mut op = op;

        let first = &this.tuple.0;
        let nrows = first.nrows();

        unsafe {{
            if {args_is_contiguous} {{
                for i in 0..nrows {{
                    op({args_get_contiguous_unchecked});
                }}
            }} else {{
                for i in 0..nrows {{
                    op({args_get_col_unchecked});
                }}
            }}
        }}
    }}

    #[inline]
    pub fn zip_unchecked<M{count}: for<'short> CwiseCol<'short>>(self, last: M{count}) -> ZipCol<({generics_use}, M{count})> {{
        let this = self;
        fancy_debug_assert!(last.nrows() == this.tuple.0.nrows());
        ZipCol {{ tuple: ({args}, last) }}
    }}

    #[inline]
    pub fn zip<M{count}: for<'short> CwiseCol<'short>>(self, last: M{count}) -> ZipCol<({generics_use}, M{count})> {{
        let this = self;
        fancy_assert!(last.nrows() == this.tuple.0.nrows());
        ZipCol {{ tuple: ({args}, last) }}
    }}
}}

impl<{generics_row_decl}> ZipRow<({generics_use},)> {{
    #[inline]
    pub fn for_each(self, op: impl FnMut({generics_row_items})) {{
        let mut this = self;
        let mut op = op;

        let first = &this.tuple.0;
        let ncols = first.ncols();

        unsafe {{
            if {args_is_contiguous} {{
                for i in 0..ncols {{
                    op({args_get_contiguous_unchecked});
                }}
            }} else {{
                for i in 0..ncols {{
                    op({args_get_row_unchecked});
                }}
            }}
        }}
    }}

    #[inline]
    pub fn zip_unchecked<M{count}: for<'short> CwiseRow<'short>>(self, last: M{count}) -> ZipRow<({generics_use}, M{count})> {{
        let this = self;
        fancy_debug_assert!(last.ncols() == this.tuple.0.ncols());
        ZipRow {{ tuple: ({args}, last) }}
    }}

    #[inline]
    pub fn zip<M{count}: for<'short> CwiseRow<'short>>(self, last: M{count}) -> ZipRow<({generics_use}, M{count})> {{
        let this = self;
        fancy_assert!(last.ncols() == this.tuple.0.ncols());
        ZipRow {{ tuple: ({args}, last) }}
    }}
}}
"#
        )?;
    }
    fs::write(&dest_path, &code)?;
    println!("cargo:rerun-if-changed=build.rs");

    Ok(())
}
