use std::{env, fmt::Write, fs, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("zip.rs");
    let mut code = String::new();
    for count in 1..=16 {
        let generics_decl = (0..count)
            .map(|i| format!("M{i}: for<'short> Mat<'short>,"))
            .collect::<Vec<_>>()
            .join("");

        let generics_use = (0..count)
            .map(|i| format!("M{i},"))
            .collect::<Vec<_>>()
            .join("");

        let generics_items = (0..count)
            .map(|i| format!("<M{i} as Mat<'_>>::Item,"))
            .collect::<Vec<_>>()
            .join("");

        let slice_decl = (0..count)
            .map(|i| format!("mut slice{i}: <M{i} as Mat<'_>>::RawSlice,"))
            .collect::<Vec<_>>()
            .join("");

        let slice_def = (0..count)
            .map(|i| format!("let slice{i} = this.tuple.{i}.get_column_slice(start, j, len);"))
            .collect::<Vec<_>>()
            .join("");

        let slice_use = (0..count)
            .map(|i| format!("slice{i},"))
            .collect::<Vec<_>>()
            .join("");

        let slice_get_elem = (0..count)
            .map(|i| format!("M{i}::get_slice_elem(&mut slice{i}, i),"))
            .collect::<Vec<_>>()
            .join("");

        let do_transpose = (0..count)
            .map(|i| format!("this.tuple.{i} = this.tuple.{i}.transpose();"))
            .collect::<Vec<_>>()
            .join("");

        let do_reverse = (0..count)
            .map(|i| format!("this.tuple.{i} = this.tuple.{i}.reverse_rows();"))
            .collect::<Vec<_>>()
            .join("");

        let args_is_col_major = (0..count)
            .map(|i| format!("this.tuple.{i}.row_stride() == 1"))
            .collect::<Vec<_>>()
            .join("&&");

        let get = (0..count)
            .map(|i| format!("this.tuple.{i}.get(i, j),"))
            .collect::<Vec<_>>()
            .join("");

        let args = (0..count)
            .map(|i| format!("this.tuple.{i}"))
            .collect::<Vec<_>>()
            .join(",");

        write!(
            code,
            r#"
#[inline(always)]
unsafe fn contiguous_impl{count}<{generics_decl}>(
    len: usize,
    op: &mut impl FnMut({generics_items}),
    {slice_decl}
) {{
    for i in 0..len {{
        (*op)({slice_get_elem});
    }}
}}

impl <{generics_decl}> Zip<({generics_use})> {{
    #[inline(always)]
    pub fn for_each(self, op: impl FnMut({generics_items})) {{
        let mut this = self;
        let mut op = op;

        if this.tuple.0.row_stride().unsigned_abs() > this.tuple.0.col_stride().unsigned_abs() {{
            {do_transpose}
        }}
        if this.tuple.0.row_stride() < 0 {{
            {do_reverse}
        }}

        let nrows = this.tuple.0.nrows();
        let ncols = this.tuple.0.ncols();

        unsafe {{
            if {args_is_col_major} {{
                let start = 0;
                let len = nrows;
                for j in 0..ncols {{
                    {slice_def}

                    contiguous_impl{count}(nrows, &mut op, {slice_use});
                }}
            }} else {{
                for j in 0..ncols {{
                    for i in 0..nrows {{
                        op({get});
                    }}
                }}
            }}
        }}
    }}

    #[inline(always)]
    pub fn for_each_triangular_lower(self, diag: Diag, op: impl FnMut({generics_items})) {{
        let strict = match diag {{
            Diag::Skip => true,
            Diag::Include => false,
        }};

        let mut this = self;
        let mut op = op;

        let mut transpose = false;
        let mut reverse_rows = false;

        if this.tuple.0.row_stride().unsigned_abs() > this.tuple.0.col_stride().unsigned_abs() {{
            {do_transpose}
            transpose = true;
        }}
        if this.tuple.0.row_stride() < 0 {{
            {do_reverse}
            reverse_rows = true;
        }}

        let nrows = this.tuple.0.nrows();
        let ncols = this.tuple.0.ncols();

        let ncols = if transpose {{ ncols }} else {{ ncols.min(nrows) }};

        unsafe {{
            if {args_is_col_major} {{
                for j in 0..ncols {{
                    let (start, end) = match (transpose, reverse_rows) {{
                        (false, false) => (j + strict as usize, nrows),
                        (false, true) => (0, (nrows - (j + strict as usize))),
                        (true, false) => (0, (j + !strict as usize).min(nrows)),
                        (true, true) => (nrows - ((j + !strict as usize).min(nrows)), nrows),
                    }};

                    let len = end - start;

                    {slice_def}

                    contiguous_impl{count}(len, &mut op, {slice_use});
                }}
            }} else {{
                for j in 0..ncols {{
                    let (start, end) = match (transpose, reverse_rows) {{
                        (false, false) => (j + strict as usize, nrows),
                        (false, true) => (0, (nrows - (j + strict as usize))),
                        (true, false) => (0, (j + !strict as usize).min(nrows)),
                        (true, true) => (nrows - ((j + !strict as usize).min(nrows)), nrows),
                    }};

                    for i in start..end {{
                        op({get});
                    }}
                }}
            }}
        }}
    }}

    #[inline(always)]
    pub fn for_each_triangular_upper(self, diag: Diag, op: impl FnMut({generics_items})) {{
        let mut this = self;
        {do_transpose}
        this.for_each_triangular_lower(diag, op);
    }}

    #[inline]
    #[track_caller]
    pub fn zip_unchecked<M{count}: for<'short> Mat<'short>>(self, last: M{count}) -> Zip<({generics_use} M{count})> {{
        let this = self;
        debug_assert!((last.nrows(), last.ncols()) == (this.tuple.0.nrows(), this.tuple.0.ncols()));
        Zip {{ tuple: ({args}, last) }}
    }}
    #[inline]
    #[track_caller]
    pub fn zip<M{count}: for<'short> Mat<'short>>(self, last: M{count}) -> Zip<({generics_use} M{count})> {{
        let this = self;
        assert!((last.nrows(), last.ncols()) == (this.tuple.0.nrows(), this.tuple.0.ncols()));
        Zip {{ tuple: ({args}, last) }}
    }}
}}
"#
        )?;
    }
    fs::write(dest_path, &code)?;
    println!("cargo:rerun-if-changed=build.rs");

    Ok(())
}
