fn main() {
    // Tell Cargo that if the given file changes, to rerun this build script.
    println!("cargo:rerun-if-changed=eigen.cpp");
    // Use the `cc` crate to build a C file and statically link it.
    #[cfg(feature = "eigen")]
    cc::Build::new()
        .cpp(true)
        .flag("-march=native")
        .flag("-fopenmp")
        .file("eigen.cpp")
        .compile("eigen");
}
