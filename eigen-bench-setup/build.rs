fn main() {
	cc::Build::new().cpp(true).flag("-march=native").file("./eigen.cpp").compile("eigen");
	println!("cargo::rerun-if-changed=./eigen.cpp");
}
