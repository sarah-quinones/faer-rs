use std::env;
use std::path::Path;
fn main() -> Result<(), Box<dyn core::error::Error>> {
	let var = "FAER_FFI";
	let header = "faer.h";
	println!("cargo::rerun-if-changed={header}");
	println!("cargo::rerun-if-env-changed={var}");
	println!("cargo::rerun-if-env-changed=CI");
	if env::var_os(var).is_some() {
		let manifest = env::var_os("CARGO_MANIFEST_DIR").unwrap();
		let manifest = Path::new(&manifest);
		let header = &manifest.join(header);
		let changed = cbindgen::generate(manifest)?.write_to_file(header);
		if env::var_os("CI").is_some() {
			assert!(!changed, "faer-ffi bindings are not up to date");
		}
	}
	Ok(())
}
