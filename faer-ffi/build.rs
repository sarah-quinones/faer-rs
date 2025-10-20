use std::path::Path;

type Error = Box<dyn std::error::Error>;

fn main() -> Result<(), Error> {
	let dir = &*std::env::var("CARGO_MANIFEST_DIR").unwrap();

	let dir = Path::new(dir);

	let config = cbindgen::Config::from_file(dir.join("cbindgen.toml"))?;

	let bindings = cbindgen::generate_with_config(dir, config)?;

	bindings.write_to_file(dir.join("faer.h"));

	Ok(())
}
