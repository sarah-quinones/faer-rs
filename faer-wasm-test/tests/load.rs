use wasmtime::*;

struct MyState {
    name: String,
    count: usize,
}

#[test]
fn main() -> Result<()> {
    let output = std::process::Command::new("cargo")
        .args(["build", "--target", "wasm32-unknown-unknown"])
        .output()
        .expect("Failed to execute cargo command");

    if !output.status.success() {
        eprintln!("Cargo build failed: {}", String::from_utf8_lossy(&output.stderr));
        panic!("Failed to build WASM module");
    }

    let wasm_path = "target/wasm32-unknown-unknown/debug/faer_wasm_test.wasm";

    println!("Compiling module...");
    let engine = Engine::default();
    let module = Module::from_file(&engine, wasm_path)?;

    println!("Initializing...");
    let mut store = Store::new(
        &engine,
        MyState {
            name: "faer test".to_string(),
            count: 0,
        },
    );

    println!("Instantiating module...");
    let imports = [];
    let instance = Instance::new(&mut store, &module, &imports)?;

    println!("Extracting export...");
    let run = instance.get_typed_func::<(), f64>(&mut store, "mul")?;

    println!("Calling export...");
    let result = run.call(&mut store, ())?;
    assert_eq!(result, 4.0);

    Ok(())
}