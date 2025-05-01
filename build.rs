// build.rs
fn main() {
    // This tells Cargo to link both cdylib and binaries
    pyo3_build_config::add_extension_module_link_args();
}
