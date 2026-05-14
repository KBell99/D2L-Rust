fn main() {
    println!(
        "cargo:rustc-link-arg-bins=-Wl,--push-state,--no-as-needed,-ltorch_cuda,-ltorch,--pop-state"
    );
    println!("cargo:rustc-link-search=native=/usr/local/libtorch/lib");
}
