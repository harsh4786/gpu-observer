fn main() {
    println!("cargo:rustc-check-cfg=cfg(bpf_target_arch, values(\"aarch64\"))");
    println!("cargo:rustc-cfg=bpf_target_arch=\"aarch64\"");
}
