use std::{env, fs, path::PathBuf};

use bindgen::{self, CargoCallbacks};
fn generate_bindings() {
    #[derive(Debug)]
    struct ParseCallback {}
    fn to_upper_snake_case(s: &str) -> String {
        let mut result = String::new();
        let mut last: Option<char> = None;
        for c in s.chars() {
            if let Some(last) = last {
                if c.is_uppercase() {
                    if last.is_lowercase() || last.is_numeric() {
                        result.push('_');
                    }
                }
            }
            result.push(c.to_ascii_uppercase());
            last = Some(c);
        }
        result
    }
    impl bindgen::callbacks::ParseCallbacks for ParseCallback {
        fn include_file(&self, _filename: &str) {
            let cb = CargoCallbacks {};
            cb.include_file(_filename);
        }
        fn enum_variant_name(
            &self,
            _enum_name: Option<&str>,
            original_variant_name: &str,
            _variant_value: bindgen::callbacks::EnumVariantValue,
        ) -> Option<String> {
            // let enum_name = enum_name?;
            // if !enum_name.starts_with("LC") {
            //     return None;
            // }
            if original_variant_name.starts_with("LC_OP_") {
                let mut name = original_variant_name.to_string();
                name = name.replace("LC_OP_", "");
                return Some(to_upper_snake_case(&name));
            }
            if original_variant_name.starts_with("LC_") {
                let mut name = original_variant_name.to_string();
                name = name.replace("LC_", "");
                return Some(to_upper_snake_case(&name));
            }
            None
        }
    }
    let bindings = bindgen::Builder::default()
        .header("./LuisaCompute/src/api/runtime.h")
        .header("./LuisaCompute/src/api/logging.h")
        .clang_arg("-I./LuisaCompute/src/")
        .clang_arg("-I./LuisaCompute/src/ir/")
        .clang_arg("-I./LuisaCompute/src/ir/luisa-compute-ir")
        .prepend_enum_name(false)
        .newtype_enum("LC.*")
        .parse_callbacks(Box::new(ParseCallback {}))
        .generate()
        .expect("Unable to generate bindings");
    bindings
        .write_to_file("src/binding.rs")
        .expect("Couldn't write bindings!");
}
fn cmake_build() -> PathBuf {
    let mut config = cmake::Config::new("./LuisaCompute");
    macro_rules! set_from_env {
        ($feature:literal, $opt:literal) => {
            let var = format!("CARGO_FEATURE_{}", $feature);
            println!("cargo:rerun-if-env-changed={}", var);
            if let Ok(_) = env::var(var) {
                config.define($opt, "ON");
            } else {
                config.define($opt, "OFF");
            }
        };
        ($opt:literal) => {
            println!("cargo:rerun-if-env-changed={}", $opt);
            if let Ok(v) = env::var($opt) {
                println!("{}={}", $opt, v);
                config.define($opt, v);
            }
        };
    }
    set_from_env!("dx", "LUISA_COMPUTE_ENABLE_DX");
    set_from_env!("cuda", "LUISA_COMPUTE_ENABLE_CUDA");
    set_from_env!("llvm", "LUISA_COMPUTE_ENABLE_LLVM");
    set_from_env!("ispc", "LUISA_COMPUTE_ENABLE_ISPC");
    set_from_env!("vulkan", "LUISA_COMPUTE_ENABLE_VULKAN");
    set_from_env!("metal", "LUISA_COMPUTE_ENABLE_METAL");
    set_from_env!("python", "LUISA_COMPUTE_ENABLE_PYTHON");
    set_from_env!("gui", "LUISA_COMPUTE_ENABLE_GUI");
    config.define("LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES", "OFF");
    config.define("LUISA_COMPUTE_BUILD_TESTS", "OFF");
    config.define("LUISA_COMPUTE_RUST", "ON");
    config.define("CMAKE_BUILD_TYPE", "Release");
    // set compiler based on env
    println!("cargo:rerun-if-env-changed=CC");
    if let Ok(v) = env::var("CC") {
        println!("CC={}", v);
        config.define("CMAKE_C_COMPILER", v);
    }
    println!("cargo:rerun-if-env-changed=CXX");
    if let Ok(v) = env::var("CXX") {
        println!("CXX={}", v);
        config.define("CMAKE_CXX_COMPILER", v);
    }
    config.profile("Release");
    config.generator("Ninja");
    config.no_build_target(true);

    // if cfg!(target_os="windows") {
    //     config.build_arg("--config");
    //     config.build_arg("Release");
    // }

    config.build()
}
fn copy_dlls(out_dir: &PathBuf) {
    let mut out_dir = out_dir.clone();
    out_dir.push(&"build/bin");
    dbg!(&out_dir);
    for entry in std::fs::read_dir(out_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().is_some()
            && (path.extension().unwrap() == "dll" || path.extension().unwrap() == "so")
        {
            // let target_dir = get_output_path();
            let comps: Vec<_> = path.components().collect();
            {
                let dest = std::path::PathBuf::from_iter(comps[..comps.len() - 6].iter())
                    .join(path.file_name().unwrap());
                if !std::path::Path::new(&dest).exists() {
                    std::fs::copy(&path, dest).unwrap();
                }
            }
            {
                let dest = std::path::PathBuf::from_iter(comps[..comps.len() - 6].iter())
                    .join("deps")
                    .join(path.file_name().unwrap());
                dbg!(&dest);
                if !std::path::Path::new(&dest).exists() {
                    std::fs::copy(&path, dest).unwrap();
                }
            }
        }
    }
}
fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE__cpp");
    if cfg!(feature = "_cpp") {
        let out_dir = cmake_build();
        generate_bindings();
        // dbg!(&out_dir);
        println!(
            "cargo:rustc-link-search=native={}/build/bin/",
            out_dir.to_str().unwrap()
        );
        println!(
            "cargo:rustc-link-search=native={}/build/lib/",
            out_dir.to_str().unwrap()
        );
        println!("cargo:rustc-link-lib=dylib=luisa-compute-api");
        copy_dlls(&out_dir);
    }
}
