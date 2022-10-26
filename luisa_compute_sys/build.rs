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
fn getenv_unwrap(v: &str) -> String {
    match env::var(v) {
        Ok(s) => s,
        Err(..) => panic!("environment variable `{}` not defined", v),
    }
}
fn getenv_option(v: &str) -> Option<String> {
    match env::var(v) {
        Ok(s) => match s.as_str() {
            "ON" | "1" => Some("ON".to_string()),
            "OFF" | "0" => Some("OFF".to_string()),
            _ => None,
        },
        Err(..) => None,
    }
}
fn cmake_build() -> PathBuf {
    // let enabled_cuda = cfg!(feature = "cuda");
    // let enabled_dx = cfg!(feature = "dx");
    // let enabled_ispc = cfg!(feature = "ispc");
    // let enabled_vk = cfg!(feature = "vk");
    // // let enabled_metal = cfg!(feature = "metal");
    // let enabled_llvm = cfg!(feature = "llvm");
    // let enable_python = cfg!(feature = "python");
    let mut config = cmake::Config::new("./LuisaCompute");
    macro_rules! set_from_env {
        ($opt:literal) => {
            println!("cargo:rerun-if-env-changed={}", $opt);
            if let Some(v) = getenv_option($opt) {
                config.define($opt, v);
            }
        };
    }
    set_from_env!("LUISA_COMPUTE_ENABLE_DX");
    set_from_env!("LUISA_COMPUTE_ENABLE_CUDA");
    set_from_env!("LUISA_COMPUTE_ENABLE_LLVM");
    set_from_env!("LUISA_COMPUTE_ENABLE_ISPC");
    set_from_env!("LUISA_COMPUTE_ENABLE_VULKAN");
    set_from_env!("LUISA_COMPUTE_ENABLE_PYTHON");

    config.define("LUISA_COMPUTE_ENABLE_GUI", "OFF");
    config.define("LUISA_COMPUTE_BUILD_TESTS", "OFF");
    config.define("CMAKE_BUILD_TYPE", "Release");
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
   
    let out_dir = cmake_build();
    // generate_bindings();
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
