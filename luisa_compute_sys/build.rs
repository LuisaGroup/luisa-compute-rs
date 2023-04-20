use std::io;
use std::path::Path;
use std::{env, fs, path::PathBuf};

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
    set_from_env!("DX", "LUISA_COMPUTE_ENABLE_DX");
    set_from_env!("CUDA", "LUISA_COMPUTE_ENABLE_CUDA");
    set_from_env!("METAL", "LUISA_COMPUTE_ENABLE_METAL");
    set_from_env!("PYTHON", "LUISA_COMPUTE_ENABLE_PYTHON");
    set_from_env!("GUI", "LUISA_COMPUTE_ENABLE_GUI");
    config.define("LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES", "OFF");
    config.define("LUISA_COMPUTE_BUILD_TESTS", "OFF");
    config.define("LUISA_COMPUTE_COMPILED_BY_RUST_FRONTEND", "ON");
    config.define("LUISA_COMPUTE_ENABLE_DSL", "OFF");
    config.define("LUISA_COMPUTE_ENABLE_CPU", "OFF");
    config.define("LUISA_COMPUTE_ENABLE_REMOTE", "OFF");
    config.define("LUISA_COMPUTE_ENABLE_RUST", "ON");
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
            && (path.extension().unwrap() == "dll"
                || path.extension().unwrap() == "so"
                || path.extension().unwrap() == "dylib")
        {
            // let target_dir = get_output_path();
            let comps: Vec<_> = path.components().collect();
            let copy_if_different = |src, dst| {
                let p_src = Path::new(&src);
                let p_dst = Path::new(&dst);
                let should_copy = p_dst.exists();
                let check_should_copy = || -> Option<bool> {
                    let src_metadata = fs::metadata(p_src).ok()?;
                    let dst_metadata = fs::metadata(p_dst).ok()?;
                    Some(src_metadata.modified().ok()? != dst_metadata.modified().ok()?)
                };
                let should_copy = should_copy || check_should_copy().unwrap_or(true);
                if should_copy {
                    std::fs::copy(p_src, p_dst).unwrap();
                }
            };
            {
                let dest = std::path::PathBuf::from_iter(comps[..comps.len() - 6].iter())
                    .join(path.file_name().unwrap());
                copy_if_different(&path, dest);
            }
            {
                let dest = std::path::PathBuf::from_iter(comps[..comps.len() - 6].iter())
                    .join("deps")
                    .join(path.file_name().unwrap());
                dbg!(&dest);
                copy_if_different(&path, dest);
            }
            {
                let dest = std::path::PathBuf::from_iter(comps[..comps.len() - 6].iter())
                    .join("examples")
                    .join(path.file_name().unwrap());
                dbg!(&dest);
                copy_if_different(&path, dest);
            }
        }
        if path.is_dir() && path.ends_with(".data") {
            let create_dir = |path: &Path| {
                if let Err(err) = std::fs::create_dir(path) {
                    assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists, "Failed to create dir.");
                }
            };
            let comps: Vec<_> = path.components().collect();
            let target_base_dir = std::path::PathBuf::from_iter(comps[..comps.len() - 6].iter()).join("examples");

            let current_target_dir = target_base_dir.join(".data");
            create_dir(current_target_dir.as_path());
            for entry in fs::read_dir(path).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();
                if path.is_dir() {
                    let secondary_name = path.components().last().unwrap();
                    let current_target_dir = current_target_dir.join(secondary_name);
                    create_dir(current_target_dir.as_path());
                    for entry in fs::read_dir(path).unwrap() {
                        let entry = entry.unwrap();
                        let src_path = entry.path();
                        let file_name = src_path.components().last().unwrap();
                        let current_target_dir = current_target_dir.join(file_name);
                        fs::copy(src_path, current_target_dir).unwrap();
                    }
                }
            }
        }
    }
}

fn main() {
    let out_dir = cmake_build();
    copy_dlls(&out_dir);
}

