use std::path::PathBuf;

use bindgen;
fn generate_bindings() {
    let bindings = bindgen::Builder::default()
        .header("./LuisaCompute/src/api/language.h")
        .clang_arg("-I./LuisaCompute/src/")
        .prepend_enum_name(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");
    bindings
        .write_to_file("src/binding.rs")
        .expect("Couldn't write bindings!");
}
fn cmake_build() -> PathBuf {
    let enabled_cuda = cfg!(feature = "cuda");
    let enabled_dx = cfg!(feature = "dx");
    let enabled_ispc = cfg!(feature = "ispc");
    // let enabled_vk = cfg!(feature = "vk");
    // let enabled_metal = cfg!(feature = "metal");
    let enabled_llvm = cfg!(feature = "llvm");
    let mut config = cmake::Config::new("./LuisaCompute");
    let map_bool_to_str = |b: bool| if b { "ON" } else { "OFF" };
    config.define("-DLUISA_COMPUTE_ENABLE_DX", map_bool_to_str(enabled_dx));
    config.define("-DLUISA_COMPUTE_ENABLE_CUDA", map_bool_to_str(enabled_cuda));
    config.define("-DLUISA_COMPUTE_ENABLE_LLVM", map_bool_to_str(enabled_llvm));
    config.define("-DLUISA_COMPUTE_ENABLE_ISPC", map_bool_to_str(enabled_ispc));
    config.define("-DLUISA_COMPUTE_ENABLE_GUI", "OFF");
    config.define("-DLUISA_COMPUTE_BUILD_TESTS", "OFF");
    config.define("CMAKE_BUILD_TYPE", "Release");
    config.generator("Ninja");
    // if cfg!(target_os="windows") {
    //     config.build_arg("--config");
    //     config.build_arg("Release");
    // }
    
    config.build()
}
fn copy_dlls(out_dir: &PathBuf) {
    let mut out_dir = out_dir.clone();
    out_dir.push(&"/bin");

    for entry in std::fs::read_dir(out_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().is_some()
            && (path.extension().unwrap() == "dll" || path.extension().unwrap() == "so")
        {
            // let target_dir = get_output_path();
            let comps: Vec<_> = path.components().collect();
            let dest = std::path::PathBuf::from_iter(comps[..comps.len() - 6].iter())
                .join(path.file_name().unwrap());
            if std::path::Path::new(&dest).exists() {
                continue;
            }
            std::fs::copy(path, dest).unwrap();
        }
    }
}
fn main() {
    generate_bindings();
    let out_dir = cmake_build();
    copy_dlls(&out_dir);
}
