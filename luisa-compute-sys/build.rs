use bindgen;

fn main() {
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