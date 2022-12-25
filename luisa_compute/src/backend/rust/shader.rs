use std::{
    env::current_exe,
    ffi::OsStr,
    fs::canonicalize,
    mem::transmute,
    path::PathBuf,
    process::{Command, Stdio},
};

fn canonicalize_and_fix_windows_path(path: PathBuf) -> std::io::Result<PathBuf> {
    let path = canonicalize(path)?;
    let mut s: String = path.to_str().unwrap().into();
    if s.starts_with(r"\\?\") {
        // s(r"\\?\".len());
        s = s[r"\\?\".len()..].into();
    }
    Ok(PathBuf::from(s))
}

pub(super) fn compile(source: String) -> std::io::Result<PathBuf> {
    let target = super::sha256(&source);
    let self_path = current_exe().map_err(|e| {
        eprintln!("current_exe() failed");
        e
    })?;
    let self_path: PathBuf = canonicalize_and_fix_windows_path(self_path)?
        .parent()
        .unwrap()
        .into();
    let mut build_dir = self_path.clone();
    build_dir.push(".jit/");
    build_dir.push(format!("{}/", target));
    // build_dir.push("build/");
    if !build_dir.exists() {
        std::fs::create_dir_all(&build_dir).map_err(|e| {
            eprintln!("fs::create_dir_all({}) failed", build_dir.display());
            e
        })?;
    }

    let target_lib = if cfg!(target_os = "windows") {
        format!("{}.dll", target)
    } else {
        format!("{}.so", target)
    };
    let lib_path = PathBuf::from(format!("{}/{}", build_dir.display(), target_lib));
    if lib_path.exists() {
        log::info!("loading cached kernel {}", target_lib);
        return Ok(lib_path);
    }
    let source_file = format!("{}/{}.rs", build_dir.display(), target);
    std::fs::write(&source_file, source).map_err(|e| {
        eprintln!("fs::write({}) failed", source_file);
        e
    })?;
    log::info!("compiling kernel {}", source_file);

    let mut args: Vec<&str> = vec![];
    args.extend(&[
        "--crate-type",
        "cdylib",
        "-C",
        "debuginfo=0",
        "-C",
        "overflow-checks=no",
        "-C",
        "target-cpu=native",
        "-C",
        "opt-level=3",
        "-o",
        &target_lib,
        &source_file,
    ]);

    match Command::new("rustc")
        .args(args)
        .current_dir(&build_dir)
        .stdout(Stdio::piped())
        .spawn()
        .expect("rustc failed to start")
        .wait_with_output()
        .expect("rustc failed")
    {
        output @ _ => match output.status.success() {
            true => {}
            false => {
                eprintln!(
                    "rustc output: {}",
                    String::from_utf8(output.stdout).unwrap(),
                );
                panic!("compile failed")
            }
        },
    }

    Ok(lib_path)
}

use super::shader_impl::*;
type KernelFn = unsafe extern "C" fn(*const KernelFnArgs);

pub struct ShaderImpl {
    lib: libloading::Library,
    entry: libloading::Symbol<'static, KernelFn>,
}
impl ShaderImpl {
    pub fn load(path: PathBuf) -> Self {
        unsafe {
            let lib = libloading::Library::new(&path)
                .unwrap_or_else(|_| panic!("cannot load library {:?}", &path));
            let entry: libloading::Symbol<KernelFn> = unsafe { lib.get(b"kernel_fn").unwrap() };
            let entry: libloading::Symbol<'static, KernelFn> = transmute(entry);
            Self { lib, entry }
        }
    }
    pub fn fn_ptr(&self) -> KernelFn {
        *self.entry
    }
}
pub const MATH_LIB_SRC: &str = include_str!("../../lang/math_impl.rs");
pub const SHADER_LIB_SRC: &str = include_str!("shader_impl.rs");
#[cfg(test)]
mod test {

    #[test]
    fn test_compile() {
        use super::*;
        let src = r#" #[no_mangle] pub extern fn add(x:i32,y:i32)->i32{x+y}
        #[no_mangle] pub extern fn mul(x:i32, y:i32)->i32{x*y}
        #[no_mangle] pub extern fn sin(x:f32)->f32{x.sin()}"#;
        let src = format!("{}{}", MATH_LIB_SRC, src);
        let path = compile(src).unwrap();
        unsafe {
            let lib = libloading::Library::new(path).unwrap();
            let add: libloading::Symbol<unsafe extern "C" fn(i32, i32) -> i32> =
                lib.get(b"add\0").unwrap();
            let mul: libloading::Symbol<unsafe extern "C" fn(i32, i32) -> i32> =
                lib.get(b"mul\0").unwrap();
            let sin: libloading::Symbol<unsafe extern "C" fn(f32) -> f32> =
                lib.get(b"sin\0").unwrap();
            assert_eq!(add(1, 2), 3);
            assert_eq!(mul(2, 4), 8);
            assert_eq!(sin(1.0), 1.0f32.sin());
        }
    }
}
