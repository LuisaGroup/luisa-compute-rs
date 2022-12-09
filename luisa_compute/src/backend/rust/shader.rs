use std::{
    env::current_exe,
    fs::canonicalize,
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
fn check_command_exists(cmd: &str) -> bool {
    // try to spawn cmd
    Command::new(cmd)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .is_ok()
}
fn find_cxx_compiler() -> Option<String> {
    if check_command_exists("clang++") {
        return Some("clang++".into());
    }
    if check_command_exists("g++") {
        return Some("g++".into());
    }
    None
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
    let source_file = format!("{}/{}.cc", build_dir.display(), target);
    std::fs::write(&source_file, source).map_err(|e| {
        eprintln!("fs::write({}) failed", source_file);
        e
    })?;
    log::info!("compiling kernel {}", source_file);
    let compiler = find_cxx_compiler().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound, "no c++ compiler found")
    })?;
    // dbg!(&source_file);
    let mut args: Vec<&str> = vec![];
    args.push("-O3");
    args.push("-std=c++17");
    args.push("-fno-math-errno");
    if cfg!(target_os = "linux") {
        args.push("-fPIC");
    }
    args.push("-shared");
    args.push(&source_file);
    args.push("-o");
    args.push(&target_lib);

    match Command::new(compiler)
        .args(args)
        .current_dir(&build_dir)
        .stdout(Stdio::piped())
        .spawn()
        .expect("clang++ failed to start")
        .wait_with_output()
        .expect("clang++ failed")
    {
        output @ _ => match output.status.success() {
            true => {}
            false => {
                eprintln!(
                    "clang++ output: {}",
                    String::from_utf8(output.stdout).unwrap(),
                );
                panic!("compile failed")
            }
        },
    }

    Ok(lib_path)
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct KernelFnArgs {
    captured: *mut u8,
    args: *mut u8,
}
type KernelFn = unsafe extern "C" fn(*const KernelFnArgs);

pub(super) struct ShaderImpl {
    lib: libloading::Library,
    entry: libloading::Symbol<'static, KernelFn>,
}

const LIB_SRC: &str = include_str!("lib.hxx");
