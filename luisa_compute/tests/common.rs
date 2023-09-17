use luisa::prelude::*;
use luisa_compute as luisa;
use std::env::current_exe;
fn _signal_handler(signal: libc::c_int) {
    if signal == libc::SIGSEGV {
        panic!("segfault detected");
    }
}
static ONCE: std::sync::Once = std::sync::Once::new();
pub fn device_name() -> String {
    match std::env::var("LUISA_TEST_DEVICE") {
        Ok(device) => device,
        Err(_) => "cpu".to_string(),
    }
}
pub fn get_device() -> Device {
    let show_log = match std::env::var("LUISA_TEST_LOG") {
        Ok(log) => log == "1",
        Err(_) => false,
    };
    ONCE.call_once(|| unsafe {
        if show_log {
            init_logger_verbose();
        }
        libc::signal(libc::SIGSEGV, _signal_handler as usize);
    });
    let curr_exe = current_exe().unwrap();
    let runtime_dir = curr_exe.parent().unwrap().parent().unwrap();
    let ctx = Context::new(runtime_dir);
    let device = device_name();
    let device = ctx.create_device(&device);
    device.create_buffer_from_slice(&[1.0f32]);
    device
}
