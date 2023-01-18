use std::env::{current_dir, current_exe};

use luisa::prelude::*;
use luisa_compute as luisa;

fn main() {
    init();
    init_logger();
    sys::init_cpp(current_exe().unwrap().parent().unwrap().parent().unwrap());
    let device = std::env::args().nth(1).unwrap_or("cpu".to_string());
    let device = create_device(&device).unwrap();
    let x = device.create_buffer::<f32>(1024).unwrap();
    let y = device.create_buffer::<f32>(1024).unwrap();
    let z = device.create_buffer::<f32>(1024).unwrap();
    x.view(..).fill_fn(|i| i as f32);
    y.view(..).fill_fn(|i| 1000.0 * i as f32);
    let kernel = device
        .create_kernel::<(Buffer<f32>,)>(&|buf_z| {
            // z is pass by arg
            let buf_x = x.var(); // x and y are captured
            let buf_y = y.var();
            let tid = dispatch_id().x();
            let x = buf_x.read(tid);
            let y = buf_y.read(tid);
            buf_z.write(tid, x + y);
        })
        .unwrap();
    kernel.dispatch([1024, 1, 1], &z).unwrap();
    let z_data = z.view(..).copy_to_vec();
    println!("{:?}", &z_data[0..16]);
}
