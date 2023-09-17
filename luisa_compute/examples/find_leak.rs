use std::env::current_exe;

use luisa::prelude::*;
use luisa_compute as luisa;

fn main() {
    luisa::init_logger();
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() <= 2,
        "Usage: {} <backend>. <backend>: cpu, cuda, dx, metal, remote",
        args[0]
    );

    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_device(if args.len() == 2 {
        args[1].as_str()
    } else {
        "cpu"
    });
    let test_pass_by_arg = |count: usize| {
        {
            let x = device.create_buffer::<f32>(count);
            let y = device.create_buffer::<f32>(count);
            let z = device.create_buffer::<f32>(count);
            x.view(..).fill_fn(|i| i as f32);
            y.view(..).fill_fn(|i| 1000.0 * i as f32);
            let kernel = device.create_kernel::<fn(Buffer<f32>, Buffer<f32>, Buffer<f32>)>(
                &|buf_x, buf_y, buf_z| {
                    let tid = dispatch_id().x();
                    let x = buf_x.read(tid);
                    let y = buf_y.read(tid);
                    buf_z.write(tid, x + y);
                },
            );
            let add_buffers = |x: &Buffer<f32>, y: &Buffer<f32>, z: &Buffer<f32>| {
                kernel.dispatch([1024, 1, 1], x, y, z);
            };
            add_buffers(&x, &y, &z);
        }
        println!("Should free up buffers here.");
    };
    let test_pass_capture = |count: usize| {
        {
            let x = device.create_buffer::<f32>(count);
            let y = device.create_buffer::<f32>(count);
            let z = device.create_buffer::<f32>(count);
            x.view(..).fill_fn(|i| i as f32);
            y.view(..).fill_fn(|i| 1000.0 * i as f32);
            let kernel = device.create_kernel::<fn(Buffer<f32>)>(&|buf_z| {
                let buf_x = x.var();
                let buf_y = y.var();
                let tid = dispatch_id().x();
                let x = buf_x.read(tid);
                let y = buf_y.read(tid);
                buf_z.write(tid, x + y);
            });
            let add_buffers = |z: &Buffer<f32>| {
                kernel.dispatch([1024, 1, 1], z);
            };
            add_buffers(&z);
        }
        println!("Should free up buffers here.");
    };
    test_pass_by_arg(1024 * 1024);
    test_pass_capture(1024 * 1024);
}
