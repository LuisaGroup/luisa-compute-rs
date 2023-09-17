use luisa::derive::*;
use luisa_compute as luisa;
use luisa::Value;
use luisa::prelude::*;
use std::env::current_exe;

fn main() {
    use luisa::*;
    init_logger();
    let ctx = Context::new(current_exe().unwrap());
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
    let add = device.create_callable::<fn(Expr<f32>, Expr<f32>)->Expr<f32>>(&|a, b| a + b);
    let x = device.create_buffer::<f32>(1024);
    let y = device.create_buffer::<f32>(1024);
    let z = device.create_buffer::<f32>(1024);
    x.view(..).fill_fn(|i| i as f32);
    y.view(..).fill_fn(|i| 1000.0 * i as f32);
    let kernel = device.create_kernel::<fn(Buffer<f32>)>(&|buf_z| {
        let buf_x = x.var();
        let buf_y = y.var();
        let tid = dispatch_id().x();
        let x = buf_x.read(tid);
        let y = buf_y.read(tid);

        buf_z.write(tid, add.call(x, y));
    });
    kernel.dispatch([1024, 1, 1], &z);
    let z_data = z.view(..).copy_to_vec();
    println!("{:?}", &z_data[0..16]);
}
