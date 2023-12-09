use luisa::{prelude::*, runtime::ExternalCallable};
use luisa_compute as luisa;
use std::env::current_exe;

fn main() {
    luisa::init_logger();
    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_device("cuda");
    let x = device.create_buffer::<f32>(1024);
    let y = device.create_buffer::<f32>(1024);
    let z = device.create_buffer::<f32>(1024);
    let time = device.create_buffer::<u64>(1024);
    x.view(..).fill_fn(|i| i as f32);
    y.view(..).fill_fn(|i| 1000.0 * i as f32);
    let clock64 = ExternalCallable::<fn() -> Expr<u64>>::new("clock64");
    let kernel = Kernel::<fn(Buffer<f32>)>::new(
        &device,
        &track!(|buf_z| {
            let buf_x = x.var();
            let buf_y = y.var();
            let tid = dispatch_id().x;
            let t0 = clock64.call();
            let x = buf_x.read(tid);
            let y = buf_y.read(tid);

            buf_z.write(tid, x + y);
            let t1 = clock64.call();
            time.write(tid, t1 - t0);
        }),
    );
    kernel.dispatch([1024, 1, 1], &z);
    let z_data = z.view(..).copy_to_vec();
    println!("{:?}", &z_data[0..16]);
    let time = time.copy_to_vec().iter().sum::<u64>() as f64 / 1024.0;
    println!("avg time: {}", time);
}
