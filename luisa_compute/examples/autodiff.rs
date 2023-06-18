use std::env::current_exe;

use luisa::*;
use luisa_compute as luisa;
fn main() {
    luisa::init_logger();

    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_device("cpu");
    let x = device.create_buffer::<f32>(1024);
    let y = device.create_buffer::<f32>(1024);
    let dx = device.create_buffer::<f32>(1024);
    let dy = device.create_buffer::<f32>(1024);
    x.fill_fn(|i| i as f32);
    y.fill_fn(|i| 1.0 + i as f32);
    let shader = device.create_kernel::<(Buffer<f32>, Buffer<f32>, Buffer<f32>, Buffer<f32>)>(
        &|buf_x: BufferVar<f32>,
          buf_y: BufferVar<f32>,
          buf_dx: BufferVar<f32>,
          buf_dy: BufferVar<f32>| {
            let tid = dispatch_id().x();
            let x = buf_x.read(tid);
            let y = buf_y.read(tid);
            autodiff(|| {
                requires_grad(x);
                requires_grad(y);
                let z = if_!(x.cmpgt(y), {
                    x * 4.0
                }, else {
                    y * 0.5
                });
                backward(z);
                buf_dx.write(tid, gradient(x));
                buf_dy.write(tid, gradient(y));
            });
        },
    );

    shader.dispatch([1024, 1, 1], &x.view(..), &y, &dx, &dy);
    let dx = dx.copy_to_vec();
    println!("{:?}", &dx[0..16]);
    let dy = dy.copy_to_vec();
    println!("{:?}", &dy[0..16]);
}
