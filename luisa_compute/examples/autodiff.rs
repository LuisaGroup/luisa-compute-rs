use std::env::current_exe;
use std::f32::consts::PI;

use luisa::lang::diff::*;
use luisa::prelude::*;
use luisa_compute as luisa;
fn main() {
    luisa::init_logger_verbose();

    let ctx = Context::new(current_exe().unwrap());
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() <= 2,
        "Usage: {} <backend>. <backend>: cpu, cuda, dx, metal, remote",
        args[0]
    );
    let device = ctx.create_device(if args.len() == 2 {
        args[1].as_str()
    } else {
        "cpu"
    });
    let x = device.create_buffer::<f32>(1024);
    let y = device.create_buffer::<f32>(1024);
    let dx_rev = device.create_buffer::<f32>(1024);
    let dy_rev = device.create_buffer::<f32>(1024);
    let dx_fwd = device.create_buffer::<f32>(1024);
    let dy_fwd = device.create_buffer::<f32>(1024);
    let dx_gt = device.create_buffer::<f32>(1024);
    let dy_gt = device.create_buffer::<f32>(1024);
    x.fill_fn(|i| i as f32);
    y.fill_fn(|i| 1.0 + i as f32);
    let shader = Kernel::<fn()>::new(
        &device,
        track!(|| {
            let tid = dispatch_id().x;
            let buf_x = x.var();
            let buf_y = y.var();
            let x = buf_x.read(tid);
            let y = buf_y.read(tid);
            let f = |x: Expr<f32>, y: Expr<f32>| {
                if x > y {
                    x * y
                } else {
                    y * x + (x / 4.0 * PI).sin()
                }
            };
            let df = |x: Expr<f32>, y: Expr<f32>| {
                if x > y {
                    (y, x)
                } else {
                    (y + (x / 4.0 * PI).cos() / 4.0 * PI, x)
                }
            };
            autodiff(|| {
                requires_grad(x);
                requires_grad(y);
                let z = f(x, y);
                backward(z);
                dx_rev.write(tid, gradient(x));
                dy_rev.write(tid, gradient(y));
            });
            forward_autodiff(2, || {
                propagate_gradient(x, &[1.0f32.expr(), 0.0f32.expr()]);
                propagate_gradient(y, &[0.0f32.expr(), 1.0f32.expr()]);
                let z = f(x, y);
                let dx = output_gradients(z)[0];
                let dy = output_gradients(z)[1];
                dx_fwd.write(tid, dx);
                dy_fwd.write(tid, dy);
            });
            {
                let (dx, dy) = df(x, y);
                dx_gt.write(tid, dx);
                dy_gt.write(tid, dy);
            }
        }),
    );

    shader.dispatch([1024, 1, 1]);
    {
        let dx = dx_rev.copy_to_vec();
        println!("{:?}", &dx[0..16]);
        let dy = dy_rev.copy_to_vec();
        println!("{:?}", &dy[0..16]);
    }
    {
        let dx = dx_fwd.copy_to_vec();
        println!("{:?}", &dx[0..16]);
        let dy = dy_fwd.copy_to_vec();
        println!("{:?}", &dy[0..16]);
    }
    {
        let dx = dx_gt.copy_to_vec();
        println!("{:?}", &dx[0..16]);
        let dy = dy_gt.copy_to_vec();
        println!("{:?}", &dy[0..16]);
    }
}
