use luisa::backend::rust::RustBackend;
use luisa::prelude::*;
use luisa_compute as luisa;

fn main() {
    init();
    let device = create_cpu_device().unwrap();
    let x = device.create_buffer::<f32>(1024).unwrap();
    let y = device.create_buffer::<f32>(1024).unwrap();
    let dx = device.create_buffer::<f32>(1024).unwrap();
    let dy = device.create_buffer::<f32>(1024).unwrap();
    x.view(..).fill_fn(|i| i as f32);
    y.view(..).fill_fn(|i| 1000.0 * i as f32);
    let kernel = device
        .create_kernel(wrap_fn!(
            4,
            |buf_x: BufferVar<f32>,
             buf_y: BufferVar<f32>,
             buf_dx: BufferVar<f32>,
             buf_dy: BufferVar<f32>| {
                let tid = dispatch_id().x();
                let x = buf_x.read(tid);
                let y = buf_y.read(tid);
                autodiff(|| {
                    requires_grad(x);
                    requires_grad(y);
                    let z = x * y.sin();
                    backward(z);
                    buf_dx.write(tid, gradient(x));
                    buf_dy.write(tid, gradient(y));
                });
            }
        ))
        .unwrap();
    kernel.dispatch([1024, 1, 1], &x, &y, &dx, &dy).unwrap();
    let mut dx_data = vec![0.0; 1024];
    dx.view(..).copy_to(&mut dx_data);
    println!("{:?}", &dx_data[0..16]);
    let mut dy_data = vec![0.0; 1024];
    dy.view(..).copy_to(&mut dy_data);
    println!("{:?}", &dy_data[0..16]);
}
