use luisa::backend::rust::RustBackend;
use luisa::prelude::*;
use luisa_compute as luisa;

fn main() {
    init();
    let device = RustBackend::create_device().unwrap();
    let x = device.create_buffer::<f32>(1024).unwrap();
    let y = device.create_buffer::<f32>(1024).unwrap();
    let z = device.create_buffer::<f32>(1024).unwrap();
    x.view(..).fill_fn(|i| i as f32);
    y.view(..).fill_fn(|i| 1000.0 * i as f32);
    let kernel = device
        .create_kernel(|builder| {
            let buf_x = builder.buffer::<f32>();
            let buf_y = builder.buffer::<f32>();
            let buf_z = builder.buffer::<f32>();
            let tid = dispatch_id().x();
            let x = buf_x.read(tid);
            let y = buf_y.read(tid);
            buf_z.write(tid, x + y);
        })
        .unwrap();
    let mut args = ArgEncoder::new();
    args.buffer(&x);
    args.buffer(&y);
    args.buffer(&z);
    kernel.dispatch(&args, [1024, 1, 1]).unwrap();
    let mut z_data = vec![0.0; 1024];
    z.view(..).copy_to(&mut z_data);
    println!("{:?}", &z_data[0..16]);
}
