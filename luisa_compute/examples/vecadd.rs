use luisa::backend::rust::RustBackend;
use luisa::prelude::*;
use luisa_compute as luisa;

fn main() {
    let device = RustBackend::create_device().unwrap();
    let x = device.create_buffer::<f32>(1024).unwrap();
    let y = device.create_buffer::<f32>(1024).unwrap();
    let z = device.create_buffer::<f32>(1024).unwrap();
    let kernel = device
        .create_kernel(|builder| {
            let buf_x = builder.buffer::<f32>();
            let buf_y = builder.buffer::<f32>();
            let buf_z = builder.buffer::<f32>();
            let tid = dispatch_id().x;
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
}
