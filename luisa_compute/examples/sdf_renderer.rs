use luisa::backend::rust::RustBackend;
use luisa::prelude::*;
use luisa_compute as luisa;

fn main() {
    init();
    let device = RustBackend::create_device().unwrap();
    let x = device.create_buffer::<Vec3>(1024).unwrap();
    let kernel = device
        .create_kernel(|builder| {
            let tid = dispatch_id().x();
            let buf_x = builder.buffer::<Vec3>();
            let v = make_float3(1.0, 2.0, 3.0);
            let v = v + v;
            buf_x.write(tid, v);
        })
        .unwrap();
    let mut args = ArgEncoder::new();
    args.buffer(&x);
    kernel.dispatch(&args, [1024, 1, 1]).unwrap();
    let mut x_data = vec![Vec3::default(); 1024];
    x.view(..).copy_to(&mut x_data);
    println!("{:?}", &x_data[0..16]);
}
