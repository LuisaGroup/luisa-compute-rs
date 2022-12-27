use std::fs::create_dir;

use luisa::backend::rust::RustBackend;
use luisa::prelude::*;
use luisa_compute as luisa;

#[derive(Copy, Clone, Debug, Value)]
#[repr(C)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

fn main() {
    init();
    let device = RustBackend::create_device().unwrap();
    let x = device.create_buffer::<Vec3>(1024).unwrap();
    let kernel = create_kernel!(device, (BufferVar<Vec3>), |buf_x| {
        let tid = dispatch_id().x();
        let v = make_float3(1.0, 2.0, 3.0);
        let v = v + v;
        buf_x.write(tid, v);
    })
    .unwrap();
    kernel.dispatch([1024, 1, 1], &x).unwrap();
    let mut x_data = vec![Vec3::default(); 1024];
    x.view(..).copy_to(&mut x_data);
    println!("{:?}", &x_data[0..16]);
}
