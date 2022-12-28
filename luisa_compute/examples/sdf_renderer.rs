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
    let spheres = device.create_buffer::<Sphere>(1).unwrap();
    spheres.view(..).copy_from(&[Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 1.0,
    }]);
    let x = device.create_buffer::<Vec3>(1024).unwrap();
    let kernel = device
        .create_kernel(wrap_fn!(
            2,
            |buf_x: BufferVar<Vec3>, spheres: BufferVar<Sphere>| {
                let tid = dispatch_id().x();
                let v = make_float3(1.0, 2.0, 3.0);
                let sphere = spheres.read(0);
                let v = (v - sphere.center()) / sphere.radius();
                buf_x.write(tid, v);
            }
        ))
        .unwrap();
    kernel.dispatch([1024, 1, 1], &x, &spheres).unwrap();
    let mut x_data = vec![Vec3::default(); 1024];
    x.view(..).copy_to(&mut x_data);
    println!("{:?}", &x_data[0..16]);
}
