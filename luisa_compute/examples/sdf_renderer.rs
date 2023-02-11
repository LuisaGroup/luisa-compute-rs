use std::env::current_exe;

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
    // luisa::sys::init_cpp(current_exe().unwrap().parent().unwrap());
    let device = create_cpu_device().unwrap();
    let spheres = device.create_buffer::<Sphere>(1).unwrap();
    spheres.view(..).copy_from(&[Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 1.0,
    }]);
    let x = device.create_buffer::<f32>(1024).unwrap();
    let kernel = device
        .create_kernel::<(Buffer<f32>, Buffer<Sphere>)>(
            &|buf_x: BufferVar<f32>, spheres: BufferVar<Sphere>| {
                let tid = dispatch_id().x();
                let o = make_float3(0.0, 0.0, -2.0);
                let d = make_float3(0.0, 0.0, 1.0);
                let sphere = spheres.read(0);
                let t = var!(f32);
                while_!(t.load().cmplt(10.0), {
                    let p = o + d * t.load();
                    let d = (p - sphere.center()).length() - sphere.radius();
                    if_!(d.cmplt(0.001), {
                        break_();
                    });
                    t.store(t.load() + d);
                });
                buf_x.write(tid, t.load());
            },
        )
        .unwrap();
    kernel.dispatch([1024, 1, 1], &x, &spheres).unwrap();
    let mut x_data = vec![f32::default(); 1024];
    x.view(..).copy_to(&mut x_data);
    println!("{:?}", &x_data[0..16]);
}
