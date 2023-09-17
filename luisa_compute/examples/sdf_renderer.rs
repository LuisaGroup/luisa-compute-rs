use std::env::current_exe;

use luisa::prelude::*;
use luisa_compute as luisa;

#[derive(Copy, Clone, Debug, Value)]
#[repr(C)]
pub struct Sphere {
    pub center: Float3,
    pub radius: f32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() <= 2,
        "Usage: {} <backend>. <backend>: cpu, cuda, dx, metal, remote",
        args[0]
    );

    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_device(if args.len() == 2 {
        args[1].as_str()
    } else {
        "cpu"
    });
    let spheres = device.create_buffer::<Sphere>(1);
    spheres.view(..).copy_from(&[Sphere {
        center: Float3::new(0.0, 0.0, 0.0),
        radius: 1.0,
    }]);
    let x = device.create_buffer::<f32>(1024);
    let shader =
        device.create_kernel::<fn(Buffer<f32>, Buffer<Sphere>)>(
            &|buf_x: BufferVar<f32>, spheres: BufferVar<Sphere>| {
                let tid = dispatch_id().x();
                let o = Float3::expr(0.0, 0.0, -2.0);
                let d = Float3::expr(0.0, 0.0, 1.0);
                let sphere = spheres.read(0);
                let t = Var::<f32>::zeroed();
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
        );
    shader.dispatch([1024, 1, 1], &x, &spheres);
    let mut x_data = vec![f32::default(); 1024];
    x.view(..).copy_to(&mut x_data);
    println!("{:?}", &x_data[0..16]);
}
