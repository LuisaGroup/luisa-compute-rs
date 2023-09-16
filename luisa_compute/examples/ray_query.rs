use std::env::current_exe;

use image::Rgb;
#[allow(unused_imports)]
use luisa::prelude::*;
use luisa::rtx::{Aabb, ProceduralCandidate, RayQuery, TriangleCandidate};
use luisa::Float3;
use luisa::{derive::*, PackedFloat3};
use luisa_compute as luisa;
use winit::event::Event as WinitEvent;
use winit::{
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop},
};

#[derive(Copy, Clone, Debug, Value)]
#[repr(C)]
pub struct Sphere {
    pub center: Float3,
    pub radius: f32,
}
impl Sphere {
    fn aabb(&self) -> Aabb {
        Aabb {
            min: PackedFloat3::new(
                self.center.x - self.radius,
                self.center.y - self.radius,
                self.center.z - self.radius,
            ),
            max: PackedFloat3::new(
                self.center.x + self.radius,
                self.center.y + self.radius,
                self.center.z + self.radius,
            ),
        }
    }
}

fn main() {
    use luisa::*;
    init_logger();

    std::env::set_var("WINIT_UNIX_BACKEND", "x11");

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
        "cuda"
    });
    let vbuffer: Buffer<Float3> = device.create_buffer_from_slice(&[
        Float3::new(-0.5, -0.5, -0.1),
        Float3::new(0.5, 0.0, -0.1),
        Float3::new(0.0, 0.5, -0.1),
    ]);
    let tbuffer: Buffer<PackedUint3> =
        device.create_buffer_from_slice(&[PackedUint3::new(0, 1, 2)]);
    let mesh = device.create_mesh(vbuffer.view(..), tbuffer.view(..), AccelOption::default());
    mesh.build(AccelBuildRequest::ForceBuild);

    let spheres = [
        Sphere {
            center: Float3::new(0.5, 0.5, 0.0),
            radius: 0.3,
        },
        Sphere {
            center: Float3::new(-0.7, 0.0, 0.0),
            radius: 0.2,
        },
        Sphere {
            center: Float3::new(0.5, 0.5, 2.0),
            radius: 0.8,
        },
    ];
    let aabb = device.create_buffer_from_slice::<rtx::Aabb>(&[
        spheres[0].aabb(),
        spheres[1].aabb(),
        spheres[2].aabb(),
    ]);
    let spheres = device.create_buffer_from_slice::<Sphere>(&spheres);
    let translate = Float3::new(0.0, 0.0, 0.0);
    let transform: Mat4 = {
        let mut m = glam::Mat4::IDENTITY;
        m = glam::Mat4::from_translation(translate.into()) * m;
        m
    }
    .into();
    let scaled: Mat4 = {
        let mut m = glam::Mat4::IDENTITY;
        m = glam::Mat4::from_scale(glam::vec3(2.0, 2.0, 2.0)) * m;
        m
    }
    .into();
    let sphere_accel = device.create_procedural_primitive(aabb.view(..), AccelOption::default());
    sphere_accel.build(AccelBuildRequest::ForceBuild);
    let accel = device.create_accel(Default::default());
    accel.push_mesh(&mesh, scaled, 0xff, false);
    accel.push_procedural_primitive(&sphere_accel, transform, 0xff);
    accel.build(AccelBuildRequest::ForceBuild);
    let img_w = 800;
    let img_h = 800;
    let img = device.create_tex2d::<Float4>(PixelStorage::Byte4, img_w, img_h, 1);
    let debug_hit_t = device.create_buffer::<f32>(4);
    let rt_kernel = device.create_kernel::<fn()>(&|| {
        let accel = accel.var();
        let px = dispatch_id().xy();
        let xy = px.float() / Float2::expr(img_w as f32, img_h as f32);
        let xy = 2.0 * xy - 1.0;
        let o = Float3::expr(0.0, 0.0, -2.0);
        let d = Float3::expr(xy.x(), xy.y(), 0.0) - o;
        let d = d.normalize();
        let ray = rtx::RayExpr::new(o + const_(translate), 1e-3, d, 1e9);
        let hit = accel.query_all(
            ray,
            255,
            RayQuery {
                on_triangle_hit: |candidate: TriangleCandidate| {
                    let bary = candidate.bary();
                    let uvw = Float3::expr(1.0 - bary.x() - bary.y(), bary.x(), bary.y());
                    let t = candidate.committed_ray_t();
                    if_!(px.cmpeq(Uint2::expr(400, 400)).all(), {
                        debug_hit_t.write(0, t);
                        debug_hit_t.write(1, candidate.ray().tmax());
                    });
                    if_!(
                        uvw.xy().length().cmplt(0.8)
                            & uvw.yz().length().cmplt(0.8)
                            & uvw.xz().length().cmplt(0.8),
                        {
                            candidate.commit();
                        }
                    );
                },
                on_procedural_hit: |candidate: ProceduralCandidate| {
                    let ray = candidate.ray();
                    let prim = candidate.prim();
                    let sphere = spheres.var().read(prim);
                    let o = ray.orig().unpack();
                    let d = ray.dir().unpack();
                    let t = var!(f32);

                    for_range(const_(0i32)..const_(100i32), |_| {
                        let dist = (o + d * t.load() - (sphere.center() + const_(translate)))
                            .length()
                            - sphere.radius();
                        if_!(dist.cmplt(0.001), {
                            if_!(px.cmpeq(Uint2::expr(400, 400)).all(), {
                                debug_hit_t.write(2, *t);
                                debug_hit_t.write(3, candidate.ray().tmax());
                            });
                            if_!(t.cmplt(ray.tmax()), {
                                candidate.commit(t.load());
                            });
                            break_();
                        });
                        t.store(t.load() + dist);
                    });
                },
            },
        );
        let img = img.view(0).var();
        let color = if_!(
            hit.triangle_hit(),
            {
                let bary = hit.bary();
                let uvw = Float3::expr(1.0 - bary.x() - bary.y(), bary.x(), bary.y());
                uvw
            },
            else,
            {
                if_!(
                    hit.procedural_hit(),
                    {
                        let prim = hit.prim_id();
                        let sphere = spheres.var().read(prim);
                        let normal = (ray.orig().unpack()
                            + ray.dir().unpack() * hit.committed_ray_t()
                            - sphere.center())
                        .normalize();
                        let light_dir = Float3::expr(1.0, 0.6, -0.2).normalize();
                        let light = Float3::expr(1.0, 1.0, 1.0);
                        let ambient = Float3::expr(0.1, 0.1, 0.1);
                        let diffuse = light * normal.dot(light_dir).max(0.0);
                        let color = ambient + diffuse;
                        color
                    },
                    else,
                    { Float3::expr(0.0, 0.0, 0.0) }
                )
            }
        );
        img.write(px, Float4::expr(color.x(), color.y(), color.z(), 1.0));
    });
    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("Luisa Compute Rust - Ray Query")
        .with_inner_size(winit::dpi::LogicalSize::new(img_w, img_h))
        .build(&event_loop)
        .unwrap();
    let swapchain = device.create_swapchain(
        &window,
        &device.default_stream(),
        img_w,
        img_h,
        false,
        true,
        3,
    );
    let mut img_buffer = vec![[0u8; 4]; (img_w * img_h) as usize];
    {
        let scope = device.default_stream().scope();
        scope.submit([
            rt_kernel.dispatch_async([img_w, img_h, 1]),
            img.view(0).copy_to_async(&mut img_buffer),
        ]);
    }
    dbg!(debug_hit_t.copy_to_vec());
    {
        let img = image::RgbImage::from_fn(img_w, img_h, |x, y| {
            let i = x + y * img_w;
            let px = img_buffer[i as usize];
            Rgb([px[0], px[1], px[2]])
        });
        img.save("rq.png").unwrap();
    }
    event_loop.run(move |event, _, control_flow| {
        control_flow.set_wait();
        match event {
            WinitEvent::WindowEvent {
                event: WindowEvent::CloseRequested,
                window_id,
            } if window_id == window.id() => *control_flow = ControlFlow::Exit,
            WinitEvent::MainEventsCleared => {
                window.request_redraw();
            }
            WinitEvent::RedrawRequested(_) => {
                let scope = device.default_stream().scope();
                scope.present(&swapchain, &img);
            }
            _ => (),
        }
    });
}
