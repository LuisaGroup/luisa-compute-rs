use image::Rgb;
use luisa_compute_api_types::StreamTag;
use rand::Rng;
use std::env::current_exe;
use std::ops::{BitAnd, DerefMut};
use std::time::Instant;
use winit::event::Event as WinitEvent;
use winit::event::WindowEvent;
use winit::event_loop::{ControlFlow, EventLoop};

#[allow(unused_imports)]
use luisa::prelude::*;
use luisa::rtx::{offset_ray_origin, Accel, AccelVar, Index, Ray};
use luisa::{Expr, Float3, Value};
use luisa_compute as luisa;
use luisa_compute::rtx::{RayQuery, TriangleCandidate};

#[derive(Value, Clone, Copy)]
#[repr(C)]
pub struct Onb {
    tangent: Float3,
    binormal: Float3,
    normal: Float3,
}

impl OnbExpr {
    fn to_world(&self, v: Expr<Float3>) -> Expr<Float3> {
        self.tangent() * v.x() + self.binormal() * v.y() + self.normal() * v.z()
    }
}

const CBOX_OBJ: &'static str = "
# The original Cornell Box in OBJ format.
# Note that the real box is not a perfect cube, so
# the faces are imperfect in this data set.
#
# Created by Guedis Cardenas and Morgan McGuire at Williams College, 2011
# Released into the Public Domain.
#
# http://graphics.cs.williams.edu/data
# http://www.graphics.cornell.edu/online/box/data.html
#

mtllib CornellBox-Original.mtl

g floor
v  -1.01  0.00   0.99
v   1.00  0.00   0.99
v   1.00  0.00  -1.04
v  -0.99  0.00  -1.04
f -4 -3 -2 -1

g ceiling
v  -1.02  1.99   0.99
v  -1.02  1.99  -1.04
v   1.00  1.99  -1.04
v   1.00  1.99   0.99
f -4 -3 -2 -1

g backWall
v  -0.99  0.00  -1.04
v   1.00  0.00  -1.04
v   1.00  1.99  -1.04
v  -1.02  1.99  -1.04
f -4 -3 -2 -1

g rightWall
v	1.00  0.00  -1.04
v	1.00  0.00   0.99
v	1.00  1.99   0.99
v	1.00  1.99  -1.04
f -4 -3 -2 -1

g leftWall
v  -1.01  0.00   0.99
v  -0.99  0.00  -1.04
v  -1.02  1.99  -1.04
v  -1.02  1.99   0.99
f -4 -3 -2 -1

g shortBox

# Top Face
v	0.53  0.60   0.75
v	0.70  0.60   0.17
v	0.13  0.60   0.00
v  -0.05  0.60   0.57
f -4 -3 -2 -1

# Left Face
v  -0.05  0.00   0.57
v  -0.05  0.60   0.57
v   0.13  0.60   0.00
v   0.13  0.00   0.00
f -4 -3 -2 -1

# Front Face
v	0.53  0.00   0.75
v	0.53  0.60   0.75
v  -0.05  0.60   0.57
v  -0.05  0.00   0.57
f -4 -3 -2 -1

# Right Face
v	0.70  0.00   0.17
v	0.70  0.60   0.17
v	0.53  0.60   0.75
v	0.53  0.00   0.75
f -4 -3 -2 -1

# Back Face
v	0.13  0.00   0.00
v	0.13  0.60   0.00
v	0.70  0.60   0.17
v	0.70  0.00   0.17
f -4 -3 -2 -1

# Bottom Face
v	0.53  0.00   0.75
v	0.70  0.00   0.17
v	0.13  0.00   0.00
v  -0.05  0.00   0.57
f -4 -3 -2 -1

g tallBox

# Top Face
v	-0.53  1.20   0.09
v	 0.04  1.20  -0.09
v	-0.14  1.20  -0.67
v	-0.71  1.20  -0.49
f -4 -3 -2 -1

# Left Face
v	-0.53  0.00   0.09
v	-0.53  1.20   0.09
v	-0.71  1.20  -0.49
v	-0.71  0.00  -0.49
f -4 -3 -2 -1

# Back Face
v	-0.71  0.00  -0.49
v	-0.71  1.20  -0.49
v	-0.14  1.20  -0.67
v	-0.14  0.00  -0.67
f -4 -3 -2 -1

# Right Face
v	-0.14  0.00  -0.67
v	-0.14  1.20  -0.67
v	 0.04  1.20  -0.09
v	 0.04  0.00  -0.09
f -4 -3 -2 -1

# Front Face
v	 0.04  0.00  -0.09
v	 0.04  1.20  -0.09
v	-0.53  1.20   0.09
v	-0.53  0.00   0.09
f -4 -3 -2 -1

# Bottom Face
v	-0.53  0.00   0.09
v	 0.04  0.00  -0.09
v	-0.14  0.00  -0.67
v	-0.71  0.00  -0.49
f -4 -3 -2 -1

g light
v	-0.24  1.98   0.16
v	-0.24  1.98  -0.22
v	 0.23  1.98  -0.22
v	 0.23  1.98   0.16
f -4 -3 -2 -1";

const SPP_PER_DISPATCH: u32 = 1u32;

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
        "cpu"
    });
    let mut buf = std::io::BufReader::new(CBOX_OBJ.as_bytes());
    let (models, _) = tobj::load_obj_buf(
        &mut buf,
        &tobj::LoadOptions {
            triangulate: true,
            ..Default::default()
        },
        |p| tobj::load_mtl(p),
    )
    .unwrap();

    let vertex_heap = device.create_bindless_array(65536);
    let index_heap = device.create_bindless_array(65536);
    let mut vertex_buffers: Vec<Buffer<PackedFloat3>> = vec![];
    let mut index_buffers: Vec<Buffer<Index>> = vec![];
    let accel = device.create_accel(AccelOption::default());
    let stream = device.create_stream(StreamTag::Graphics);
    stream.with_scope(|s| {
        let mut cmds = vec![];
        for (index, model) in models.iter().enumerate() {
            let vertex_buffer = device.create_buffer(model.mesh.positions.len() / 3);
            let index_buffer = device.create_buffer(model.mesh.indices.len() / 3);
            let mesh = device.create_mesh(
                vertex_buffer.view(..),
                index_buffer.view(..),
                AccelOption::default(),
            );
            cmds.push(vertex_buffer.view(..).copy_from_async(unsafe {
                let vertex_ptr = model.mesh.positions.as_ptr();
                std::slice::from_raw_parts(
                    vertex_ptr as *const PackedFloat3,
                    model.mesh.positions.len() / 3,
                )
            }));
            cmds.push(index_buffer.view(..).copy_from_async(unsafe {
                let index_ptr = model.mesh.indices.as_ptr();
                std::slice::from_raw_parts(index_ptr as *const Index, model.mesh.indices.len() / 3)
            }));

            vertex_buffers.push(vertex_buffer);
            index_buffers.push(index_buffer);
            vertex_heap.emplace_buffer_async(index, vertex_buffers.last().unwrap());
            index_heap.emplace_buffer_async(index, index_buffers.last().unwrap());
            cmds.push(mesh.build_async(AccelBuildRequest::ForceBuild));
            let m = if index == 5 || index == 6 {
                glam::Mat4::from_translation(glam::vec3(0.0, 0.2, 0.0))
            } else {
                glam::Mat4::IDENTITY
            };
            accel.push_mesh(&mesh, m.into(), u32::MAX, false);
        }
        cmds.push(vertex_heap.update_async());
        cmds.push(index_heap.update_async());
        cmds.push(accel.build_async(AccelBuildRequest::ForceBuild));
        s.submit(cmds);
        s.synchronize();
    });

    // use create_kernel_async to compile multiple kernels in parallel
    let path_tracer = device
        .create_kernel_async::<fn(Tex2d<Float4>, Tex2d<u32>, Accel, Uint2)>(
            &|image: Tex2dVar<Float4>,
              seed_image: Tex2dVar<u32>,
              accel: AccelVar,
              resolution: Expr<Uint2>| {
                set_block_size([16u32, 16u32, 1u32]);
                let cbox_materials = const_([
                    Float3::new(0.725f32, 0.710f32, 0.680f32), // floor
                    Float3::new(0.725f32, 0.710f32, 0.680f32), // ceiling
                    Float3::new(0.725f32, 0.710f32, 0.680f32), // back wall
                    Float3::new(0.140f32, 0.450f32, 0.091f32), // right wall
                    Float3::new(0.630f32, 0.065f32, 0.050f32), // left wall
                    Float3::new(0.725f32, 0.710f32, 0.680f32), // short box
                    Float3::new(0.725f32, 0.710f32, 0.680f32), // tall box
                    Float3::new(0.000f32, 0.000f32, 0.000f32), // light
                ]);

                let lcg = |state: Var<u32>| -> Expr<f32> {
                    let lcg = create_static_callable::<fn(Var<u32>)->Expr<f32>>(|state: Var<u32>| {
                        const LCG_A: u32 = 1664525u32;
                        const LCG_C: u32 = 1013904223u32;
                        *state.get_mut() = LCG_A * *state + LCG_C;
                        (*state & 0x00ffffffu32).float() * (1.0f32 / 0x01000000u32 as f32)
                    });
                    lcg.call(state)
                };

                let make_ray = |o: Expr<Float3>, d: Expr<Float3>, tmin: Expr<f32>, tmax: Expr<f32>| -> Expr<Ray> {
                    struct_!(rtx::Ray {
                        orig: o.into(),
                        tmin: tmin,
                        dir:d.into(),
                        tmax: tmax
                    })
                };

                let generate_ray = |p: Expr<Float2>| -> Expr<Ray> {
                    const FOV: f32 = 27.8f32 * std::f32::consts::PI / 180.0f32;
                    let origin = make_float3(-0.01f32, 0.995f32, 5.0f32);

                    let pixel = origin
                        + make_float3(
                        p.x() * f32::tan(0.5f32 * FOV),
                        p.y() * f32::tan(0.5f32 * FOV),
                        -1.0f32,
                    );
                    let direction = (pixel - origin).normalize();
                    make_ray(origin, direction, 0.0f32.into(), f32::MAX.into())
                };

                let balanced_heuristic = |pdf_a: Expr<f32>, pdf_b: Expr<f32>| {
                    pdf_a / (pdf_a + pdf_b).max(1e-4f32)
                };

                let make_onb = |normal: Expr<Float3>| -> Expr<Onb> {
                    let binormal = if_!(
                        normal.x().abs().cmpgt(normal.z().abs()), {
                            make_float3(-normal.y(), normal.x(), 0.0f32)
                        }, else {
                            make_float3(0.0f32, -normal.z(), normal.y())
                        }
                    );
                    let tangent = binormal.cross(normal).normalize();
                    OnbExpr::new(tangent, binormal, normal)
                };

                let cosine_sample_hemisphere = |u: Expr<Float2>| {
                    let r = u.x().sqrt();
                    let phi = 2.0f32 * std::f32::consts::PI * u.y();
                    make_float3(r * phi.cos(), r * phi.sin(), (1.0f32 - u.x()).sqrt())
                };

                let coord = dispatch_id().xy();
                let frame_size = resolution.x().min(resolution.y()).float();
                let state = var!(u32);
                state.store(seed_image.read(coord));

                let rx = lcg(state);
                let ry = lcg(state);

                let pixel = (coord.float() + make_float2(rx, ry)) / frame_size * 2.0f32 - 1.0f32;

                let radiance = var!(Float3);
                radiance.store(make_float3(0.0f32, 0.0f32, 0.0f32));
                for_range(0..SPP_PER_DISPATCH as u32, |_| {
                    let init_ray = generate_ray(pixel * make_float2(1.0f32, -1.0f32));
                    let ray = var!(Ray);
                    ray.store(init_ray);

                    let beta = var!(Float3);
                    beta.store(make_float3(1.0f32, 1.0f32, 1.0f32));
                    let pdf_bsdf = var!(f32);
                    pdf_bsdf.store(0.0f32);

                    let light_position = make_float3(-0.24f32, 1.98f32, 0.16f32);
                    let light_u = make_float3(-0.24f32, 1.98f32, -0.22f32) - light_position;
                    let light_v = make_float3(0.23f32, 1.98f32, 0.16f32) - light_position;
                    let light_emission = make_float3(17.0f32, 12.0f32, 4.0f32);
                    let light_area = light_u.cross(light_v).length();
                    let light_normal = light_u.cross(light_v).normalize();

                    let filter = |c: &TriangleCandidate| {
                        let valid = var!(bool, true);
                        if_!(c.inst().cmpeq(5u32), { valid.store((c.bary().y() * 6.0f32).fract().cmplt(0.6f32)); });
                        if_!(c.inst().cmpeq(6u32), { valid.store((c.bary().y() * 5.0f32).fract().cmplt(0.5f32)); });
                        valid.load()
                    };

                    let depth = var!(u32);
                    while_!(depth.load().cmplt(10u32), {
                        // let hit = accel.trace_closest(ray);
                        let hit = accel.query_all(ray, 255, RayQuery {
                            on_triangle_hit: |c: TriangleCandidate| {
                                if_!(filter(&c), { c.commit(); });
                            },
                            on_procedural_hit: |c| {}
                        });

                        if_!(hit.miss(), {
                            break_();
                        });

                        let vertex_buffer = vertex_heap.var().buffer::<PackedFloat3>(hit.inst_id());
                        let triangle = index_heap
                            .var()
                            .buffer::<Index>(hit.inst_id())
                            .read(hit.prim_id());

                        let p0: Expr<Float3> = vertex_buffer.read(triangle.x()).into();
                        let p1: Expr<Float3> = vertex_buffer.read(triangle.y()).into();
                        let p2: Expr<Float3> = vertex_buffer.read(triangle.z()).into();
                        let m = accel.instance_transform(hit.inst_id());
                        let p = p0 * (1.0f32 - hit.bary().x() - hit.bary().y()) + p1 * hit.bary().x() + p2 * hit.bary().y();
                        let p = (m * make_float4(p.x(), p.y(), p.z(), 1.0f32)).xyz();
                        let n = (p1 - p0).cross(p2 - p0);
                        let n = (m * make_float4(n.x(), n.y(), n.z(), 0.0f32)).xyz().normalize();

                        let origin: Expr<Float3> = ray.load().orig().into();
                        let direction: Expr<Float3> = ray.load().dir().into();
                        let cos_wi = -direction.dot(n);
                        if_!(cos_wi.abs().cmplt(1e-4f32), {
                            break_();
                        });
                        let pp = offset_ray_origin(p, n);
                        let albedo = cbox_materials.read(hit.inst_id());
                        // hit light
                        if_!(hit.inst_id().cmpeq(7u32), {
                            if_!(depth.load().cmpeq(0u32), {
                                radiance.store(radiance.load() + light_emission);
                            }, else {
                                let pdf_light = (p - origin).length_squared() / (light_area * cos_wi);
                                if_!(pdf_light.cmpgt(0.0f32), {
                                    let mis_weight = balanced_heuristic(pdf_bsdf.load(), pdf_light);
                                    radiance.store(radiance.load() + mis_weight * *beta * light_emission);
                                });
                            });
                            break_();
                        }, else{

                            // sample light
                            let ux_light = lcg(state);
                            let uy_light = lcg(state);
                            let p_light = light_position + ux_light * light_u + uy_light * light_v;

                            let pp_light = offset_ray_origin(p_light, light_normal);
                            let d_light = (pp - pp_light).length();
                            let wi_light = (pp_light - pp).normalize();
                            let shadow_ray = make_ray(offset_ray_origin(pp, n), wi_light, 0.0f32.into(), d_light);
                            let occluded = accel.query_any(shadow_ray, 255, RayQuery {
                                on_triangle_hit: |c: TriangleCandidate| {
                                if_!(filter(&c), { c.commit(); });
                            },
                                on_procedural_hit: |c| {}
                            });
                            let occluded = !occluded.miss();
                            let cos_wi_light = wi_light.dot(n);
                            let cos_light = -light_normal.dot(wi_light);

                            if_!(!occluded & (cos_wi_light * cos_wi).cmpgt(0.0f32) & cos_light.cmpgt(1e-4f32), {
                                let pdf_light = (d_light * d_light) / (light_area * cos_light);
                                let pdf_bsdf = cos_wi_light * std::f32::consts::FRAC_1_PI;
                                let mis_weight = balanced_heuristic(pdf_light, pdf_bsdf);
                                let bsdf = albedo * std::f32::consts::FRAC_1_PI * cos_wi_light;
                                radiance.store(*radiance+ *beta * bsdf * mis_weight * light_emission / pdf_light.max(1e-4f32));
                            });
                        });
                        // sample BSDF
                        let onb = make_onb(n);
                        let ux = lcg(state);
                        let uy = lcg(state);
                        let new_direction = onb.to_world(cosine_sample_hemisphere(make_float2(ux, uy)));
                        *ray.get_mut() = make_ray(pp, new_direction, 0.0f32.into(), std::f32::MAX.into());
                        *beta.get_mut() *= albedo;
                        pdf_bsdf.store(cos_wi.abs() * std::f32::consts::FRAC_1_PI);

                        // russian roulette
                        let l = make_float3(0.212671f32, 0.715160f32, 0.072169f32).dot(*beta);
                        if_!(l.cmpeq(0.0f32), { break_(); });
                        let q = l.max(0.05f32);
                        let r = lcg(state);
                        if_!(r.cmpgt(q), { break_(); });
                        *beta.get_mut() = *beta / q;

                        *depth.get_mut() += 1;
                    });
                });
                radiance.store(radiance.load() / SPP_PER_DISPATCH as f32);
                seed_image.write(coord, *state);
                if_!(radiance.load().is_nan().any(), { radiance.store(make_float3(0.0f32, 0.0f32, 0.0f32)); });
                let radiance = radiance.load().clamp(0.0f32, 30.0f32);
                let old = image.read(dispatch_id().xy());
                let spp = old.w();
                let radiance = radiance + old.xyz();
                image.write(dispatch_id().xy(), make_float4(radiance.x(), radiance.y(), radiance.z(), spp + 1.0f32));
            },
        )
        ;
    let display = device.create_kernel_async::<fn(Tex2d<Float4>, Tex2d<Float4>)>(&|acc, display| {
        set_block_size([16, 16, 1]);
        let coord = dispatch_id().xy();
        let radiance = acc.read(coord);
        let spp = radiance.w();
        let radiance = radiance.xyz() / spp;

        // workaround a rust-analyzer bug
        let r = 1.055f32 * radiance.powf(1.0 / 2.4) - 0.055;

        let srgb = Float3Expr::select(radiance.cmplt(0.0031308), radiance * 12.92, r);
        display.write(coord, make_float4(srgb.x(), srgb.y(), srgb.z(), 1.0f32));
    });
    let img_w = 1024;
    let img_h = 1024;
    let acc_img = device.create_tex2d::<Float4>(PixelStorage::Float4, img_w, img_h, 1);
    let seed_img = device.create_tex2d::<u32>(PixelStorage::Int1, img_w, img_h, 1);
    {
        let mut rng = rand::thread_rng();
        let seed_buffer = (0..img_w * img_h)
            .map(|_| rng.gen::<u32>())
            .collect::<Vec<_>>();
        seed_img.view(0).copy_from(&seed_buffer);
    }
    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("Luisa Compute Rust - Ray Tracing")
        .with_inner_size(winit::dpi::LogicalSize::new(img_w, img_h))
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();
    let swapchain = device.create_swapchain(
        &window,
        &device.default_stream(),
        img_w,
        img_h,
        false,
        false,
        3,
    );
    let display_img = device.create_tex2d::<Float4>(swapchain.pixel_storage(), img_w, img_h, 1);
    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll();
        match event {
            WinitEvent::WindowEvent {
                event: WindowEvent::CloseRequested,
                window_id,
            } if window_id == window.id() => {
                // FIXME: support half4 pixel storage
                let mut img_buffer = vec![[0u8; 4]; (img_w * img_h) as usize];
                {
                    let scope = device.default_stream().scope();
                    scope.submit([display_img.view(0).copy_to_async(&mut img_buffer)]);
                }
                {
                    let img = image::RgbImage::from_fn(img_w, img_h, |x, y| {
                        let i = x + y * img_w;
                        let px = img_buffer[i as usize];
                        Rgb([px[0], px[1], px[2]])
                    });
                    img.save("cbox.png").unwrap();
                }
                control_flow.set_exit();
            }
            WinitEvent::MainEventsCleared => {
                window.request_redraw();
            }
            WinitEvent::RedrawRequested(_) => {
                let tic = Instant::now();
                {
                    let scope = device.default_stream().scope();
                    scope.present(&swapchain, &display_img);
                    scope.submit([
                        path_tracer.dispatch_async(
                            [img_w, img_h, 1],
                            &acc_img,
                            &seed_img,
                            &accel,
                            &Uint2::new(img_w, img_h),
                        ),
                        display.dispatch_async([img_w, img_h, 1], &acc_img, &display_img),
                    ]);
                }
                let toc = Instant::now();
                let elapsed = (toc - tic).as_secs_f32();
                log::info!(
                    "time: {}ms {}ms/spp",
                    elapsed * 1e3,
                    elapsed * 1e3 / SPP_PER_DISPATCH as f32
                );
                window.request_redraw();
            }
            _ => (),
        }
    });
}