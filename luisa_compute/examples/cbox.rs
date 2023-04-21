use std::env::current_exe;
use std::ops::Not;

#[allow(unused_imports)]
use luisa::prelude::*;
use luisa::rtx::Accel;
use luisa::{Expr, Float3, Value};
use luisa_compute as luisa;

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
    let device = ctx
        .create_device(if args.len() == 2 {
            args[1].as_str()
        } else {
            "cpu"
        })
        .unwrap();
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

    let vertex_heap = device.create_bindless_array(65536).unwrap();
    let index_heap = device.create_bindless_array(65536).unwrap();
    let mut vertex_buffers: Vec<Buffer<PackedFloat3>> = vec![];
    let mut index_buffers: Vec<Buffer<RtxIndex>> = vec![];
    let accel = device.create_accel(AccelOption::default()).unwrap();

    for (index, model) in models.iter().enumerate() {
        let vertex_buffer = device
            .create_buffer_from_slice(unsafe {
                let vertex_ptr = &model.mesh.positions as *const _ as *const f32;
                std::slice::from_raw_parts(
                    vertex_ptr as *const PackedFloat3,
                    model.mesh.positions.len() / 3,
                )
            })
            .unwrap();
        let index_buffer = device
            .create_buffer_from_slice(unsafe {
                let index_ptr = &model.mesh.indices as *const _ as *const u32;
                std::slice::from_raw_parts(
                    index_ptr as *const RtxIndex,
                    model.mesh.indices.len() / 3,
                )
            })
            .unwrap();
        let mesh = device
            .create_mesh(
                vertex_buffer.view(..),
                index_buffer.view(..),
                AccelOption::default(),
            )
            .unwrap();
        vertex_buffers.push(vertex_buffer);
        index_buffers.push(index_buffer);
        vertex_heap.emplace_buffer(index, vertex_buffers.last().unwrap());
        index_heap.emplace_buffer(index, index_buffers.last().unwrap());
        mesh.build(AccelBuildRequest::ForceBuild);
        accel.push_mesh(&mesh, glam::Mat4::IDENTITY.into(), 255, true);
    }
    accel.build(AccelBuildRequest::ForceBuild);

    let raytracing_kernel = device
        .create_kernel::<(Tex2d<Float4>, Tex2d<u32>, Accel, Uint2)>(
            &|image: Tex2dVar<Float4>,
              seed_image: Tex2dVar<u32>,
              accel: AccelVar,
              resolution: Uint2Expr| {
                set_block_size([16u32, 16u32, 1u32]);

                let cbox_materials: Expr<[Float3; 8]> = const_([
                    Float3::new(0.725f32, 0.710f32, 0.680f32), // floor
                    Float3::new(0.725f32, 0.710f32, 0.680f32), // ceiling
                    Float3::new(0.725f32, 0.710f32, 0.680f32), // back wall
                    Float3::new(0.140f32, 0.450f32, 0.091f32), // right wall
                    Float3::new(0.630f32, 0.065f32, 0.050f32), // left wall
                    Float3::new(0.725f32, 0.710f32, 0.680f32), // short box
                    Float3::new(0.725f32, 0.710f32, 0.680f32), // tall box
                    Float3::new(0.000f32, 0.000f32, 0.000f32), // light
                ]);

                let lcg = |state: U32Var| -> PrimExpr<f32> {
                    const LCG_A: u32 = 1664525u32;
                    const LCG_C: u32 = 1013904223u32;
                    state.store(LCG_A * state.load() + LCG_C);
                    (state.load() & 0x00ffffffu32).float() * (1.0f32 / f32::from_bits(0x01000000u32))
                };

                let make_ray = |o: Expr<Float3>, d: Expr<Float3>, tmin: Expr<f32>, tmax: Expr<f32>| -> Expr<RtxRay> {
                    RtxRayExpr::new(o, tmin, d, tmax)
                };

                let generate_ray = |p: Expr<Float2>| -> Expr<RtxRay> {
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

                let balanced_heuristic = |pdf_a: PrimExpr<f32>, pdf_b: PrimExpr<f32>| {
                    pdf_a / (pdf_a + pdf_b).max(1e-4f32)
                };

                let offset_ray_origin = |p: Expr<Float3>, n: Expr<Float3>| -> Expr<Float3> {
                    const ORIGIN: f32 = 1.0f32 / 32.0f32;
                    const FLOAT_SCALE: f32 = 1.0f32 / 65536.0f32;
                    const INT_SCALE: f32 = 256.0f32;
                    let of_i = (INT_SCALE * n).int();
                    let p_i = p.int() + Int3Expr::select(p.cmplt(0.0f32), -of_i, of_i);
                    Float3Expr::select(p.abs().cmplt(ORIGIN), p + FLOAT_SCALE * n, p_i.float())
                };

                let make_onb = |normal: Expr<Float3>| -> Expr<Onb> {
                    let binormal = select(
                        normal.x().abs().cmpgt(normal.z().abs()),
                        make_float3(-normal.y(), normal.x(), 0.0f32),
                        make_float3(0.0f32, -normal.z(), normal.y()),
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
                // let mut radiance = make_float3(0.0f32, 0.0f32, 0.0f32);
                let radiance = var!(Float3);
                radiance.store(make_float3(0.0f32, 0.0f32, 0.0f32));

                let loop_index = var!(u32);
                while_!(loop_index.load().cmplt(SPP_PER_DISPATCH), {
                    let init_ray = generate_ray(pixel * make_float2(1.0f32, -1.0f32));
                    let ray = var!(RtxRay);
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

                    let depth = var!(u32);
                    while_!(depth.load().cmplt(10u32), {
                        let hit = accel.trace_closest(ray);
                        if_!(hit.inst_id().cmpeq(u32::MAX), {
                            break_();
                        });

                        let vertex_buffer = vertex_heap.var().buffer::<PackedFloat3>(hit.inst_id());
                        let triangle = index_heap
                            .var()
                            .buffer::<RtxIndex>(hit.inst_id())
                            .read(hit.prim_id());
                        let p0: Expr<Float3> = vertex_buffer.read(triangle.x()).into();
                        let p1: Expr<Float3> = vertex_buffer.read(triangle.y()).into();
                        let p2: Expr<Float3> = vertex_buffer.read(triangle.z()).into();
                        let p = p0 * (1.0f32 - hit.u() - hit.v()) + p1 * hit.u() + p2 * hit.v();
                        let n = (p1 - p0).cross(p2 - p0).normalize();

                        let origin: Expr<Float3> = ray.load().orig().into();
                        let direction: Expr<Float3> = ray.load().dir().into();
                        let cos_wi = -direction.dot(n);
                        if_!(cos_wi.cmplt(1e-4f32), {
                            break_();
                        });

                        // hit light
                        if_!(hit.inst_id().cmpeq(7u32), {
                            if_!(depth.load().cmpeq(0u32), {
                                radiance.store(radiance.load() + light_emission);
                            }, else {
                                let pdf_light = (p - origin).length_squared();
                                let mis_weight = balanced_heuristic(pdf_bsdf.load(), pdf_light);
                                radiance.store(radiance.load() + mis_weight * beta.load() * light_emission);
                            });
                            break_();
                        });

                        // sample light
                        let ux_light = lcg(state);
                        let uy_light = lcg(state);
                        let p_light = light_position + ux_light * light_u + uy_light * light_v;
                        let pp = offset_ray_origin(p, n);
                        let pp_light = offset_ray_origin(p_light, light_normal);
                        let d_light = (pp - pp_light).length();
                        let wi_light = (pp_light - pp).normalize();
                        let shadow_ray = make_ray(offset_ray_origin(pp, n), wi_light, 0.0f32.into(), d_light);
                        let occluded = accel.trace_any(shadow_ray);
                        let cos_wi_light = wi_light.dot(n);
                        let cos_light = -light_normal.dot(wi_light);
                        let albedo = cbox_materials.read(hit.inst_id());
                        if_!(occluded.not() & cos_wi_light.cmpgt(1e-4f32) & cos_light.cmpgt(1e-4f32), {
                            let pdf_light = (d_light * d_light) / (light_area * cos_light);
                            let pdf_bsdf = cos_wi_light * std::f32::consts::FRAC_1_PI;
                            let mis_weight = balanced_heuristic(pdf_light, pdf_bsdf);
                            let bsdf = albedo * std::f32::consts::FRAC_1_PI * cos_wi_light;
                            radiance.store(radiance.load() + beta.load() * bsdf * mis_weight * light_emission / pdf_light.max(1e-4f32));
                        });

                        // sample BSDF
                        let onb = make_onb(n);
                        let ux = lcg(state);
                        let uy = lcg(state);
                        let new_direction = onb.to_world(cosine_sample_hemisphere(make_float2(ux, uy)));
                        ray.store(make_ray(pp, new_direction, 0.0f32.into(), std::f32::MAX.into()));
                        beta.store(beta.load() * albedo);
                        pdf_bsdf.store(cos_wi * std::f32::consts::FRAC_1_PI);

                        // russian roulette
                        let l = make_float3(0.212671f32, 0.715160f32, 0.072169f32).dot(beta.load());
                        if_!(l.cmpeq(0.0f32), { break_(); });
                        let q = l.max(0.05f32);
                        let r = lcg(state);
                        if_!(r.cmpgt(q), { break_(); });
                        beta.store(beta.load() / q);

                        depth.store(depth.load() + 1)
                    });
                    loop_index.store(loop_index.load() + 1)
                });
                radiance.store(radiance.load() / SPP_PER_DISPATCH as f32);
                seed_image.write(coord, state.load());
                if_!(radiance.load().is_nan().any(), { radiance.store(make_float3(0.0f32, 0.0f32, 0.0f32)); });
                let radiance = radiance.load().clamp(0.0f32, 30.0f32);
                image.write(dispatch_id().xy(), make_float4(radiance.x(), radiance.y(), radiance.z(), 1.0f32));
            },
        )
        .unwrap();
}
