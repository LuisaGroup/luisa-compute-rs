use luisa::prelude::*;
use luisa_compute as luisa;
use std::env::current_exe;
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

    let palette = device.create_callable::<(Expr<f32>,), Expr<Float3>>(&|d| {
        make_float3(0.2, 0.7, 0.9).lerp(make_float3(1.0, 0.0, 1.0), Float3Expr::splat(d))
    });
    let rotate = device.create_callable::<(Expr<Float2>, Expr<f32>), Expr<Float2>>(&|mut p, a| {
        let c = a.cos();
        let s = a.sin();
        make_float2(p.dot(make_float2(c, s)), p.dot(make_float2(-s, c)))
    });
    let map = device.create_callable::<(Expr<Float3>, Expr<f32>), Expr<f32>>(&|mut p, time| {
        for i in 0..8 {
            let t = time * 0.2;
            let r = rotate.call(p.xz(), t);
            p = make_float3(r.x(), r.y(), p.y()).xzy();
            let r = rotate.call(p.xy(), t * 1.89);
            p = make_float3(r.x(), r.y(), p.z());
            p = make_float3(p.x().abs() - 0.5, p.y(), p.z().abs() - 0.5)
        }
        Float3Expr::splat(1.0).copysign(p).dot(p) * 0.2
    });
    let rm = device.create_callable::<(Expr<Float3>, Expr<Float3>, Expr<f32>), Expr<Float4>>(
        &|ro, rd, time| {
            let t = var!(f32, 0.0);
            let col = var!(Float3);
            let d = var!(f32);
            for_range(0i32..64, |i| {
                let p = ro + rd * *t;
                *d.get_mut() = map.call(p, time) * 0.5;
                if_!(d.cmplt(0.02) | d.cmpgt(100.0), { break_() });
                *col.get_mut() += palette.call(p.length() * 0.1 / (400.0 * *d));
                *t.get_mut() += *d;
            });
            let col = *col;
            make_float4(col.x(), col.y(), col.z(), 1.0 / (100.0 * *d))
        },
    );
    let clear_kernel = device.create_kernel::<(Tex2d<Float4>,)>(&|img| {
        let coord = dispatch_id().xy();
        img.write(coord, make_float4(0.3, 0.4, 0.5, 1.0));
    });
    let main_kernel = device.create_kernel::<(Tex2d<Float4>, f32)>(&|img, time| {
        let xy = dispatch_id().xy();
        let resolution = dispatch_size().xy();
        let uv = (xy.float() - resolution.float() * 0.5) / resolution.x().float();
        let r = rotate.call(make_float2(0.0, -50.0), time);
        let ro = make_float3(r.x(), r.y(), 0.0).xzy();
        let cf = (-ro).normalize();
        let cs = cf.cross(make_float3(0.0, 10.0, 0.0)).normalize();
        let cu = cf.cross(cs).normalize();
        let uuv = ro + cf * 3.0 + uv.x() * cs + uv.y() * cu;
        let rd = (uuv - ro).normalize();
        let col = rm.call(ro, rd, time);
        let color = col.xyz();
        let alpha = col.w();
        let old = img.read(xy).xyz();
        let accum = color.lerp(old, Float3Expr::splat(alpha));
        img.write(xy, make_float4(accum.x(), accum.y(), accum.z(), 1.0));
    });
}
