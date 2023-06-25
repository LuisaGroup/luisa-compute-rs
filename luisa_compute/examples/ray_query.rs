use std::env::current_exe;

use image::Rgb;
#[allow(unused_imports)]
use luisa::prelude::*;
use luisa::rtx::{RayQuery, TriangleCandidate};
use luisa_compute as luisa;
use winit::event::Event as WinitEvent;
use winit::{
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop},
};
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
    let vbuffer: Buffer<Float3> = device.create_buffer_from_slice(&[
        Float3::new(-0.5, -0.5, 0.0),
        Float3::new(0.5, 0.0, 0.0),
        Float3::new(0.0, 0.5, 0.0),
    ]);
    let tbuffer: Buffer<PackedUint3> =
        device.create_buffer_from_slice(&[PackedUint3::new(0, 1, 2)]);
    let mesh = device.create_mesh(vbuffer.view(..), tbuffer.view(..), AccelOption::default());
    mesh.build(AccelBuildRequest::ForceBuild);
    let accel = device.create_accel(Default::default());
    accel.push_mesh(&mesh, Mat4::identity(), 0xff, true);
    accel.build(AccelBuildRequest::ForceBuild);
    let img_w = 800;
    let img_h = 800;
    let img = device.create_tex2d::<Float4>(PixelStorage::Byte4, img_w, img_h, 1);
    let rt_kernel = device.create_kernel::<()>(&|| {
        let accel = accel.var();
        let px = dispatch_id().xy();
        let xy = px.float() / make_float2(img_w as f32, img_h as f32);
        let xy = 2.0 * xy - 1.0;
        let o = make_float3(0.0, 0.0, -1.0);
        let d = make_float3(xy.x(), xy.y(), 0.0) - o;
        let d = d.normalize();
        let ray = rtx::RayExpr::new(o, 1e-3, d, 1e9);
        let hit = accel.query_all(
            ray,
            255,
            RayQuery {
                on_triangle_hit: |candidate: TriangleCandidate| {
                    let bary = candidate.bary();
                    let uvw = make_float3(1.0 - bary.x() - bary.y(), bary.x(), bary.y());
                    if_!(
                        uvw.xy().length().cmplt(0.8)
                            & uvw.yz().length().cmplt(0.8)
                            & uvw.xz().length().cmplt(0.8),
                        {
                            candidate.commit();
                        }
                    );
                },
                on_procedural_hit: |_| {},
            },
        );
        let img = img.view(0).var();
        let color = select(
            hit.valid(),
            make_float3(hit.bary().x(), hit.bary().x(), 1.0),
            make_float3(0.0, 0.0, 0.0),
        );
        img.write(px, make_float4(color.x(), color.y(), color.z(), 1.0));
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
    {
        let img = image::RgbImage::from_fn(img_w, img_h, |x, y| {
            let i = x + y * img_w;
            let px = img_buffer[i as usize];
            Rgb([px[0], px[1], px[2]])
        });
        img.save("triangle.png").unwrap();
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
