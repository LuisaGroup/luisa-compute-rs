use std::env::current_exe;

use image::Rgb;
use luisa::lang::types::vector::alias::*;
use luisa::lang::types::vector::*;
use luisa::lang::types::*;
use luisa::lang::*;
use luisa::prelude::*;
use luisa::rtx::{AccelBuildRequest, AccelOption, Ray};
use luisa_compute as luisa;
use winit::event::{Event as WinitEvent, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
fn main() {
    luisa::init_logger();

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
    let vbuffer: Buffer<[f32; 3]> =
        device.create_buffer_from_slice(&[[-0.5, -0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]);
    let tbuffer: Buffer<[u32; 3]> = device.create_buffer_from_slice(&[[0, 1, 2]]);
    let mesh = device.create_mesh(vbuffer.view(..), tbuffer.view(..), AccelOption::default());
    mesh.build(AccelBuildRequest::ForceBuild);
    let accel = device.create_accel(Default::default());
    accel.push_mesh(&mesh, Mat4::identity(), 0xff, true);
    accel.build(AccelBuildRequest::ForceBuild);
    let img_w = 800;
    let img_h = 800;
    let img = device.create_tex2d::<Float4>(PixelStorage::Byte4, img_w, img_h, 1);
    let rt_kernel = Kernel::<fn()>::new(
        &device,
        track!(|| {
            let accel = accel.var();
            let px = dispatch_id().xy();
            let xy = px.as_::<Float2>() / Float2::expr(img_w as f32, img_h as f32);
            let xy = 2.0 * xy - 1.0;
            let o = Float3::expr(0.0, 0.0, -1.0);
            let d = Float3::expr(xy.x, xy.y, 0.0) - o;
            let d = d.normalize();
            let ray = Ray::new_expr(
                Expr::<[f32; 3]>::from(o),
                1e-3,
                Expr::<[f32; 3]>::from(d),
                1e9,
            );
            let hit = accel.trace_closest(ray);
            let img = img.view(0).var();
            let color = select(
                hit.valid(),
                Float3::expr(hit.u, hit.v, 1.0),
                Float3::expr(0.0, 0.0, 0.0),
            );
            img.write(px, Float4::expr(color.x, color.y, color.z, 1.0));
        }),
    );
    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("Luisa Compute Rust - Ray Tracing")
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
