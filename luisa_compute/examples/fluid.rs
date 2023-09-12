#![allow(non_snake_case)]

use std::mem::swap;
use std::{env::current_exe, time::Instant};

use luisa::init_logger;
#[allow(unused_imports)]
use luisa::prelude::*;
use luisa::*;
use luisa_compute as luisa;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

const N_GRID: i32 = 512;

fn main() {
    init_logger();
    std::env::set_var("WINIT_UNIX_BACKEND", "x11");
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 2 {
        log::info!(
            "Usage: {} <backend>. <backend>: cpu, cuda, dx, metal, remote",
            args[0]
        );
        return;
    }
    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_device(if args.len() == 2 {
        args[1].as_str()
    } else {
        "metal"
    });

    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("Luisa Compute Rust - Fluid")
        .with_inner_size(winit::dpi::LogicalSize::new(N_GRID, N_GRID))
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();
    let swapchain = device.create_swapchain(
        &window,
        &device.default_stream(),
        N_GRID as u32,
        N_GRID as u32,
        false,
        false,
        3,
    );
    let display =
        device.create_tex2d::<Float4>(swapchain.pixel_storage(), N_GRID as u32, N_GRID as u32, 1);

    let sim_fps = 60.0f32;
    let sim_substeps = 2;
    let iterations = 100;
    let dt = (1.0f32 / sim_fps) / (sim_substeps as f32);
    let mut sim_time = 0.0f32;
    let speed = 400.0f32;

    let grid_size = (N_GRID * N_GRID) as usize;
    let mut u0 = device.create_buffer::<Float2>(grid_size);
    let mut u1 = device.create_buffer::<Float2>(grid_size);

    let mut rho0 = device.create_buffer::<f32>(grid_size);
    let mut rho1 = device.create_buffer::<f32>(grid_size);

    let mut p0 = device.create_buffer::<f32>(grid_size);
    let mut p1 = device.create_buffer::<f32>(grid_size);
    let div = device.create_buffer::<f32>(grid_size);

    let index = |xy: Expr<Uint2>| -> Expr<u32> {
        let p = xy.clamp(
            make_uint2(0, 0),
            make_uint2(N_GRID as u32 - 1, N_GRID as u32 - 1),
        );
        p.x() + p.y() * N_GRID as u32
    };

    let lookup_float = |f: &BufferVar<f32>, x: Int, y: Int| -> Float {
        return f.read(index(make_uint2(x.uint(), y.uint())));
    };

    let sample_float = |f: BufferVar<f32>, x: Float, y: Float| -> Float {
        let lx = x.floor().int();
        let ly = y.floor().int();

        let tx = x - lx.float();
        let ty = y - ly.float();

        let s0 = lookup_float(&f, lx, ly).lerp(lookup_float(&f, lx + 1, ly), tx);
        let s1 = lookup_float(&f, lx, ly + 1).lerp(lookup_float(&f, lx + 1, ly + 1), tx);

        return s0.lerp(s1, ty);
    };

    let lookup_vel = |f: &BufferVar<Float2>, x: Int, y: Int| -> Float2Expr {
        return f.read(index(make_uint2(x.uint(), y.uint())));
    };

    let sample_vel = |f: BufferVar<Float2>, x: Float, y: Float| -> Float2Expr {
        let lx = x.floor().int();
        let ly = y.floor().int();

        let tx = x - lx.float();
        let ty = y - ly.float();

        let s0 = lookup_vel(&f, lx, ly).lerp(lookup_vel(&f, lx + 1, ly), make_float2(tx, tx));
        let s1 =
            lookup_vel(&f, lx, ly + 1).lerp(lookup_vel(&f, lx + 1, ly + 1), make_float2(tx, tx));

        return s0.lerp(s1, make_float2(ty, ty));
    };

    let advect = device
        .create_kernel_async::<fn(Buffer<Float2>, Buffer<Float2>, Buffer<f32>, Buffer<f32>)>(
            &|u0, u1, rho0, rho1| {
                let coord = dispatch_id().xy();
                let u = u0.read(index(coord));

                // trace backward
                let mut p = make_float2(coord.x().float(), coord.y().float());
                p = p - u * dt;

                // advect
                u1.write(index(coord), sample_vel(u0, p.x(), p.y()));
                rho1.write(index(coord), sample_float(rho0, p.x(), p.y()));
            },
        );

    let divergence = device.create_kernel_async::<fn(Buffer<Float2>, Buffer<f32>)>(&|u, div| {
        let coord = dispatch_id().xy();
        if_!(coord.x().cmplt(N_GRID - 1) & coord.y().cmplt(N_GRID - 1), {
            let dx = (u.read(index(make_uint2(coord.x() + 1, coord.y()))).x()
                - u.read(index(coord)).x())
                * 0.5;
            let dy = (u.read(index(make_uint2(coord.x(), coord.y() + 1))).y()
                - u.read(index(coord)).y())
                * 0.5;
            div.write(index(coord), dx + dy);
        });
    });

    let pressure_solve =
        device.create_kernel_async::<fn(Buffer<f32>, Buffer<f32>, Buffer<f32>)>(&|p0, p1, div| {
            let coord = dispatch_id().xy();
            let i = coord.x().int();
            let j = coord.y().int();
            let ij = index(coord);

            let s1 = lookup_float(&p0, i - 1, j);
            let s2 = lookup_float(&p0, i + 1, j);
            let s3 = lookup_float(&p0, i, j - 1);
            let s4 = lookup_float(&p0, i, j + 1);

            // Jacobi update
            let err = s1 + s2 + s3 + s4 - div.read(ij);
            p1.write(ij, err * 0.25f32);
        });

    let pressure_apply = device.create_kernel_async::<fn(Buffer<f32>, Buffer<Float2>)>(&|p, u| {
        let coord = dispatch_id().xy();
        let i = coord.x().int();
        let j = coord.y().int();
        let ij = index(coord);

        if_!(
            i.cmpgt(0) & i.cmplt(N_GRID - 1) & j.cmpgt(0) & j.cmplt(N_GRID - 1),
            {
                // pressure gradient
                let f_p = make_float2(
                    p.read(index(make_uint2(i.uint() + 1, j.uint())))
                        - p.read(index(make_uint2(i.uint() - 1, j.uint()))),
                    p.read(index(make_uint2(i.uint(), j.uint() + 1)))
                        - p.read(index(make_uint2(i.uint(), j.uint() - 1))),
                ) * 0.5f32;

                u.write(ij, u.read(ij) - f_p);
            }
        );
    });

    let integrate = device.create_kernel_async::<fn(Buffer<Float2>, Buffer<f32>)>(&|u, rho| {
        let coord = dispatch_id().xy();
        let ij = index(coord);

        // gravity
        let f_g = make_float2(-90.8f32, 0.0f32) * rho.read(ij);

        // integrate
        u.write(ij, u.read(ij) + dt * f_g);

        // fade
        rho.write(ij, rho.read(ij) * (1.0f32 - 0.1f32 * dt));
    });

    let init =
        device.create_kernel_async::<fn(Buffer<f32>, Buffer<Float2>, Float2)>(&|rho, u, dir| {
            let coord = dispatch_id().xy();
            let i = coord.x().int();
            let j = coord.y().int();
            let ij = index(coord);
            let d = make_float2((i - N_GRID / 2).float(), (j - N_GRID / 2).float()).length();

            let radius = 5.0f32;
            if_!(d.cmplt(radius), {
                rho.write(ij, 1.0f32);
                u.write(ij, dir);
            });
        });

    let init_grid = device.create_kernel_async::<fn()>(&|| {
        let idx = index(dispatch_id().xy());
        u0.var().write(idx, make_float2(0.0f32, 0.0f32));
        u1.var().write(idx, make_float2(0.0f32, 0.0f32));

        rho0.var().write(idx, 0.0f32);
        rho1.var().write(idx, 0.0f32);

        p0.var().write(idx, 0.0f32);
        p1.var().write(idx, 0.0f32);
        div.var().write(idx, 0.0f32);
    });

    let clear_pressure = device.create_kernel_async::<fn()>(&|| {
        let idx = index(dispatch_id().xy());
        p0.var().write(idx, 0.0f32);
        p1.var().write(idx, 0.0f32);
    });

    let draw_rho = device.create_kernel_async::<fn()>(&|| {
        let coord = dispatch_id().xy();
        let ij = index(coord);
        let value = rho0.var().read(ij);
        display.var().write(
            make_uint2(coord.x(), (N_GRID - 1) as u32 - coord.y()),
            make_float4(value, 0.0f32, 0.0f32, 1.0f32),
        );
    });

    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll();
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                window_id,
            } if window_id == window.id() => {
                *control_flow = ControlFlow::Exit;
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                let tic = Instant::now();
                {
                    let scope = device.default_stream().scope();
                    scope.present(&swapchain, &display);
                    let mut commands: Vec<Command> = Vec::new();
                    for _ in 0..sim_substeps {
                        let angle = (sim_time * 4.0f32).sin() * 1.5f32;
                        let vel = Float2::new(angle.cos() * speed, angle.sin() * speed);

                        // update emitters
                        commands.push(init.dispatch_async(
                            [N_GRID as u32, N_GRID as u32, 1],
                            &rho0,
                            &u0,
                            &vel,
                        ));
                        // force integrate
                        commands.push(integrate.dispatch_async(
                            [N_GRID as u32, N_GRID as u32, 1],
                            &u0,
                            &rho0,
                        ));
                        commands.push(divergence.dispatch_async(
                            [N_GRID as u32, N_GRID as u32, 1],
                            &u0,
                            &div,
                        ));

                        // pressure solve
                        commands.push(clear_pressure.dispatch_async([
                            N_GRID as u32,
                            N_GRID as u32,
                            1,
                        ]));
                        for _ in 0..iterations {
                            commands.push(pressure_solve.dispatch_async(
                                [N_GRID as u32, N_GRID as u32, 1],
                                &p0,
                                &p1,
                                &div,
                            ));
                            swap(&mut p0, &mut p1);
                        }

                        commands.push(pressure_apply.dispatch_async(
                            [N_GRID as u32, N_GRID as u32, 1],
                            &p0,
                            &u0,
                        ));
                        commands.push(advect.dispatch_async(
                            [N_GRID as u32, N_GRID as u32, 1],
                            &u0,
                            &u1,
                            &rho0,
                            &rho1,
                        ));

                        swap(&mut u0, &mut u1);
                        swap(&mut rho0, &mut rho1);
                        sim_time += dt;
                    }

                    commands.push(draw_rho.dispatch_async([N_GRID as u32, N_GRID as u32, 1]));
                    scope.submit(commands);
                }
                let toc = Instant::now();
                let elapsed = (toc - tic).as_secs_f32();
                log::info!("time: {}ms", elapsed * 1e3);
                window.request_redraw();
            }
            _ => (),
        }
    });
}
