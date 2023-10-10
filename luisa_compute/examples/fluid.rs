#![allow(non_snake_case)]

use std::env::current_exe;
use std::mem::swap;
use std::time::Instant;

use luisa::lang::types::vector::alias::*;
use luisa::prelude::*;
use luisa_compute as luisa;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

const N_GRID: i32 = 512;

fn main() {
    luisa::init_logger();
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
        "cpu"
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

    let index = track!(|xy: Expr<Uint2>| -> Expr<u32> {
        let p = xy.clamp(
            Uint2::expr(0, 0),
            Uint2::expr(N_GRID as u32 - 1, N_GRID as u32 - 1),
        );
        p.x + p.y * N_GRID as u32
    });

    let lookup_float = track!(
        |f: &BufferVar<f32>, x: Expr<i32>, y: Expr<i32>| -> Expr<f32> {
            f.read(index(Uint2::expr(x.as_u32(), y.as_u32())))
        }
    );

    let sample_float = track!(
        |f: BufferVar<f32>, x: Expr<f32>, y: Expr<f32>| -> Expr<f32> {
            let lx = x.floor().as_i32();
            let ly = y.floor().as_i32();

            let tx = x - lx.as_f32();
            let ty = y - ly.as_f32();

            let s0 = lookup_float(&f, lx, ly).lerp(lookup_float(&f, lx + 1, ly), tx);
            let s1 = lookup_float(&f, lx, ly + 1).lerp(lookup_float(&f, lx + 1, ly + 1), tx);

            s0.lerp(s1, ty)
        }
    );

    let lookup_vel = track!(
        |f: &BufferVar<Float2>, x: Expr<i32>, y: Expr<i32>| -> Expr<Float2> {
            f.read(index(Uint2::expr(x.as_u32(), y.as_u32())))
        }
    );

    let sample_vel = track!(
        |f: BufferVar<Float2>, x: Expr<f32>, y: Expr<f32>| -> Expr<Float2> {
            let lx = x.floor().as_i32();
            let ly = y.floor().as_i32();

            let tx = x - lx.as_f32();
            let ty = y - ly.as_f32();

            let s0 = lookup_vel(&f, lx, ly).lerp(lookup_vel(&f, lx + 1, ly), Float2::expr(tx, tx));
            let s1 = lookup_vel(&f, lx, ly + 1)
                .lerp(lookup_vel(&f, lx + 1, ly + 1), Float2::expr(tx, tx));

            s0.lerp(s1, Float2::expr(ty, ty))
        }
    );

    let advect = Kernel::<fn(Buffer<Float2>, Buffer<Float2>, Buffer<f32>, Buffer<f32>)>::new_async(
        &device,
        &track!(|u0, u1, rho0, rho1| {
            let coord = dispatch_id().xy();
            let u = u0.read(index(coord));

            // trace backward
            let mut p = Float2::expr(coord.x.as_f32(), coord.y.as_f32());
            p = p - u * dt;

            // advect
            u1.write(index(coord), sample_vel(u0, p.x, p.y));
            rho1.write(index(coord), sample_float(rho0, p.x, p.y));
        }),
    );

    let divergence = Kernel::<fn(Buffer<Float2>, Buffer<f32>)>::new_async(
        &device,
        &track!(|u, div| {
            let coord = dispatch_id().xy();
            if coord.x < (N_GRID as u32 - 1) && coord.y < (N_GRID as u32 - 1) {
                let dx = (u.read(index(Uint2::expr(coord.x + 1, coord.y))).x
                    - u.read(index(coord)).x)
                    * 0.5;
                let dy = (u.read(index(Uint2::expr(coord.x, coord.y + 1))).y
                    - u.read(index(coord)).y)
                    * 0.5;
                div.write(index(coord), dx + dy);
            }
        }),
    );

    let pressure_solve = Kernel::<fn(Buffer<f32>, Buffer<f32>, Buffer<f32>)>::new_async(
        &device,
        &track!(|p0, p1, div| {
            let coord = dispatch_id().xy();
            let i = coord.x.as_i32();
            let j = coord.y.as_i32();
            let ij = index(coord);

            let s1 = lookup_float(&p0, i - 1, j);
            let s2 = lookup_float(&p0, i + 1, j);
            let s3 = lookup_float(&p0, i, j - 1);
            let s4 = lookup_float(&p0, i, j + 1);

            // Jacobi update
            let err = s1 + s2 + s3 + s4 - div.read(ij);
            p1.write(ij, err * 0.25f32);
        }),
    );

    let pressure_apply = Kernel::<fn(Buffer<f32>, Buffer<Float2>)>::new_async(
        &device,
        &track!(|p, u| {
            let coord = dispatch_id().xy();
            let i = coord.x.as_i32();
            let j = coord.y.as_i32();
            let ij = index(coord);

            if i > 0 && i < (N_GRID - 1) && j > 0 && j < (N_GRID - 1) {
                // pressure gradient
                let f_p = Float2::expr(
                    p.read(index(Uint2::expr(i.as_u32() + 1, j.as_u32())))
                        - p.read(index(Uint2::expr(i.as_u32() - 1, j.as_u32()))),
                    p.read(index(Uint2::expr(i.as_u32(), j.as_u32() + 1)))
                        - p.read(index(Uint2::expr(i.as_u32(), j.as_u32() - 1))),
                ) * 0.5f32;

                u.write(ij, u.read(ij) - f_p);
            }
        }),
    );

    let integrate = Kernel::<fn(Buffer<Float2>, Buffer<f32>)>::new_async(
        &device,
        &track!(|u, rho| {
            let coord = dispatch_id().xy();
            let ij = index(coord);

            // gravity
            let f_g = Float2::expr(-90.8f32, 0.0f32) * rho.read(ij);

            // integrate
            u.write(ij, u.read(ij) + dt * f_g);

            // fade
            rho.write(ij, rho.read(ij) * (1.0f32 - 0.1f32 * dt));
        }),
    );

    let init = Kernel::<fn(Buffer<f32>, Buffer<Float2>, Float2)>::new_async(
        &device,
        &track!(|rho, u, dir| {
            let coord = dispatch_id().xy();
            let i = coord.x.as_i32();
            let j = coord.y.as_i32();
            let ij = index(coord);
            let d = Float2::expr((i - N_GRID / 2).as_f32(), (j - N_GRID / 2).as_f32()).length();

            let radius = 5.0f32;
            if d < radius {
                rho.write(ij, 1.0f32);
                u.write(ij, dir);
            }
        }),
    );


    let clear_pressure = Kernel::<fn()>::new_async(&device, &|| {
        let idx = index(dispatch_id().xy());
        p0.var().write(idx, 0.0f32);
        p1.var().write(idx, 0.0f32);
    });

    let draw_rho = Kernel::<fn()>::new_async(
        &device,
        &track!(|| {
            let coord = dispatch_id().xy();
            let ij = index(coord);
            let value = rho0.var().read(ij);
            display.var().write(
                Uint2::expr(coord.x, (N_GRID - 1) as u32 - coord.y),
                Float4::expr(value, 0.0f32, 0.0f32, 1.0f32),
            );
        }),
    );

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
