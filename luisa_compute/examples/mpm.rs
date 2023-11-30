#![allow(non_snake_case)]
use std::env::current_exe;
use std::time::Instant;

use luisa::lang::types::vector::alias::*;
use luisa::lang::types::vector::Mat2;
use luisa::prelude::*;
use luisa_compute as luisa;
use rand::Rng;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

const N_GRID: usize = 128;
const N_STEPS: usize = 50;
const N_PARTICLES: usize = N_GRID * N_GRID / 2;
const DX: f32 = 1.0f32 / N_GRID as f32;
const DT: f32 = 1e-4f32;
const P_RHO: f32 = 1.0f32;
const P_VOL: f32 = (DX * 0.5f32) * (DX * 0.5f32);
const P_MASS: f32 = P_RHO * P_VOL;
const GRAVITY: f32 = 9.81f32;
const BOUND: u32 = 3;
const E: f32 = 400.0f32;
const RESOLUTION: u32 = 512;

#[tracked]
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
    let ctx = luisa::Context::new(current_exe().unwrap());
    let device = ctx.create_device(if args.len() == 2 {
        args[1].as_str()
    } else {
        "cpu"
    });

    let mut rng = rand::thread_rng();
    let x = device.create_buffer_from_slice(
        (0..N_PARTICLES)
            .map(|_| {
                let rx: f32 = rng.gen();
                let ry: f32 = rng.gen();
                Float2::new(rx * 0.4f32 + 0.2f32, ry * 0.4f32 + 0.2f32)
            })
            .collect::<Vec<_>>()
            .as_slice(),
    );
    let v = device.create_buffer_from_slice(&[Float2::new(0.0f32, -1.0f32); N_PARTICLES]);
    let C = device.create_buffer_from_slice(
        &[Mat2 {
            cols: [Float2::new(0.0f32, 0.0f32); 2],
        }; N_PARTICLES],
    );
    let J = device.create_buffer_from_slice(&[1.0f32; N_PARTICLES]);
    let grid_v = device.create_buffer::<f32>(N_GRID * N_GRID * 2);
    let grid_m = device.create_buffer::<f32>(N_GRID * N_GRID);

    let event_loop = EventLoop::new().unwrap();
    let window = winit::window::WindowBuilder::new()
        .with_title("Luisa Compute Rust - MPM")
        .with_inner_size(winit::dpi::LogicalSize::new(RESOLUTION, RESOLUTION))
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();
    let swapchain = device.create_swapchain(
        &window,
        &device.default_stream(),
        RESOLUTION,
        RESOLUTION,
        false,
        false,
        3,
    );
    let display =
        device.create_tex2d::<Float4>(swapchain.pixel_storage(), RESOLUTION, RESOLUTION, 1);

    let trace = track!(|mat: Expr<Mat2>| -> Expr<f32> { mat.col(0).x + mat.col(1).y });

    let index = track!(|xy: Expr<Uint2>| -> Expr<u32> {
        let p = xy.clamp(
            Uint2::expr(0, 0),
            Uint2::expr(N_GRID as u32 - 1, N_GRID as u32 - 1),
        );
        p.x + p.y * N_GRID as u32
    });

    let clear_grid = Kernel::<fn()>::new(&device, &|| {
        let idx = index(dispatch_id().xy());
        grid_v.write(idx * 2, 0.0f32);
        grid_v.write(idx * 2 + 1, 0.0f32);
        grid_m.write(idx, 0.0f32);
    });

    let point_to_grid = Kernel::<fn()>::new(&device, &|| {
        let p = dispatch_id().x;
        let xp = x.read(p) / DX;
        let base = (xp - 0.5f32).cast_i32();
        let fx = xp - base.cast_f32();

        let w = [
            0.5f32 * (1.5f32 - fx) * (1.5f32 - fx),
            0.75f32 - (fx - 1.0f32) * (fx - 1.0f32),
            0.5f32 * (fx - 0.5f32) * (fx - 0.5f32),
        ];
        let stress = -4.0f32 * DT * E * P_VOL * (J.read(p) - 1.0f32) / (DX * DX);
        let affine = Mat2::diag_expr(Float2::expr(stress, stress)) + P_MASS as f32 * C.read(p);
        let vp = v.read(p);
        for_unrolled(0..9usize, |ii| {
            let (i, j) = (ii % 3, ii / 3);
            let offset = Int2::expr(i as i32, j as i32);
            let dpos = (offset.cast_f32() - fx) * DX;
            let weight = w[i].x * w[j].y;
            let vadd = weight * (P_MASS * vp + affine * dpos);
            let idx = index((base + offset).cast_u32());
            grid_v.var().atomic_fetch_add(idx * 2, vadd.x);
            grid_v.var().atomic_fetch_add(idx * 2 + 1, vadd.y);
            grid_m.var().atomic_fetch_add(idx, weight * P_MASS);
        });
    });

    let simulate_grid = Kernel::<fn()>::new(&device, &|| {
        let coord = dispatch_id().xy();
        let i = index(coord);
        let v = Var::<Float2>::zeroed();
        v.store(Float2::expr(
            grid_v.read(i * 2u32),
            grid_v.read(i * 2u32 + 1u32),
        ));
        let m = grid_m.read(i);

        v.store(select(m > 0.0f32, v.load() / m, v.load()));
        let vx = v.load().x;
        let vy = v.load().y - DT * GRAVITY;
        let vx = select(
            coord.x < BOUND && (vx < 0.0f32) || coord.x + BOUND > N_GRID as u32 && (vx > 0.0f32),
            0.0f32.expr(),
            vx,
        );
        let vy = select(
            coord.y < BOUND && (vy < 0.0f32) || coord.y + BOUND > N_GRID as u32 && (vy > 0.0f32),
            0.0f32.expr(),
            vy,
        );
        grid_v.write(i * 2, vx);
        grid_v.write(i * 2 + 1, vy);
    });

    let grid_to_point = Kernel::<fn()>::new(&device, &|| {
        let p = dispatch_id().x;
        let xp = x.read(p) / DX;
        let base = (xp - 0.5f32).cast_i32();
        let fx = xp - base.cast_f32();

        let w = [
            0.5f32 * (1.5f32 - fx) * (1.5f32 - fx),
            0.75f32 - (fx - 1.0f32) * (fx - 1.0f32),
            0.5f32 * (fx - 0.5f32) * (fx - 0.5f32),
        ];
        let new_v = Var::<Float2>::zeroed();
        let new_C = Var::<Mat2>::zeroed();
        new_v.store(Float2::expr(0.0f32, 0.0f32));
        new_C.store(Mat2::expr(Float2::expr(0., 0.), Float2::expr(0., 0.)));
        for_unrolled(0..9usize, |ii| {
            let (i, j) = (ii % 3, ii / 3);

            let offset = Int2::expr(i as i32, j as i32);
            let dpos = (offset.cast_f32() - fx) * DX.expr();
            let weight = w[i].x * w[j].y;
            let idx = index((base + offset).cast_u32());
            let g_v = Float2::expr(grid_v.read(idx * 2u32), grid_v.read(idx * 2u32 + 1u32));
            new_v.store(new_v.load() + weight * g_v);
            new_C.store(new_C.load() + 4.0f32 * weight * g_v.outer_product(dpos) / (DX * DX));
        });

        v.write(p, new_v);
        x.write(p, x.read(p) + new_v.load() * DT);
        J.var()
            .write(p, J.read(p) * (1.0f32 + DT * trace(new_C.load())));
        C.write(p, new_C);
    });

    let clear_display = Kernel::<fn()>::new(&device, &|| {
        display.write(
            dispatch_id().xy(),
            Float4::expr(0.1f32, 0.2f32, 0.3f32, 1.0f32),
        );
    });
    let draw_particles = Kernel::<fn()>::new(&device, &|| {
        let p = dispatch_id().x;
        for i in -1..=1 {
            for j in -1..=1 {
                let pos = (x.read(p) * RESOLUTION as f32).cast_i32() + Int2::expr(i, j);
                if pos.x >= (0i32)
                    && pos.x < (RESOLUTION as i32)
                    && pos.y >= (0i32)
                    && pos.y < (RESOLUTION as i32)
                {
                    display.write(
                        Uint2::expr(pos.x.cast_u32(), RESOLUTION - 1u32 - pos.y.cast_u32()),
                        Float4::expr(0.4f32, 0.6f32, 0.6f32, 1.0f32),
                    );
                }
            }
        }
    });
    escape!({
        event_loop.set_control_flow(ControlFlow::Poll);
        event_loop.run(move |event, elwt| {
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    window_id,
                } if window_id == window.id() => {
                    elwt.exit();
                }
                Event::AboutToWait => {
                    window.request_redraw();
                }
                Event::WindowEvent {
                    event: WindowEvent::RedrawRequested,
                    window_id,
                } if window_id == window.id() => {
                    let tic = Instant::now();
                    {
                        let scope = device.default_stream().scope();
                        scope.present(&swapchain, &display);
                        let mut commands = Vec::new();
                        for _ in 0..N_STEPS {
                            commands.push(clear_grid.dispatch_async([
                                N_GRID as u32,
                                N_GRID as u32,
                                1,
                            ]));
                            commands.push(point_to_grid.dispatch_async([N_PARTICLES as u32, 1, 1]));
                            commands.push(simulate_grid.dispatch_async([
                                N_GRID as u32,
                                N_GRID as u32,
                                1,
                            ]));
                            commands.push(grid_to_point.dispatch_async([N_PARTICLES as u32, 1, 1]));
                        }
                        commands.push(clear_display.dispatch_async([
                            RESOLUTION as u32,
                            RESOLUTION as u32,
                            1,
                        ]));
                        commands.push(draw_particles.dispatch_async([N_PARTICLES as u32, 1, 1]));
                        scope.submit(commands);
                    }
                    let toc = Instant::now();
                    let elapsed = (toc - tic).as_secs_f32();
                    log::info!("time: {}ms", elapsed * 1e3);
                    window.request_redraw();
                }
                _ => (),
            }
        }).unwrap();
    });
}
