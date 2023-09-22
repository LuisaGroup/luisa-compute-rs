use luisa::lang::types::array::VLArrayVar;
use luisa::lang::types::core::*;
use luisa::lang::types::vector::alias::*;
use luisa::prelude::*;
use luisa_compute as luisa;
use luisa_compute_api_types::StreamTag;
use rand::prelude::*;
#[path = "common.rs"]
mod common;
use common::*;

#[test]
fn event() {
    let device = get_device();
    let a: Buffer<i32> = device.create_buffer_from_slice(&[0]);
    let b: Buffer<i32> = device.create_buffer_from_slice(&[0]);
    // compute (1 + 3) * (4 + 5)
    let add = device.create_kernel::<fn(Buffer<i32>, i32)>(&|buf: BufferVar<i32>, v: Expr<i32>| {
        track!(buf.write(0, buf.read(0) + v));
    });
    let mul = device.create_kernel::<fn(Buffer<i32>, Buffer<i32>)>(
        &|a: BufferVar<i32>, b: BufferVar<i32>| {
            track!(a.write(0, a.read(0) * b.read(0)));
        },
    );
    let stream_a = device.create_stream(StreamTag::Compute);
    let stream_b = device.create_stream(StreamTag::Compute);
    {
        let scope_a = stream_a.scope();
        let scope_b = stream_b.scope();
        let event = device.create_event();
        let _ = &scope_a
            << add.dispatch_async([1, 1, 1], &a, &1)
            << add.dispatch_async([1, 1, 1], &b, &4)
            << event.signal(1);
        let _ = &scope_b
            << event.wait(1)
            << add.dispatch_async([1, 1, 1], &a, &3)
            << add.dispatch_async([1, 1, 1], &b, &5)
            << event.signal(2);
        let _ =
            &scope_a << event.wait(2) << mul.dispatch_async([1, 1, 1], &a, &b) << event.signal(3);
        event.synchronize(3);
        // scope_a
        //     .submit([add.dispatch_async([1, 1, 1], &a, &1)])
        //     .submit([add.dispatch_async([1, 1, 1], &b, &4)])
        //     .signal(&event, 1);
        // scope_b
        //     .wait(&event, 1)
        //     .submit([add.dispatch_async([1, 1, 1], &a, &3)])
        //     .submit([add.dispatch_async([1, 1, 1], &b, &5)])
        //     .signal(&event, 2);
        // scope_a
        //     .wait(&event, 2)
        //     .submit([mul.dispatch_async([1, 1, 1], &a, &b)])
        //     .signal(&event, 3);
        // event.synchronize(3);
    }
    let v = a.copy_to_vec();
    assert_eq!(v[0], (1 + 3) * (4 + 5));
}
#[test]
#[should_panic]
fn callable_return_mismatch() {
    let device = get_device();
    let _abs = device.create_callable::<fn(Expr<f32>) -> Expr<f32>>(&track!(|x| {
        if x > 0.0 {
            return true.expr();
        }
        -x
    }));
}
#[test]
#[should_panic]
fn callable_return_mismatch2() {
    let device = get_device();
    let _abs = device.create_callable::<fn(Expr<f32>) -> Expr<f32>>(&track!(|x| {
        if x > 0.0 {
            return;
        }
        -x
    }));
}

#[test]
#[should_panic]
fn callable_return_void_mismatch() {
    let device = get_device();
    let _abs = device.create_callable::<fn(Var<f32>)>(&track!(|x| {
        if x > 0.0 {
            return true.expr();
        }
        *x = -x;
    }));
}
#[test]
fn callable_early_return() {
    let device = get_device();
    let abs = device.create_callable::<fn(Expr<f32>) -> Expr<f32>>(track!(&|x| {
        if x > 0.0 {
            return x;
        }
        -x
    }));
    let x = device.create_buffer::<f32>(1024);
    let mut rng = StdRng::seed_from_u64(0);
    x.fill_fn(|_| rng.gen());
    let y = device.create_buffer::<f32>(1024);
    device
        .create_kernel::<fn()>(&|| {
            let i = dispatch_id().x;
            let x = x.var().read(i);
            let y = y.var();
            y.write(i, abs.call(x));
        })
        .dispatch([x.len() as u32, 1, 1]);
    let x = x.copy_to_vec();
    let y = y.copy_to_vec();
    for i in 0..x.len() {
        assert_eq!(y[i], x[i].abs());
    }
}
#[test]
fn callable() {
    let device = get_device();
    let write = device.create_callable::<fn(BufferVar<u32>, Expr<u32>, Var<u32>)>(
        &|buf: BufferVar<u32>, i: Expr<u32>, v: Var<u32>| {
            buf.write(i, v.load());
            track!(*v+=1;)
        },
    );
    let add =
        device.create_callable::<fn(Expr<u32>, Expr<u32>) -> Expr<u32>>(&|a, b| track!(a + b));
    let x = device.create_buffer::<u32>(1024);
    let y = device.create_buffer::<u32>(1024);
    let z = device.create_buffer::<u32>(1024);
    let w = device.create_buffer::<u32>(1024);
    x.view(..).fill_fn(|i| i as u32);
    y.view(..).fill_fn(|i| 1000 * i as u32);
    let kernel = device.create_kernel::<fn(Buffer<u32>)>(&track!(|buf_z| {
        let buf_x = x.var();
        let buf_y = y.var();
        let buf_w = w.var();
        let tid = dispatch_id().x;
        let x = buf_x.read(tid);
        let y = buf_y.read(tid);
        let z = add.call(x, y).var();
        write.call(buf_z, tid, z);
        buf_w.write(tid, z);
    }));
    kernel.dispatch([1024, 1, 1], &z);
    let z_data = z.view(..).copy_to_vec();
    let w_data = w.view(..).copy_to_vec();
    for i in 0..z_data.len() {
        assert_eq!(z_data[i], (i + 1000 * i) as u32);
        assert_eq!(w_data[i], (i + 1000 * i) as u32 + 1);
    }
}
#[test]
fn vec_cast() {
    let device = get_device();
    let f: Buffer<Float2> = device.create_buffer(1024);
    let i: Buffer<Int2> = device.create_buffer(1024);
    f.view(..)
        .fill_fn(|i| Float2::new(i as f32 + 0.5, i as f32 + 1.5));
    let kernel = device.create_kernel_with_options::<fn()>(
        &|| {
            let f = f.var();
            let i = i.var();
            let tid = dispatch_id().x;
            let v = f.read(tid);
            i.write(tid, v.as_int2());
        },
        KernelBuildOptions {
            name: Some("vec_cast".to_string()),
            ..KernelBuildOptions::default()
        },
    );
    kernel.dispatch([1024, 1, 1]);
    let mut i_data = vec![Int2::new(0, 0); 1024];
    i.view(..).copy_to(&mut i_data);
    for i in 0..1024 {
        assert_eq!(i_data[i].x, i as i32);
        assert_eq!(i_data[i].y, i as i32 + 1);
    }
}
#[test]
fn bool_op() {
    let device = get_device();
    let x: Buffer<bool> = device.create_buffer(1024);
    let y: Buffer<bool> = device.create_buffer(1024);
    let and: Buffer<bool> = device.create_buffer(1024);
    let or: Buffer<bool> = device.create_buffer(1024);
    let xor: Buffer<bool> = device.create_buffer(1024);
    let not: Buffer<bool> = device.create_buffer(1024);
    let mut rng = rand::thread_rng();
    x.view(..).fill_fn(|_| rng.gen());
    y.view(..).fill_fn(|_| rng.gen());
    let kernel = device.create_kernel::<fn()>(&track!(|| {
        let tid = dispatch_id().x;
        let x = x.var().read(tid);
        let y = y.var().read(tid);
        let and = and.var();
        let or = or.var();
        let xor = xor.var();
        let not = not.var();
        and.write(tid, x & y);
        or.write(tid, x | y);
        xor.write(tid, x ^ y);
        not.write(tid, !x);
    }));
    kernel.dispatch([1024, 1, 1]);
    let x = x.view(..).copy_to_vec();
    let y = y.view(..).copy_to_vec();
    let and = and.view(..).copy_to_vec();
    let or = or.view(..).copy_to_vec();
    let xor = xor.view(..).copy_to_vec();
    let not = not.view(..).copy_to_vec();
    for i in 0..1024 {
        let xi = x[i];
        let yi = y[i];
        assert_eq!(and[i], xi & yi);
        assert_eq!(or[i], xi | yi);
        assert_eq!(xor[i], xi ^ yi);
        assert_eq!(not[i], !xi);
    }
}
#[test]
fn bvec_op() {
    let device = get_device();
    let x: Buffer<Bool2> = device.create_buffer(1024);
    let y: Buffer<Bool2> = device.create_buffer(1024);
    let and: Buffer<Bool2> = device.create_buffer(1024);
    let or: Buffer<Bool2> = device.create_buffer(1024);
    let xor: Buffer<Bool2> = device.create_buffer(1024);
    let not: Buffer<Bool2> = device.create_buffer(1024);
    let mut rng = rand::thread_rng();
    x.view(..).fill_fn(|_| Bool2::new(rng.gen(), rng.gen()));
    y.view(..).fill_fn(|_| Bool2::new(rng.gen(), rng.gen()));
    let kernel = device.create_kernel::<fn()>(&track!(|| {
        let tid = dispatch_id().x;
        let x = x.var().read(tid);
        let y = y.var().read(tid);
        let and = and.var();
        let or = or.var();
        let xor = xor.var();
        let not = not.var();
        and.write(tid, x & y);
        or.write(tid, x | y);
        xor.write(tid, x ^ y);
        not.write(tid, !x);
    }));
    kernel.dispatch([1024, 1, 1]);
    let x = x.view(..).copy_to_vec();
    let y = y.view(..).copy_to_vec();
    let and = and.view(..).copy_to_vec();
    let or = or.view(..).copy_to_vec();
    let xor = xor.view(..).copy_to_vec();
    let not = not.view(..).copy_to_vec();
    for i in 0..1024 {
        let xi = x[i];
        let yi = y[i];
        assert_eq!(and[i].x, xi.x & yi.x);
        assert_eq!(or[i].x, xi.x | yi.x);
        assert_eq!(xor[i].x, xi.x ^ yi.x);
        assert_eq!(not[i].x, !xi.x);
        assert_eq!(and[i].y, xi.y & yi.y);
        assert_eq!(or[i].y, xi.y | yi.y);
        assert_eq!(xor[i].y, xi.y ^ yi.y);
        assert_eq!(not[i].y, !xi.y);
    }
}
#[test]
fn vec_bit_minmax() {
    let device = get_device();
    let x: Buffer<Int2> = device.create_buffer(1024);
    let y: Buffer<Int2> = device.create_buffer(1024);
    let z: Buffer<Int2> = device.create_buffer(1024);
    let and: Buffer<Int2> = device.create_buffer(1024);
    let or: Buffer<Int2> = device.create_buffer(1024);
    let xor: Buffer<Int2> = device.create_buffer(1024);
    let not: Buffer<Int2> = device.create_buffer(1024);
    let min = device.create_buffer::<Int2>(1024);
    let max = device.create_buffer::<Int2>(1024);
    let clamp = device.create_buffer::<Int2>(1024);
    let mut rng = rand::thread_rng();
    x.view(..).fill_fn(|_| Int2::new(rng.gen(), rng.gen()));
    y.view(..).fill_fn(|_| Int2::new(rng.gen(), rng.gen()));
    z.view(..).fill_fn(|_| Int2::new(rng.gen(), rng.gen()));
    let kernel = device.create_kernel::<fn()>(&track!(|| {
        let tid = dispatch_id().x;
        let x = x.var().read(tid);
        let y = y.var().read(tid);
        let z = z.var().read(tid);
        let and = and.var();
        let or = or.var();
        let xor = xor.var();
        let not = not.var();
        let min = min.var();
        let max = max.var();
        let clamp = clamp.var();
        and.write(tid, x & y);
        or.write(tid, x | y);
        xor.write(tid, x ^ y);
        not.write(tid, !x);
        min.write(tid, luisa::min(x, y));
        max.write(tid, luisa::max(x, y));
        clamp.write(tid, z.clamp(luisa::min(x, y), luisa::max(x, y)));
    }));
    kernel.dispatch([1024, 1, 1]);
    let x = x.view(..).copy_to_vec();
    let y = y.view(..).copy_to_vec();
    let z = z.view(..).copy_to_vec();
    let and = and.view(..).copy_to_vec();
    let or = or.view(..).copy_to_vec();
    let xor = xor.view(..).copy_to_vec();
    let not = not.view(..).copy_to_vec();
    let min = min.view(..).copy_to_vec();
    let max = max.view(..).copy_to_vec();
    let clamp = clamp.view(..).copy_to_vec();
    for i in 0..1024 {
        let xi = x[i];
        let yi = y[i];
        let zi = z[i];
        assert_eq!(and[i].x, xi.x & yi.x);
        assert_eq!(or[i].x, xi.x | yi.x);
        assert_eq!(xor[i].x, xi.x ^ yi.x);
        assert_eq!(not[i].x, !xi.x);
        assert_eq!(min[i].x, xi.x.min(yi.x));
        assert_eq!(max[i].x, xi.x.max(yi.x));
        assert_eq!(and[i].y, xi.y & yi.y);
        assert_eq!(or[i].y, xi.y | yi.y);
        assert_eq!(xor[i].y, xi.y ^ yi.y);
        assert_eq!(not[i].y, !xi.y);
        assert_eq!(min[i].y, xi.y.min(yi.y));
        assert_eq!(max[i].y, xi.y.max(yi.y));

        assert_eq!(clamp[i].x, zi.x.clamp(min[i].x, max[i].x));
        assert_eq!(clamp[i].y, zi.y.clamp(min[i].y, max[i].y));
    }
}
#[test]
fn vec_permute() {
    let device = get_device();
    let v2: Buffer<Int2> = device.create_buffer(1024);
    let v3: Buffer<Int3> = device.create_buffer(1024);
    v2.view(..)
        .fill_fn(|i| Int2::new(i as i32 + 0, i as i32 + 1));
    let kernel = device.create_kernel::<fn()>(&|| {
        let v2 = v2.var();
        let v3 = v3.var();
        let tid = dispatch_id().x;
        let v = v2.read(tid);
        v3.write(tid, v.xyx());
    });
    kernel.dispatch([1024, 1, 1]);
    let mut i_data = vec![Int3::new(0, 0, 0); 1024];
    v3.view(..).copy_to(&mut i_data);
    for i in 0..1024 {
        assert_eq!(i_data[i].x, i as i32);
        assert_eq!(i_data[i].y, i as i32 + 1);
        assert_eq!(i_data[i].z, i as i32);
    }
}

#[test]
fn if_phi() {
    let device = get_device();
    let x: Buffer<i32> = device.create_buffer(1024);
    let even: Buffer<bool> = device.create_buffer(1024);
    x.view(..).fill_fn(|i| i as i32);
    let kernel = device.create_kernel::<fn()>(&track!(|| {
        let x = x.var();
        let even = even.var();
        let tid = dispatch_id().x;
        let v = x.read(tid);
        let result = if v % 2 == 0 {
            true.expr()
        } else {
            false.expr()
        };
        even.write(tid, result);
    }));
    kernel.dispatch([1024, 1, 1]);
    let mut i_data = vec![false; 1024];
    even.view(..).copy_to(&mut i_data);
    for i in 0..1024 {
        assert_eq!(i_data[i], i % 2 == 0);
    }
}

#[test]
fn switch_phi() {
    let device = get_device();
    let x: Buffer<i32> = device.create_buffer(1024);
    let y: Buffer<i32> = device.create_buffer(1024);
    let z: Buffer<f32> = device.create_buffer(1024);
    x.view(..).fill_fn(|i| i as i32);
    let kernel = device.create_kernel::<fn()>(&|| {
        let buf_x = x.var();
        let buf_y = y.var();
        let buf_z = z.var();
        let tid = dispatch_id().x;
        let x = buf_x.read(tid);
        let (y, z) = switch::<(Expr<i32>, Expr<f32>)>(x)
            .case(0, || (0i32.expr(), 1.0.expr()))
            .case(1, || (1i32.expr(), 2.0.expr()))
            .case(2, || (2i32.expr(), 3.0.expr()))
            .default(|| (3.expr(), 4.0.expr()))
            .finish();
        buf_y.write(tid, y);
        buf_z.write(tid, z);
    });
    kernel.dispatch([1024, 1, 1]);
    let y_data = y.view(..).copy_to_vec();
    let z_data = z.view(..).copy_to_vec();
    for i in 0..1024 {
        match i {
            0 => {
                assert_eq!(y_data[i], 0);
                assert_eq!(z_data[i], 1.0);
            }
            1 => {
                assert_eq!(y_data[i], 1);
                assert_eq!(z_data[i], 2.0);
            }
            2 => {
                assert_eq!(y_data[i], 2);
                assert_eq!(z_data[i], 3.0);
            }
            _ => {
                assert_eq!(y_data[i], 3);
                assert_eq!(z_data[i], 4.0);
            }
        }
    }
}

#[test]
fn switch_unreachable() {
    let device = get_device();
    let x: Buffer<i32> = device.create_buffer(1024);
    let y: Buffer<i32> = device.create_buffer(1024);
    let z: Buffer<f32> = device.create_buffer(1024);
    x.view(..).fill_fn(|i| i as i32 % 3);
    let kernel = device.create_kernel::<fn()>(&|| {
        let buf_x = x.var();
        let buf_y = y.var();
        let buf_z = z.var();
        let tid = dispatch_id().x;
        let x = buf_x.read(tid);
        let (y, z) = switch::<(Expr<i32>, Expr<f32>)>(x)
            .case(0, || (0.expr(), 1.0.expr()))
            .case(1, || (1.expr(), 2.0.expr()))
            .case(2, || (2.expr(), 3.0.expr()))
            .finish();
        buf_y.write(tid, y);
        buf_z.write(tid, z);
    });
    kernel.dispatch([1024, 1, 1]);
    let y_data = y.view(..).copy_to_vec();
    let z_data = z.view(..).copy_to_vec();
    for i in 0..1024 {
        match i % 3 {
            0 => {
                assert_eq!(y_data[i], 0);
                assert_eq!(z_data[i], 1.0);
            }
            1 => {
                assert_eq!(y_data[i], 1);
                assert_eq!(z_data[i], 2.0);
            }
            2 => {
                assert_eq!(y_data[i], 2);
                assert_eq!(z_data[i], 3.0);
            }
            _ => {
                unreachable!()
            }
        }
    }
}

#[test]
fn array_read_write() {
    let device = get_device();
    let x: Buffer<[i32; 4]> = device.create_buffer(1024);
    let kernel = device.create_kernel::<fn()>(&track!(|| {
        let buf_x = x.var();
        let tid = dispatch_id().x;
        let arr = Var::<[i32; 4]>::zeroed();
        let i = i32::var_zeroed();
        while i < 4 {
            arr.write(i.as_u32(), tid.as_i32() + i);
            *i += 1;
        }
        buf_x.write(tid, arr);
    }));
    kernel.dispatch([1024, 1, 1]);
    let x_data = x.view(..).copy_to_vec();
    for i in 0..1024 {
        assert_eq!(
            x_data[i],
            [i as i32, i as i32 + 1, i as i32 + 2, i as i32 + 3]
        );
    }
}
#[test]
fn array_read_write3() {
    let device = get_device();
    let x: Buffer<[i32; 4]> = device.create_buffer(1024);
    let kernel = device.create_kernel::<fn()>(&track!(|| {
        let buf_x = x.var();
        let tid = dispatch_id().x;
        let arr = Var::<[i32; 4]>::zeroed();
        for_range(0..4u32, |i| {
            arr.write(i, tid.as_i32() + i.as_i32());
        });
        buf_x.write(tid, arr);
    }));
    kernel.dispatch([1024, 1, 1]);
    let x_data = x.view(..).copy_to_vec();
    for i in 0..1024 {
        assert_eq!(
            x_data[i],
            [i as i32, i as i32 + 1, i as i32 + 2, i as i32 + 3]
        );
    }
}
#[test]
fn array_read_write4() {
    let device = get_device();
    let x: Buffer<[i32; 4]> = device.create_buffer(1024);
    let kernel = device.create_kernel::<fn()>(&track!(|| {
        let buf_x = x.var();
        let tid = dispatch_id().x;
        let arr = Var::<[i32; 4]>::zeroed();
        for_range(0..6u32, |_| {
            for_range(0..4u32, |i| {
                arr.write(i, arr.read(i) + tid.as_i32() + i.as_i32());
            });
        });
        buf_x.write(tid, arr);
    }));
    kernel.dispatch([1024, 1, 1]);
    let x_data = x.view(..).copy_to_vec();
    for i in 0..1024 {
        assert_eq!(
            x_data[i],
            [
                6 * i as i32,
                6 * i as i32 + 1 * 6,
                6 * i as i32 + 2 * 6,
                6 * i as i32 + 3 * 6
            ]
        );
    }
}
#[test]
fn array_read_write2() {
    let device = get_device();
    let x: Buffer<[i32; 4]> = device.create_buffer(1024);
    let y: Buffer<i32> = device.create_buffer(1024);
    let kernel = device.create_kernel::<fn()>(&track!(|| {
        let buf_x = x.var();
        let buf_y = y.var();
        let tid = dispatch_id().x;
        let arr = Var::<[i32; 4]>::zeroed();
        let i = i32::var_zeroed();
        while i < 4 {
            arr.write(i.as_u32(), tid.as_i32() + i);
            *i += 1;
        }
        buf_x.write(tid, arr);
        buf_y.write(tid, arr.read(0));
    }));
    kernel.dispatch([1024, 1, 1]);
    let x_data = x.view(..).copy_to_vec();
    let y_data = y.view(..).copy_to_vec();
    for i in 0..1024 {
        assert_eq!(
            x_data[i],
            [i as i32, i as i32 + 1, i as i32 + 2, i as i32 + 3]
        );
        assert_eq!(y_data[i], i as i32);
    }
}
#[test]
fn array_read_write_vla() {
    let device = get_device();
    let x: Buffer<[i32; 4]> = device.create_buffer(1024);
    let y: Buffer<i32> = device.create_buffer(1024);
    let kernel = device.create_kernel::<fn()>(&track!(|| {
        let buf_x = x.var();
        let buf_y = y.var();
        let tid = dispatch_id().x;
        let vl = VLArrayVar::<i32>::zero(4);
        let i = i32::var_zeroed();
        while i < 4 {
            vl.write(i.as_u32(), tid.as_i32() + i);
            *i += 1;
        }
        let arr = Var::<[i32; 4]>::zeroed();
        let i = i32::var_zeroed();
        while i < 4 {
            arr.write(i.as_u32(), vl.read(i.as_u32()));
            *i += 1;
        }
        buf_x.write(tid, arr);
        buf_y.write(tid, arr.read(0));
    }));
    kernel.dispatch([1024, 1, 1]);
    let x_data = x.view(..).copy_to_vec();
    let y_data = y.view(..).copy_to_vec();
    for i in 0..1024 {
        assert_eq!(
            x_data[i],
            [i as i32, i as i32 + 1, i as i32 + 2, i as i32 + 3]
        );
        assert_eq!(y_data[i], i as i32);
    }
}
#[test]
fn array_read_write_async_compile() {
    let device = get_device();
    let x: Buffer<[i32; 4]> = device.create_buffer(1024);
    let kernel = device.create_kernel::<fn()>(&track!(|| {
        let buf_x = x.var();
        let tid = dispatch_id().x;
        let arr = Var::<[i32; 4]>::zeroed();
        let i = i32::var_zeroed();
        while i < 4 {
            arr.write(i.as_u32(), tid.as_i32() + i);
            *i += 1;
        }
        buf_x.write(tid, arr);
    }));
    kernel.dispatch([1024, 1, 1]);
    let x_data = x.view(..).copy_to_vec();
    for i in 0..1024 {
        assert_eq!(
            x_data[i],
            [i as i32, i as i32 + 1, i as i32 + 2, i as i32 + 3]
        );
    }
}
#[test]
fn capture_same_buffer_multiple_view() {
    let device = get_device();
    let x = device.create_buffer::<f32>(128);
    let sum = device.create_buffer::<f32>(1);
    x.view(..).fill_fn(|i| i as f32);
    sum.view(..).fill(0.0);
    let shader = device.create_kernel::<fn()>(&track!(|| {
        let tid = dispatch_id().x;
        let buf_x_lo = x.view(0..64).var();
        let buf_x_hi = x.view(64..).var();
        let x = if tid < 64 {
            buf_x_lo.read(tid)
        } else {
            buf_x_hi.read(tid - 64)
        };
        let buf_sum = sum.var();

        buf_sum.atomic_fetch_add(0, x);
    }));
    shader.dispatch([x.len() as u32, 1, 1]);
    let mut sum_data = vec![0.0];
    sum.view(..).copy_to(&mut sum_data);
    let actual = sum_data[0];
    let expected = (x.len() as f32 - 1.0) * x.len() as f32 * 0.5;
    assert!((actual - expected).abs() < 1e-4);
}

#[test]
fn uniform() {
    let device = get_device();
    let x = device.create_buffer::<f32>(128);
    let sum = device.create_buffer::<f32>(1);
    x.view(..).fill_fn(|i| i as f32);
    sum.view(..).fill(0.0);
    let shader = device.create_kernel::<fn(Float3)>(&track!(|v: Expr<Float3>| {
        let tid = dispatch_id().x;
        let buf_x_lo = x.view(0..64).var();
        let buf_x_hi = x.view(64..).var();
        let x = if tid < 64 {
            buf_x_lo.read(tid)
        } else {
            buf_x_hi.read(tid - 64)
        };
        let buf_sum = sum.var();
        let x = x * v.reduce_prod();
        buf_sum.atomic_fetch_add(0, x);
    }));
    shader.dispatch([x.len() as u32, 1, 1], &Float3::new(1.0, 2.0, 3.0));
    let mut sum_data = vec![0.0];
    sum.view(..).copy_to(&mut sum_data);
    let actual = sum_data[0];
    let expected = (x.len() as f32 - 1.0) * x.len() as f32 * 0.5 * 6.0;
    assert!((actual - expected).abs() < 1e-4);
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
struct Big {
    a: [f32; 32],
}
#[test]
fn byte_buffer() {
    let device = get_device();
    let buf = device.create_byte_buffer(1024);
    let mut big = Big { a: [1.0; 32] };
    for i in 0..32 {
        big.a[i] = i as f32;
    }
    let mut cnt = 0usize;
    macro_rules! push {
        ($t:ty, $v:expr) => {{
            let old = cnt;
            let s = std::mem::size_of::<$t>();
            let view = buf.view(cnt..cnt + s);
            let bytes = unsafe { std::slice::from_raw_parts(&$v as *const $t as *const u8, s) };
            view.copy_from(bytes);
            cnt += s;
            old
        }};
    }
    let i0 = push!(Float3, Float3::new(0.0, 0.0, 0.0));
    let i1 = push!(Big, big);
    let i2 = push!(i32, 0i32);
    let i3 = push!(f32, 1f32);
    device
        .create_kernel::<fn()>(&track!(|| {
            let buf = buf.var();
            let i0 = i0 as u64;
            let i1 = i1 as u64;
            let i2 = i2 as u64;
            let i3 = i3 as u64;
            let v0 = buf.read::<Float3>(i0).var();
            let v1 = buf.read::<Big>(i1).var();
            let v2 = buf.read::<i32>(i2).var();
            let v3 = buf.read::<f32>(i3).var();
            *v0 = Float3::expr(1.0, 2.0, 3.0);
            for_range(0u32..32u32, |i| {
                v1.a.write(i, i.as_f32() * 2.0);
            });
            *v2 = 1i32.expr();
            *v3 = 2.0.expr();
            buf.write::<Float3>(i0, v0.load());
            buf.write::<Big>(i1, v1.load());
            buf.write::<i32>(i2, v2.load());
            buf.write::<f32>(i3, v3.load());
        }))
        .dispatch([1, 1, 1]);
    let data = buf.copy_to_vec();
    macro_rules! pop {
        ($t:ty, $offset:expr) => {{
            let s = std::mem::size_of::<$t>();
            let bytes = &data[$offset..$offset + s];
            let v = unsafe {
                std::mem::transmute_copy::<[u8; { std::mem::size_of::<$t>() }], $t>(
                    bytes.try_into().unwrap(),
                )
            };
            v
        }};
    }
    let v0 = pop!(Float3, i0);
    let v1 = pop!(Big, i1);
    let v2 = pop!(i32, i2);
    let v3 = pop!(f32, i3);
    assert_eq!(v0, Float3::new(1.0, 2.0, 3.0));
    assert_eq!(v2, 1);
    assert_eq!(v3, 2.0);
    for i in 0..32 {
        assert!(v1.a[i] == i as f32 * 2.0);
    }
}

#[test]
fn bindless_byte_buffer() {
    let device = get_device();
    let buf = device.create_byte_buffer(1024);
    let out = device.create_byte_buffer(1024);
    let mut big = Big { a: [1.0; 32] };
    for i in 0..32 {
        big.a[i] = i as f32;
    }
    let heap = device.create_bindless_array(64);
    heap.emplace_byte_buffer(0, &buf);
    let mut cnt = 0usize;
    macro_rules! push {
        ($t:ty, $v:expr) => {{
            let old = cnt;
            let s = std::mem::size_of::<$t>();
            let view = buf.view(cnt..cnt + s);
            let bytes = unsafe { std::slice::from_raw_parts(&$v as *const $t as *const u8, s) };
            view.copy_from(bytes);
            cnt += s;
            old
        }};
    }
    let i0 = push!(Float3, Float3::new(0.0, 0.0, 0.0));
    let i1 = push!(Big, big);
    let i2 = push!(i32, 0i32);
    let i3 = push!(f32, 1f32);
    device
        .create_kernel::<fn(ByteBuffer)>(&track!(|out: ByteBufferVar| {
            let heap = heap.var();
            let buf = heap.byte_address_buffer(0u32);
            let i0 = i0 as u64;
            let i1 = i1 as u64;
            let i2 = i2 as u64;
            let i3 = i3 as u64;
            let v0 = buf.read::<Float3>(i0).var();
            let v1 = buf.read::<Big>(i1).var();
            let v2 = buf.read::<i32>(i2).var();
            let v3 = buf.read::<f32>(i3).var();
            *v0 = Float3::expr(1.0, 2.0, 3.0);
            for_range(0u32..32u32, |i| {
                v1.a.write(i, i.as_f32() * 2.0);
            });
            *v2 = 1i32.expr();
            *v3 = 2.0.expr();
            out.write::<Float3>(i0, v0.load());
            out.write::<Big>(i1, v1.load());
            out.write::<i32>(i2, v2.load());
            out.write::<f32>(i3, v3.load());
        }))
        .dispatch([1, 1, 1], &out);
    let data = out.copy_to_vec();
    macro_rules! pop {
        ($t:ty, $offset:expr) => {{
            let s = std::mem::size_of::<$t>();
            let bytes = &data[$offset..$offset + s];
            let v = unsafe {
                std::mem::transmute_copy::<[u8; { std::mem::size_of::<$t>() }], $t>(
                    bytes.try_into().unwrap(),
                )
            };
            v
        }};
    }
    let v0 = pop!(Float3, i0);
    let v1 = pop!(Big, i1);
    let v2 = pop!(i32, i2);
    let v3 = pop!(f32, i3);
    assert_eq!(v0, Float3::new(1.0, 2.0, 3.0));
    assert_eq!(v2, 1);
    assert_eq!(v3, 2.0);
    for i in 0..32 {
        assert!(v1.a[i] == i as f32 * 2.0);
    }
}
