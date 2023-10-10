use std::cell::RefCell;

use luisa::lang::types::array::VLArrayVar;
use luisa::lang::types::dynamic::*;
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
    let add = device.create_kernel::<fn(Buffer<i32>, i32)>(&|buf, v| {
        track!(buf.write(0, buf.read(0) + v));
    });
    let mul = Kernel::<fn(Buffer<i32>, Buffer<i32>)>::new(&device, &|a, b| {
        track!(a.write(0, a.read(0) * b.read(0)));
    });
    let stream_a = device.create_stream(StreamTag::Compute);
    let stream_b = device.create_stream(StreamTag::Compute);
    {
        let scope_a = stream_a.scope();
        let scope_b = stream_b.scope();
        let event = device.create_event();
        scope_a
            .submit([add.dispatch_async([1, 1, 1], &a, &1)])
            .submit([add.dispatch_async([1, 1, 1], &b, &4)])
            .signal(&event, 1);
        scope_b
            .wait(&event, 1)
            .submit([add.dispatch_async([1, 1, 1], &a, &3)])
            .submit([add.dispatch_async([1, 1, 1], &b, &5)])
            .signal(&event, 2);
        scope_a
            .wait(&event, 2)
            .submit([mul.dispatch_async([1, 1, 1], &a, &b)])
            .signal(&event, 3);
        event.synchronize(3);
    }
    let v = a.copy_to_vec();
    assert_eq!(v[0], (1 + 3) * (4 + 5));
}
#[test]
fn nested_callable_capture_by_value() {
    let device = get_device();
    let add = track!(Callable::<fn(Expr<f32>, Expr<f32>) -> Expr<f32>>::new(
        &device,
        |a, b| {
            // callables can be defined within callables
            let partial_add = Callable::<fn(Expr<f32>) -> Expr<f32>>::new(&device, |y| a + y);
            partial_add.call(b)
        }
    ));
    let x = device.create_buffer::<f32>(1024);
    let y = device.create_buffer::<f32>(1024);
    let z = device.create_buffer::<f32>(1024);
    x.view(..).fill_fn(|i| i as f32);
    y.view(..).fill_fn(|i| 1000.0 * i as f32);
    let kernel = Kernel::<fn(Buffer<f32>)>::new(
        &device,
        &track!(|buf_z| {
            let buf_x = x.var();
            let buf_y = y.var();
            let tid = dispatch_id().x;
            let x = buf_x.read(tid);
            let y = buf_y.read(tid);

            buf_z.write(tid, add.call(x, y));
        }),
    );
    kernel.dispatch([1024, 1, 1], &z);
    let z_data = z.view(..).copy_to_vec();
    for i in 0..x.len() {
        assert_eq!(z_data[i], (i as f32 + 1000.0 * i as f32));
    }
}
#[test]
fn nested_callable_capture_by_ref() {
    let device = get_device();
    let add = track!(Callable::<fn(Expr<f32>, Expr<f32>) -> Expr<f32>>::new(
        &device,
        |a, b| {
            let ret = a.var();
            // callables can be defined within callables
            let partial_add = Callable::<fn(Expr<f32>)>::new(&device, |y| *ret += y);
            partial_add.call(b);
            **ret
        }
    ));
    let x = device.create_buffer::<f32>(1024);
    let y = device.create_buffer::<f32>(1024);
    let z = device.create_buffer::<f32>(1024);
    x.view(..).fill_fn(|i| i as f32);
    y.view(..).fill_fn(|i| 1000.0 * i as f32);
    let kernel = Kernel::<fn(Buffer<f32>)>::new(
        &device,
        &track!(|buf_z| {
            let buf_x = x.var();
            let buf_y = y.var();
            let tid = dispatch_id().x;
            let x = buf_x.read(tid);
            let y = buf_y.read(tid);

            buf_z.write(tid, add.call(x, y));
        }),
    );
    kernel.dispatch([1024, 1, 1], &z);
    let z_data = z.view(..).copy_to_vec();
    for i in 0..x.len() {
        assert_eq!(z_data[i], (i as f32 + 1000.0 * i as f32));
    }
}
#[test]
fn nested_callable_outline_twice() {
    let device = get_device();

    let x = device.create_buffer::<f32>(1024);
    let y = device.create_buffer::<f32>(1024);
    let z = device.create_buffer::<f32>(1024);
    x.view(..).fill_fn(|i| i as f32);
    y.view(..).fill_fn(|i| 1000.0 * i as f32);
    let kernel = Kernel::<fn(Buffer<f32>)>::new(
        &device,
        &track!(|buf_z| {
            let buf_x = x.var();
            let buf_y = y.var();
            let tid = dispatch_id().x;
            let x = buf_x.read(tid);
            let y = buf_y.read(tid);
            let z = 0.0f32.var();
            outline(|| {
                outline(|| {
                    *z += y;
                });
                outline(|| {
                    *z += x;
                })
            });
            buf_z.write(tid, z);
        }),
    );
    kernel.dispatch([1024, 1, 1], &z);
    let z_data = z.view(..).copy_to_vec();
    for i in 0..x.len() {
        assert_eq!(z_data[i], (i as f32 + 1000.0 * i as f32));
    }
}

#[derive(Clone, Copy, Debug, Value, Soa, PartialEq)]
#[repr(C)]
#[value_new(pub)]
pub struct A {
    v: Float3,
}
#[test]
fn nested_callable_capture_gep() {
    let device = get_device();

    let x = device.create_buffer::<f32>(1024);
    let y = device.create_buffer::<f32>(1024);
    let z = device.create_buffer::<f32>(1024);
    x.view(..).fill_fn(|i| i as f32);
    y.view(..).fill_fn(|i| 1000.0 * i as f32);
    let kernel = Kernel::<fn(Buffer<f32>)>::new(
        &device,
        &track!(|buf_z| {
            let buf_x = x.var();
            let buf_y = y.var();
            let tid = dispatch_id().x;
            let a = Var::<A>::zeroed();
            outline(|| {
                let x = buf_x.read(tid);
                let y = buf_y.read(tid);
                *a.v = Float3::expr(x, y, 0.0);
                outline(|| {
                    let v = a.v;
                    *v.z += v.x;
                });
                outline(|| {
                    let v = a.v;
                    *a.v.z += v.y;
                });
                buf_z.write(tid, a.v.z);
            });
        }),
    );
    kernel.dispatch([1024, 1, 1], &z);
    let z_data = z.view(..).copy_to_vec();
    for i in 0..x.len() {
        assert_eq!(z_data[i], (i as f32 + 1000.0 * i as f32));
    }
}
#[test]
fn nested_callable_capture_buffer() {
    let device = get_device();
    let x = device.create_buffer::<f32>(1024);
    let y = device.create_buffer::<f32>(1024);
    let z = device.create_buffer::<f32>(1024);
    x.view(..).fill_fn(|i| i as f32);
    y.view(..).fill_fn(|i| 1000.0 * i as f32);
    let kernel = Kernel::<fn(Buffer<f32>)>::new(
        &device,
        &track!(|buf_z| {
            let tid = dispatch_id().x;
            let z = 0.0f32.var();
            outline(|| {
                let buf_x = x.var();
                let buf_y = y.var();
                let x = buf_x.read(tid);
                let y = buf_y.read(tid);
                *z = x + y;
            });
            buf_z.write(tid, z);
        }),
    );
    kernel.dispatch([1024, 1, 1], &z);
    let z_data = z.view(..).copy_to_vec();
    for i in 0..x.len() {
        assert_eq!(z_data[i], (i as f32 + 1000.0 * i as f32));
    }
}
#[test]
fn nested_callable_capture_buffer_var() {
    let device = get_device();
    let x = device.create_buffer::<f32>(1024);
    let y = device.create_buffer::<f32>(1024);
    let z = device.create_buffer::<f32>(1024);
    x.view(..).fill_fn(|i| i as f32);
    y.view(..).fill_fn(|i| 1000.0 * i as f32);
    let kernel = Kernel::<fn(Buffer<f32>)>::new(
        &device,
        &track!(|buf_z| {
            let buf_x = x.var();
            let buf_y = y.var();
            let tid = dispatch_id().x;

            let z = 0.0f32.var();
            outline(|| {
                let x = buf_x.read(tid);
                let y = buf_y.read(tid);
                *z = x + y;
            });
            buf_z.write(tid, z);
        }),
    );
    kernel.dispatch([1024, 1, 1], &z);
    let z_data = z.view(..).copy_to_vec();
    for i in 0..x.len() {
        assert_eq!(z_data[i], (i as f32 + 1000.0 * i as f32));
    }
}
#[test]
#[should_panic]
fn callable_different_device() {
    let device1 = get_device();
    let device2 = get_device();
    let abs = Callable::<fn(Expr<f32>) -> Expr<f32>>::new(
        &device1,
        track!(|x| {
            if x > 0.0 {
                return x;
            }
            -x
        }),
    );
    let _foo = Callable::<fn(Expr<f32>) -> Expr<f32>>::new(&device2, |x| abs.call(x));
}
#[test]
#[should_panic]
fn callable_return_mismatch() {
    let device = get_device();
    let _abs = Callable::<fn(Expr<f32>) -> Expr<f32>>::new(
        &device,
        track!(|x| {
            if x > 0.0 {
                return true.expr();
            }
            -x
        }),
    );
}
#[test]
#[should_panic]
fn callable_return_mismatch2() {
    let device = get_device();
    let _abs = Callable::<fn(Expr<f32>) -> Expr<f32>>::new(
        &device,
        track!(|x| {
            if x > 0.0 {
                return;
            }
            -x
        }),
    );
}

#[test]
#[should_panic]
fn callable_return_void_mismatch() {
    let device = get_device();
    let _abs = Callable::<fn(Var<f32>)>::new(
        &device,
        track!(|x| {
            if x > 0.0 {
                return true.expr();
            }
            *x = -x;
        }),
    );
}
#[test]
#[should_panic]
#[tracked]
fn illegal_scope_sharing() {
    let device = get_device();
    let tid = RefCell::new(None);
    Kernel::<fn()>::new(&device, &|| {
        let i = dispatch_id().x;
        if i % 2 == 0 {
            *tid.borrow_mut() = Some(i + 1);
        }
        let _v = tid.borrow().unwrap() + 1;
    });
}
#[test]
#[should_panic]
fn callable_illegal_sharing() {
    let device = get_device();
    let tid = RefCell::new(None);
    Kernel::<fn()>::new(&device, &|| {
        let i = dispatch_id().x;
        *tid.borrow_mut() = Some(i);
    });
    Kernel::<fn()>::new(
        &device,
        &track!(|| {
            let tid = tid.borrow().unwrap();
            let _i = dispatch_id().x + tid;
        }),
    );
}
#[test]
fn callable_early_return() {
    let device = get_device();
    let abs = Callable::<fn(Expr<f32>) -> Expr<f32>>::new(
        &device,
        track!(|x| {
            if x > 0.0 {
                return x;
            }
            -x
        }),
    );
    let x = device.create_buffer::<f32>(1024);
    let mut rng = StdRng::seed_from_u64(0);
    x.fill_fn(|_| rng.gen());
    let y = device.create_buffer::<f32>(1024);
    Kernel::<fn()>::new(&device, &|| {
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
fn var_copy_inner() {
    let device = get_device();
    let write = Callable::<fn(BufferVar<u32>, Expr<u32>, Var<u32>)>::new(
        &device,
        track!(|buf: BufferVar<u32>, i: Expr<u32>, v: Var<u32>| {
            buf.write(i, v.load());
            let u = v.var();
            *u += 1;
        }),
    );
    let add = Callable::<fn(Expr<u32>, Expr<u32>) -> Expr<u32>>::new(&device, |a, b| track!(a + b));
    let x = device.create_buffer::<u32>(1024);
    let y = device.create_buffer::<u32>(1024);
    let z = device.create_buffer::<u32>(1024);
    let w = device.create_buffer::<u32>(1024);
    x.view(..).fill_fn(|i| i as u32);
    y.view(..).fill_fn(|i| 1000 * i as u32);
    let kernel = Kernel::<fn(Buffer<u32>)>::new(
        &device,
        &track!(|buf_z| {
            let buf_x = x.var();
            let buf_y = y.var();
            let buf_w = w.var();
            let tid = dispatch_id().x;
            let x = buf_x.read(tid);
            let y = buf_y.read(tid);
            let z = add.call(x, y).var();
            write.call(buf_z, tid, z);
            buf_w.write(tid, z);
        }),
    );
    kernel.dispatch([1024, 1, 1], &z);
    let z_data = z.view(..).copy_to_vec();
    let w_data = w.view(..).copy_to_vec();
    for i in 0..z_data.len() {
        assert_eq!(z_data[i], (i + 1000 * i) as u32);
        assert_eq!(w_data[i], (i + 1000 * i) as u32);
    }
}
#[test]
fn callable() {
    let device = get_device();
    let write = Callable::<fn(BufferVar<u32>, Expr<u32>, Var<u32>)>::new(
        &device,
        |buf: BufferVar<u32>, i: Expr<u32>, v: Var<u32>| {
            buf.write(i, v.load());
            track!(*v += 1);
        },
    );
    let add = Callable::<fn(Expr<u32>, Expr<u32>) -> Expr<u32>>::new(&device, |a, b| track!(a + b));
    let x = device.create_buffer::<u32>(1024);
    let y = device.create_buffer::<u32>(1024);
    let z = device.create_buffer::<u32>(1024);
    let w = device.create_buffer::<u32>(1024);
    x.view(..).fill_fn(|i| i as u32);
    y.view(..).fill_fn(|i| 1000 * i as u32);
    let kernel = Kernel::<fn(Buffer<u32>)>::new(
        &device,
        &track!(|buf_z| {
            let buf_x = x.var();
            let buf_y = y.var();
            let buf_w = w.var();
            let tid = dispatch_id().x;
            let x = buf_x.read(tid);
            let y = buf_y.read(tid);
            let z = add.call(x, y).var();
            write.call(buf_z, tid, z);
            buf_w.write(tid, z);
        }),
    );
    kernel.dispatch([1024, 1, 1], &z);
    let z_data = z.view(..).copy_to_vec();
    let w_data = w.view(..).copy_to_vec();
    for i in 0..z_data.len() {
        assert_eq!(z_data[i], (i + 1000 * i) as u32);
        assert_eq!(w_data[i], (i + 1000 * i) as u32 + 1);
    }
}
#[test]
fn callable_capture() {
    let device = get_device();

    let add = Callable::<fn(Expr<u32>, Expr<u32>) -> Expr<u32>>::new(&device, |a, b| track!(a + b));
    let x = device.create_buffer::<u32>(1024);
    let y = device.create_buffer::<u32>(1024);
    let z = device.create_buffer::<u32>(1024);
    let w = device.create_buffer::<u32>(1024);
    x.view(..).fill_fn(|i| i as u32);
    y.view(..).fill_fn(|i| 1000 * i as u32);
    let write = Callable::<fn(Expr<u32>, Var<u32>)>::new(&device, |i, v| {
        z.write(i, v);
        track!(*v += 1);
    });
    let kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
            let buf_x = x.var();
            let buf_y = y.var();
            let buf_w = w.var();
            let tid = dispatch_id().x;
            let x = buf_x.read(tid);
            let y = buf_y.read(tid);
            let z = add.call(x, y).var();
            write.call(tid, z);
            buf_w.write(tid, z);
        }),
    );
    kernel.dispatch([1024, 1, 1]);
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
    let kernel = Kernel::<fn()>::new_with_options(
        &device,
        KernelBuildOptions {
            name: Some("vec_cast".to_string()),
            ..KernelBuildOptions::default()
        },
        &|| {
            let f = f.var();
            let i = i.var();
            let tid = dispatch_id().x;
            let v = f.read(tid);
            i.write(tid, v.as_int2());
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
    if device.name() == "dx" {
        return;
    }
    let x: Buffer<bool> = device.create_buffer(1024);
    let y: Buffer<bool> = device.create_buffer(1024);
    let and: Buffer<bool> = device.create_buffer(1024);
    let or: Buffer<bool> = device.create_buffer(1024);
    let xor: Buffer<bool> = device.create_buffer(1024);
    let not: Buffer<bool> = device.create_buffer(1024);
    let mut rng = rand::thread_rng();
    x.view(..).fill_fn(|_| rng.gen());
    y.view(..).fill_fn(|_| rng.gen());
    let kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
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
        }),
    );
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
    if device.name() == "dx" {
        return;
    }
    let x: Buffer<Bool2> = device.create_buffer(1024);
    let y: Buffer<Bool2> = device.create_buffer(1024);
    let and: Buffer<Bool2> = device.create_buffer(1024);
    let or: Buffer<Bool2> = device.create_buffer(1024);
    let xor: Buffer<Bool2> = device.create_buffer(1024);
    let not: Buffer<Bool2> = device.create_buffer(1024);
    let mut rng = rand::thread_rng();
    x.view(..).fill_fn(|_| Bool2::new(rng.gen(), rng.gen()));
    y.view(..).fill_fn(|_| Bool2::new(rng.gen(), rng.gen()));
    let kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
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
        }),
    );
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
fn test_var_replace() {
    let device = get_device();
    let xs: Buffer<Int4> = device.create_buffer(1024);
    let ys: Buffer<Int4> = device.create_buffer(1024);
    let kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
            let tid = dispatch_id().x;
            let x = xs.var().read(tid).var();
            *x = Int4::expr(1, 2, 3, 4);
            let y = **x;
            *x.y = 10;
            *x.z = 20;
            xs.write(tid, x);
            ys.write(tid, y);
        }),
    );
    kernel.dispatch([1024, 1, 1]);
    let xs = xs.view(..).copy_to_vec();
    let ys = ys.view(..).copy_to_vec();
    for i in 0..1024 {
        assert_eq!(xs[i].x, 1);
        assert_eq!(xs[i].y, 10);
        assert_eq!(xs[i].z, 20);
        assert_eq!(xs[i].w, 4);
        assert_eq!(ys[i].x, 1);
        assert_eq!(ys[i].y, 2);
        assert_eq!(ys[i].z, 3);
        assert_eq!(ys[i].w, 4);
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
    let kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
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
        }),
    );
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
    let kernel = Kernel::<fn()>::new(&device, &|| {
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
    if device.name() == "dx" {
        return;
    }
    let x: Buffer<i32> = device.create_buffer(1024);
    let even: Buffer<bool> = device.create_buffer(1024);
    x.view(..).fill_fn(|i| i as i32);
    let kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
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
        }),
    );
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
    let kernel = Kernel::<fn()>::new(&device, &|| {
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
    let kernel = Kernel::<fn()>::new(&device, &|| {
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
    let kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
            let buf_x = x.var();
            let tid = dispatch_id().x;
            let arr = Var::<[i32; 4]>::zeroed();
            let i = i32::var_zeroed();
            while i < 4 {
                arr.write(i.as_u32(), tid.as_i32() + i);
                *i += 1;
            }
            buf_x.write(tid, arr);
        }),
    );
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
    let kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
            let buf_x = x.var();
            let tid = dispatch_id().x;
            let arr = Var::<[i32; 4]>::zeroed();
            for_range(0..4u32, |i| {
                arr.write(i, tid.as_i32() + i.as_i32());
            });
            buf_x.write(tid, arr);
        }),
    );
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
    let kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
            let buf_x = x.var();
            let tid = dispatch_id().x;
            let arr = Var::<[i32; 4]>::zeroed();
            for_range(0..6u32, |_| {
                for_range(0..4u32, |i| {
                    arr.write(i, arr.read(i) + tid.as_i32() + i.as_i32());
                });
            });
            buf_x.write(tid, arr);
        }),
    );
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
    let kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
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
        }),
    );
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
    let kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
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
        }),
    );
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
    let kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
            let buf_x = x.var();
            let tid = dispatch_id().x;
            let arr = Var::<[i32; 4]>::zeroed();
            let i = i32::var_zeroed();
            while i < 4 {
                arr.write(i.as_u32(), tid.as_i32() + i);
                *i += 1;
            }
            buf_x.write(tid, arr);
        }),
    );
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
    let shader = Kernel::<fn()>::new(
        &device,
        &track!(|| {
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
        }),
    );
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
    let shader = Kernel::<fn(Float3)>::new(
        &device,
        &track!(|v: Expr<Float3>| {
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
        }),
    );
    shader.dispatch([x.len() as u32, 1, 1], &Float3::new(1.0, 2.0, 3.0));
    let mut sum_data = vec![0.0];
    sum.view(..).copy_to(&mut sum_data);
    let actual = sum_data[0];
    let expected = (x.len() as f32 - 1.0) * x.len() as f32 * 0.5 * 6.0;
    assert!((actual - expected).abs() < 1e-4);
}
#[derive(Clone, Copy, Debug, Value, Default)]
#[repr(C)]
struct Big {
    a: [f32; 32],
}
#[test]
fn buffer_u8() {
    let device = get_device();
    if device.name() == "dx" {
        return;
    }
    let buf = device.create_buffer::<u8>(1024);
    let kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
            let tid = dispatch_id().x;
            buf.write(tid, (tid & 0xff).as_u8());
        }),
    );
    kernel.dispatch([1024, 1, 1]);
}
#[test]
fn buffer_view_copy() {
    let device = get_device();
    let n = 1024;
    let buf = device.create_buffer::<f32>(n);
    let first_half = buf.view(0..n / 2);
    let second_half = buf.view(n / 2..);
    first_half.fill_fn(|i| i as f32);
    second_half.fill_fn(|i| -(i as f32));
    let data = buf.copy_to_vec();
    for i in 0..n {
        if i < n / 2 {
            assert_eq!(data[i], i as f32);
        } else {
            assert_eq!(data[i], -((i - n / 2) as f32));
        }
    }
}
#[test]
fn buffer_view() {
    let device = get_device();
    let n = 1024;
    let buf = device.create_buffer::<f32>(n);
    let first_half = buf.view(0..n / 2);
    let second_half = buf.view(n / 2..);
    let kernel = Kernel::<fn(Buffer<f32>, Buffer<f32>)>::new(
        &device,
        &track!(|a, b| {
            let tid = dispatch_id().x;
            a.write(tid, tid.as_f32());
            b.write(tid, -tid.as_f32());
        }),
    );
    kernel.dispatch([n as u32 / 2, 1, 1], &first_half, &second_half);
    let data = buf.copy_to_vec();
    for i in 0..n {
        if i < n / 2 {
            assert_eq!(data[i], i as f32);
        } else {
            assert_eq!(data[i], -((i - n / 2) as f32));
        }
    }
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
    Kernel::<fn()>::new(
        &device,
        &track!(|| unsafe {
            let buf = buf.var();
            let i0 = i0 as u64;
            let i1 = i1 as u64;
            let i2 = i2 as u64;
            let i3 = i3 as u64;
            let v0 = buf.read_as::<Float3>(i0).var();
            let v1 = buf.read_as::<Big>(i1).var();
            let v2 = buf.read_as::<i32>(i2).var();
            let v3 = buf.read_as::<f32>(i3).var();
            *v0 = Float3::expr(1.0, 2.0, 3.0);
            for_range(0u32..32u32, |i| {
                v1.a.write(i, i.as_f32() * 2.0);
            });
            *v2 = 1i32.expr();
            *v3 = 2.0.expr();
            buf.write_as::<Float3>(i0, v0.load());
            buf.write_as::<Big>(i1, v1.load());
            buf.write_as::<i32>(i2, v2.load());
            buf.write_as::<f32>(i3, v3.load());
        }),
    )
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
#[allow(unused_assignments)]
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
    Kernel::<fn(ByteBuffer)>::new(
        &device,
        &track!(|out: ByteBufferVar| unsafe {
            let heap = heap.var();
            let buf = heap.byte_address_buffer(0u32);
            let i0 = i0 as u64;
            let i1 = i1 as u64;
            let i2 = i2 as u64;
            let i3 = i3 as u64;
            let v0 = buf.read_as::<Float3>(i0).var();
            let v1 = buf.read_as::<Big>(i1).var();
            let v2 = buf.read_as::<i32>(i2).var();
            let v3 = buf.read_as::<f32>(i3).var();
            *v0 = Float3::expr(1.0, 2.0, 3.0);
            for_range(0u32..32u32, |i| {
                v1.a.write(i, i.as_f32() * 2.0);
            });
            *v2 = 1i32.expr();
            *v3 = 2.0.expr();
            out.write_as::<Float3>(i0, v0.load());
            out.write_as::<Big>(i1, v1.load());
            out.write_as::<i32>(i2, v2.load());
            out.write_as::<f32>(i3, v3.load());
        }),
    )
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

#[test]
fn is_finite() {
    let device = get_device();
    let x = device.create_buffer::<f32>(1024);
    x.fill_fn(|i| i as f32);
    let kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
            let tid = dispatch_id().x;
            let x = x.read(tid);
            lc_assert!(x.is_finite());
            lc_assert!(!x.is_nan());
            lc_assert!(!x.is_infinite());
        }),
    );
    kernel.dispatch([1024, 1, 1]);
}
#[test]
fn is_infinite() {
    let device = get_device();
    let x = device.create_buffer::<f32>(1024);
    x.fill_fn(|i| 1.0 + i as f32);
    let kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
            let tid = dispatch_id().x;
            let x = x.read(tid) / 0.0;
            lc_assert!(!x.is_finite());
            lc_assert!(!x.is_nan());
            lc_assert!(x.is_infinite());
        }),
    );
    kernel.dispatch([1024, 1, 1]);
}
#[test]
fn is_nan() {
    let device = get_device();
    let x = device.create_buffer::<f32>(1024);
    x.fill_fn(|i| i as f32);
    let kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
            let tid = dispatch_id().x;
            let x = x.read(tid) / 0.0 - x.read(tid) / 0.0;
            lc_assert!(!x.is_finite());
            lc_assert!(x.is_nan());
            lc_assert!(!x.is_infinite());
        }),
    );
    kernel.dispatch([1024, 1, 1]);
}
// #[derive(Clone, Copy, Debug, Value, PartialEq)]
// #[repr(C)]
// #[value_new(pub)]
// pub struct CustomAdd {
//     i: f32,
// }
// impl std::ops::Add<Expr<CustomAdd>> for  CustomAddExpr {
//     type Output = Expr<CustomAdd>;
//     fn add(self, rhs: Expr<CustomAdd>) -> Self::Output {
//         todo!()
//     }
// }
// impl AddExpr<CustomAddExpr> for CustomAddExpr {
//     type Output = Expr<CustomAdd>;
//     #[tracked]
//     fn add(self, rhs: CustomAddExpr) -> Expr<CustomAdd> {
//         let rhs = rhs.as_expr_from_proxy();
//         let self_ = self.self_.var();
//         *self_.i += rhs.i;
//         **self_
//     }
// }
// impl AddMaybeExpr<CustomAddExpr, luisa_compute::lang::types::ExprType> for Expr<CustomAdd> {
//     type Output = Expr<CustomAdd>;
//     fn __add(self, rhs: CustomAddExpr) -> Self::Output {
//         self.add(rhs)
//     }
// }
// fn custom_add(a: Expr<CustomAdd>, b: Expr<CustomAdd>) -> Expr<CustomAdd> {
//     track!(*a + b)
// }

#[derive(Clone, Copy, Debug, Value, Soa, PartialEq)]
#[repr(C)]
#[value_new(pub)]
pub struct Foo {
    i: u32,
    v: Float2,
    a: [i32; 4],
}
#[derive(Clone, Copy, Debug, Value, Soa, PartialEq)]
#[repr(C)]
#[value_new(pub)]
pub struct Bar {
    i: u32,
    v: Float2,
    a: [i32; 4],
    f: Foo,
}
#[test]
fn soa() {
    let device = get_device();
    let mut rng = thread_rng();
    let bars = device.create_buffer_from_fn(1024, |_| Bar {
        i: rng.gen(),
        v: Float2::new(rng.gen(), rng.gen()),
        a: [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
        f: Foo {
            i: rng.gen(),
            v: Float2::new(rng.gen(), rng.gen()),
            a: [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
        },
    });
    let bars_soa = device.create_soa_buffer::<Bar>(1024);
    bars_soa.copy_from_buffer(&bars);
    let also_bars = device.create_buffer(1024);
    bars_soa.copy_to_buffer(&also_bars);
    let bars_data = bars.view(..).copy_to_vec();
    let also_bars_data = also_bars.view(..).copy_to_vec();
    assert_eq!(bars_data, also_bars_data);
}
#[test]
fn soa_view() {
    let device = get_device();
    let mut rng = thread_rng();
    let bars = device.create_buffer_from_fn(1024, |_| Bar {
        i: rng.gen(),
        v: Float2::new(rng.gen(), rng.gen()),
        a: [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
        f: Foo {
            i: rng.gen(),
            v: Float2::new(rng.gen(), rng.gen()),
            a: [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
        },
    });
    let bars_soa = device.create_soa_buffer::<Bar>(2048);
    bars_soa.view(..1024).copy_from_buffer(&bars);
    bars_soa.view(1024..2048).copy_from_buffer(&bars);

    let also_bars = device.create_buffer(1024);
    bars_soa.view(..1024).copy_to_buffer(&also_bars);
    let bars_data = bars.view(..).copy_to_vec();
    let also_bars_data = also_bars.view(..).copy_to_vec();
    assert_eq!(bars_data, also_bars_data);

    let also_bars = device.create_buffer(1024);
    bars_soa.view(1024..2048).copy_to_buffer(&also_bars);
    let bars_data = bars.view(..).copy_to_vec();
    let also_bars_data = also_bars.view(..).copy_to_vec();
    assert_eq!(bars_data, also_bars_data);
}
#[test]
fn atomic() {
    let device = get_device();
    let mut rng = thread_rng();
    let foos = device.create_buffer_from_fn(1024, |_| Foo {
        i: rng.gen(),
        v: Float2::new(rng.gen(), rng.gen()),
        a: [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
    });
    let foo_max_init = Foo {
        i: u32::MIN,
        v: Float2::new(f32::MIN, f32::MIN),
        a: [i32::MIN; 4],
    };
    let foo_min_init = Foo {
        i: u32::MAX,
        v: Float2::new(f32::MAX, f32::MAX),
        a: [i32::MAX; 4],
    };
    let foo_max = device.create_buffer_from_slice(&[foo_max_init]);
    let foo_min = device.create_buffer_from_slice(&[foo_min_init]);
    let kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
            let i = dispatch_id().x;
            let foos = foos.var();
            let foo = foos.read(i);
            let foo_max = foo_max.var().atomic_ref(0);
            let foo_min = foo_min.var().atomic_ref(0);
            foo_max.i.fetch_max(foo.i);
            foo_max.v.x.fetch_max(foo.v.x);
            foo_max.v.y.fetch_max(foo.v.y);
            for i in 0..4u32 {
                foo_max.a[i].fetch_max(foo.a[i]);
            }
            foo_min.i.fetch_min(foo.i);
            foo_min.v.x.fetch_min(foo.v.x);
            foo_min.v.y.fetch_min(foo.v.y);
            for i in 0..4u32 {
                foo_min.a[i].fetch_min(foo.a[i]);
            }
        }),
    );
    kernel.dispatch([foos.len() as u32, 1, 1]);
    let foos = foos.view(..).copy_to_vec();
    let foo_min = foo_min.view(..).copy_to_vec()[0];
    let foo_max = foo_max.view(..).copy_to_vec()[0];
    let mut expected_foo_max = foo_max_init;
    let mut expected_foo_min = foo_min_init;
    for foo in foos {
        expected_foo_max.i = expected_foo_max.i.max(foo.i);
        expected_foo_max.v.x = expected_foo_max.v.x.max(foo.v.x);
        expected_foo_max.v.y = expected_foo_max.v.y.max(foo.v.y);
        for i in 0..4 {
            expected_foo_max.a[i] = expected_foo_max.a[i].max(foo.a[i]);
        }
        expected_foo_min.i = expected_foo_min.i.min(foo.i);
        expected_foo_min.v.x = expected_foo_min.v.x.min(foo.v.x);
        expected_foo_min.v.y = expected_foo_min.v.y.min(foo.v.y);
        for i in 0..4 {
            expected_foo_min.a[i] = expected_foo_min.a[i].min(foo.a[i]);
        }
    }
    assert_eq!(foo_max, expected_foo_max);
    assert_eq!(foo_min, expected_foo_min);
}

#[test]
fn dyn_callable() {
    let device = get_device();
    let add = DynCallable::<fn(DynExpr, DynExpr) -> DynExpr>::new(
        &device,
        Box::new(|a: DynExpr, b: DynExpr| -> DynExpr {
            if let Some(a) = a.downcast::<f32>() {
                let b = b.downcast::<f32>().unwrap();
                return DynExpr::new(track!(a + b));
            } else if let Some(a) = a.downcast::<i32>() {
                let b = b.downcast::<i32>().unwrap();
                return DynExpr::new(track!(a + b));
            } else {
                unreachable!()
            }
        }),
    );
    let x = device.create_buffer::<f32>(1024);
    let y = device.create_buffer::<f32>(1024);
    let z = device.create_buffer::<f32>(1024);
    let w = device.create_buffer::<i32>(1024);
    x.view(..).fill_fn(|i| i as f32);
    y.view(..).fill_fn(|i| 1000.0 * i as f32);
    let kernel = Kernel::<fn(Buffer<f32>)>::new(
        &device,
        &track!(|buf_z| {
            let buf_x = x.var();
            let buf_y = y.var();
            let tid = dispatch_id().x;
            let x = buf_x.read(tid);
            let y = buf_y.read(tid);

            buf_z.write(tid, add.call(x.into(), y.into()).get::<f32>());
            w.var().write(
                tid,
                add.call(x.as_::<i32>().into(), y.as_::<i32>().into())
                    .get::<i32>(),
            );
        }),
    );
    kernel.dispatch([1024, 1, 1], &z);
    let z_data = z.view(..).copy_to_vec();
    let w_data = w.view(..).copy_to_vec();
    for i in 0..1024 {
        assert_eq!(z_data[i], i as f32 + 1000.0 * i as f32);
        assert_eq!(w_data[i], i as i32 + 1000 * i as i32);
    }
}

#[test]
fn dispatch_async() {
    let device = get_device();
    let x = device.create_buffer::<f32>(1024);
    x.fill_fn(|i| i as f32);
    let kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
            for _ in 0..10000000 {
                let buf_x = x.var();
                let tid = dispatch_id().x;
                let x = buf_x.read(tid);
                buf_x.write(tid, x + 1.0);
            }
        }),
    );
    let s = device.default_stream().scope();
    s.submit([
        kernel.dispatch_async([1024, 1, 1]),
        kernel.dispatch_async([1024, 1, 1]),
    ]);
    drop(kernel);
}

#[test]
fn buffer_size() {
    let device = get_device();
    let x = device.create_buffer::<Big>(1024);
    x.fill(Big::default());
    let out = device.create_buffer::<u32>(1);
    out.fill(2);
    device
        .create_kernel::<fn()>(&track!(|| {
            lc_assert!((x.len() as u64).eq(x.var().len_expr()));
            lc_assert!((x.len() as u32).eq(x.var().len_expr().cast_u32()));
            let tid = dispatch_id().x;
            if tid == 0 {
                out.write(0, x.var().len_expr_u32());
            }
        }))
        .dispatch([1024, 1, 1]);
    let out = out.view(..).copy_to_vec();
    assert_eq!(out[0], 1024);
}

#[test]
#[tracked]
fn test_tracked() {
    let v = true;

    if v {
    } else {
        panic!();
    }

    let v = false;

    if !v {
    } else {
        panic!();
    }
}
