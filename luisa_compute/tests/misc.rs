use std::{ops::Range, sync::Once};

use luisa::prelude::*;
use luisa_compute as luisa;
use rand::prelude::*;
use rayon::{
    prelude::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
static START: Once = Once::new();
fn init_once() {
    START.call_once(|| {
        init();
    });
}

#[test]
fn vec_cast() {
    init_once();
    let device = create_cpu_device().unwrap();
    let f: Buffer<Vec2> = device.create_buffer(1024).unwrap();
    let i: Buffer<IVec2> = device.create_buffer(1024).unwrap();
    f.view(..)
        .fill_fn(|i| Vec2::new(i as f32 + 0.5, i as f32 + 1.5));
    let kernel = device
        .create_kernel::<()>(&|| {
            let f = f.var();
            let i = i.var();
            let tid = dispatch_id().x();
            let v = f.read(tid);
            i.write(tid, v.as_ivec2());
        })
        .unwrap();
    kernel.dispatch([1024, 1, 1]).unwrap();
    let mut i_data = vec![glam::IVec2::ZERO.into(); 1024];
    i.view(..).copy_to(&mut i_data);
    for i in 0..1024 {
        assert_eq!(i_data[i].x, i as i32);
        assert_eq!(i_data[i].y, i as i32 + 1);
    }
}
#[test]
fn vec_bit_minmax() {
    init_once();
    let device = create_cpu_device().unwrap();
    let x: Buffer<IVec2> = device.create_buffer(1024).unwrap();
    let y: Buffer<IVec2> = device.create_buffer(1024).unwrap();
    let z: Buffer<IVec2> = device.create_buffer(1024).unwrap();
    let and: Buffer<IVec2> = device.create_buffer(1024).unwrap();
    let or: Buffer<IVec2> = device.create_buffer(1024).unwrap();
    let xor: Buffer<IVec2> = device.create_buffer(1024).unwrap();
    let not: Buffer<IVec2> = device.create_buffer(1024).unwrap();
    let min = device.create_buffer::<IVec2>(1024).unwrap();
    let max = device.create_buffer::<IVec2>(1024).unwrap();
    let clamp = device.create_buffer::<IVec2>(1024).unwrap();
    let mut rng = rand::thread_rng();
    x.view(..).fill_fn(|_| IVec2::new(rng.gen(), rng.gen()));
    y.view(..).fill_fn(|_| IVec2::new(rng.gen(), rng.gen()));
    z.view(..).fill_fn(|_| IVec2::new(rng.gen(), rng.gen()));
    let kernel = device
        .create_kernel::<()>(&|| {
            let tid = dispatch_id().x();
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
            min.write(tid, x.min(y));
            max.write(tid, x.max(y));
            clamp.write(tid, z.clamp(x.min(y), x.max(y)));
        })
        .unwrap();
    kernel.dispatch([1024, 1, 1]).unwrap();
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
    init_once();
    let device = create_cpu_device().unwrap();
    let v2: Buffer<IVec2> = device.create_buffer(1024).unwrap();
    let v3: Buffer<IVec3> = device.create_buffer(1024).unwrap();
    v2.view(..)
        .fill_fn(|i| IVec2::new(i as i32 + 0, i as i32 + 1));
    let kernel = device
        .create_kernel::<()>(&|| {
            let v2 = v2.var();
            let v3 = v3.var();
            let tid = dispatch_id().x();
            let v = v2.read(tid);
            v3.write(tid, v.xyx());
        })
        .unwrap();
    kernel.dispatch([1024, 1, 1]).unwrap();
    let mut i_data = vec![glam::IVec3::ZERO.into(); 1024];
    v3.view(..).copy_to(&mut i_data);
    for i in 0..1024 {
        assert_eq!(i_data[i].x, i as i32);
        assert_eq!(i_data[i].y, i as i32 + 1);
        assert_eq!(i_data[i].z, i as i32);
    }
}

#[test]
fn if_phi() {
    init_once();
    let device = create_cpu_device().unwrap();
    let x: Buffer<i32> = device.create_buffer(1024).unwrap();
    let even: Buffer<bool> = device.create_buffer(1024).unwrap();
    x.view(..).fill_fn(|i| i as i32);
    let kernel = device
        .create_kernel::<()>(&|| {
            let x = x.var();
            let even = even.var();
            let tid = dispatch_id().x();
            let v = x.read(tid);
            let result = if_!((v % 2).cmpeq(0), { Bool::from(true) }, else { Bool::from(false) });
            even.write(tid, result);
        })
        .unwrap();
    kernel.dispatch([1024, 1, 1]).unwrap();
    let mut i_data = vec![false; 1024];
    even.view(..).copy_to(&mut i_data);
    for i in 0..1024 {
        assert_eq!(i_data[i], i % 2 == 0);
    }
}

#[test]
fn switch_phi() {
    init_once();
    let device = create_cpu_device().unwrap();
    let x: Buffer<i32> = device.create_buffer(1024).unwrap();
    let y: Buffer<i32> = device.create_buffer(1024).unwrap();
    let z: Buffer<f32> = device.create_buffer(1024).unwrap();
    x.view(..).fill_fn(|i| i as i32);
    let kernel = device
        .create_kernel::<()>(&|| {
            let buf_x = x.var();
            let buf_y = y.var();
            let buf_z = z.var();
            let tid = dispatch_id().x();
            let x = buf_x.read(tid);
            let (y, z) = switch::<(Expr<i32>, Expr<f32>)>(x)
                .case(0, || (Int32::from(0), Float32::from(1.0)))
                .case(1, || (Int32::from(1), Float32::from(2.0)))
                .case(2, || (Int32::from(2), Float32::from(3.0)))
                .default(|| (Int32::from(3), Float32::from(4.0)))
                .finish();
            buf_y.write(tid, y);
            buf_z.write(tid, z);
        })
        .unwrap();
    kernel.dispatch([1024, 1, 1]).unwrap();
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
    init_once();
    let device = create_cpu_device().unwrap();
    let x: Buffer<i32> = device.create_buffer(1024).unwrap();
    let y: Buffer<i32> = device.create_buffer(1024).unwrap();
    let z: Buffer<f32> = device.create_buffer(1024).unwrap();
    x.view(..).fill_fn(|i| i as i32 % 3);
    let kernel = device
        .create_kernel::<()>(&|| {
            let buf_x = x.var();
            let buf_y = y.var();
            let buf_z = z.var();
            let tid = dispatch_id().x();
            let x = buf_x.read(tid);
            let (y, z) = switch::<(Expr<i32>, Expr<f32>)>(x)
                .case(0, || (Int32::from(0), Float32::from(1.0)))
                .case(1, || (Int32::from(1), Float32::from(2.0)))
                .case(2, || (Int32::from(2), Float32::from(3.0)))
                .finish();
            buf_y.write(tid, y);
            buf_z.write(tid, z);
        })
        .unwrap();
    kernel.dispatch([1024, 1, 1]).unwrap();
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
    init_once();
    let device = create_cpu_device().unwrap();
    let x: Buffer<[i32; 4]> = device.create_buffer(1024).unwrap();
    let kernel = device
        .create_kernel::<()>(&|| {
            let buf_x = x.var();
            let tid = dispatch_id().x();
            let arr = local_zeroed::<[i32; 4]>();
            let i = local_zeroed::<i32>();
            while_!(i.load().cmplt(4), {
                arr.write(i.load().uint(), tid.int() + i.load());
                i.store(i.load() + 1);
            });
            buf_x.write(tid, arr.load());
        })
        .unwrap();
    kernel.dispatch([1024, 1, 1]).unwrap();
    let x_data = x.view(..).copy_to_vec();
    for i in 0..1024 {
        assert_eq!(
            x_data[i],
            [i as i32, i as i32 + 1, i as i32 + 2, i as i32 + 3]
        );
    }
}
