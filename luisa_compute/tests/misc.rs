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
        .create_kernel(wrap_fn!(0, || {
            let f = f.var();
            let i = i.var();
            let tid = dispatch_id().x();
            let v = f.read(tid);
            i.write(tid, v.as_ivec2());
        }))
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
fn vec_permute() {
    init_once();
    let device = create_cpu_device().unwrap();
    let v2: Buffer<IVec2> = device.create_buffer(1024).unwrap();
    let v3: Buffer<IVec3> = device.create_buffer(1024).unwrap();
    v2.view(..)
        .fill_fn(|i| IVec2::new(i as i32 + 0, i as i32 + 1));
    let kernel = device
        .create_kernel(wrap_fn!(0, || {
            let v2 = v2.var();
            let v3 = v3.var();
            let tid = dispatch_id().x();
            let v = v2.read(tid);
            v3.write(tid, v.permute3(0, 1, 0));
        }))
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
