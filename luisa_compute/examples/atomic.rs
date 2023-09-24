use std::env::current_exe;

use luisa::prelude::*;
use luisa_compute as luisa;

fn main() {
    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_device("cpu");
    let x = device.create_buffer::<f32>(128);
    let sum = device.create_buffer::<f32>(1);
    x.view(..).fill_fn(|i| i as f32);
    sum.view(..).fill(0.0);
    let shader = Kernel::<fn()>::new(
        &device,
        &track!(|| {
            let buf_x = x.var();
            let buf_sum = sum.var();
            let tid = dispatch_id().x;
            buf_sum.atomic_fetch_add(0, buf_x.read(tid));
        }),
    );
    shader.dispatch([x.len() as u32, 1, 1]);
    let mut sum_data = vec![0.0];
    sum.view(..).copy_to(&mut sum_data);
    println!(
        "actual: {:?}, expected: {}",
        &sum_data[0],
        (x.len() as f32 - 1.0) * x.len() as f32 * 0.5
    );
}
