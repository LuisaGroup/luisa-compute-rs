use std::env::current_exe;

use luisa::Context;
use luisa_compute as luisa;

fn main() {
    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_device("cpu").unwrap();
    let x = device.create_buffer::<f32>(128).unwrap();
    let sum = device.create_buffer::<f32>(1).unwrap();
    x.view(..).fill_fn(|i| i as f32);
    sum.view(..).fill(0.0);
    let shader = device
        .create_kernel::<()>(&|| {
            let buf_x = x.var();
            let buf_sum = sum.var();
            let tid = luisa::dispatch_id().x();
            buf_sum.atomic_fetch_add(0, buf_x.read(tid));
        })
        .unwrap();
    shader.dispatch([x.len() as u32, 1, 1]).unwrap();
    let mut sum_data = vec![0.0];
    sum.view(..).copy_to(&mut sum_data);
    println!(
        "actual: {:?}, expected: {}",
        &sum_data[0],
        (x.len() as f32 - 1.0) * x.len() as f32 * 0.5
    );
}
