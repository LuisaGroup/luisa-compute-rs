use luisa::backend::rust::RustBackend;
use luisa::prelude::*;
use luisa_compute as luisa;

fn main() {
    init();
    let device = create_cpu_device().unwrap();
    let x = device.create_buffer::<f32>(1024).unwrap();
    let y = device.create_buffer::<f32>(1024).unwrap();
    let z = device.create_buffer::<f32>(1024).unwrap();
    x.view(..).fill_fn(|i| i as f32);
    y.view(..).fill_fn(|i| 1000.0 * i as f32);
    let kernel = device
        .create_kernel(wrap_fn!(
            3,
            |buf_x: BufferVar<f32>, buf_y: BufferVar<f32>, buf_z: BufferVar<f32>| {
                let tid = dispatch_id().x();
                let x = buf_x.read(tid);
                let y = buf_y.read(tid);
                buf_z.write(tid, x + y);
            }
        ))
        .unwrap();
    kernel.dispatch([1024, 1, 1], &x, &y, &z).unwrap();
    let mut z_data = vec![0.0; 1024];
    z.view(..).copy_to(&mut z_data);
    println!("{:?}", &z_data[0..16]);
}
