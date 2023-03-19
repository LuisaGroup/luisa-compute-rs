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
    let bindless = device.create_bindless_array(2).unwrap();
    bindless.emplace_buffer_async(0, &x);
    bindless.emplace_buffer_async(1, &y);
    bindless.update();
    let shader = device
        .create_shader::<(Buffer<f32>,)>(&|buf_z| {
            let bindless = bindless.var();
            let tid = dispatch_id().x();
            let buf_x = bindless.buffer::<f32>(Uint::from(0));
            let buf_y = bindless.buffer::<f32>(Uint::from(1));
            let x = buf_x.read(tid).uint().float();
            let y = buf_y.read(tid);
            buf_z.write(tid, x + y);
        })
        .unwrap();
    shader.dispatch([1024, 1, 1], &z).unwrap();
    let mut z_data = vec![0.0; 1024];
    z.view(..).copy_to(&mut z_data);
    println!("{:?}", &z_data[0..16]);
}
