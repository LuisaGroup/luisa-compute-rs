use luisa::prelude::*;
use luisa_compute as luisa;
use std::env::current_exe;

fn main() {
    luisa::init_logger_verbose();
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() <= 2,
        "Usage: {} <backend>. <backend>: cpu, cuda, dx, metal, remote",
        args[0]
    );

    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_device(if args.len() == 2 {
        args[1].as_str()
    } else {
        "cpu"
    });
    let x = device.create_buffer::<f32>(1024);
    let y = device.create_buffer::<f32>(1024);
    let z = device.create_buffer::<f32>(1024);
    x.view(..).fill_fn(|i| i as f32);
    y.view(..).fill_fn(|i| 1000.0 * i as f32);

    let kernel = device.create_kernel_with_options::<fn(Buffer<f32>)>(
        KernelBuildOptions {
            name: Some("vecadd".into()),
            ..Default::default()
        },
        &track!(|buf_z| {
            // z is pass by arg
            let buf_x = &x; // x and y are captured
            let buf_y = &y;
            let tid = dispatch_id().x;
            let x = buf_x.read(tid);
            let y = buf_y.read(tid);
            buf_z.write(tid, x + y);
        }),
    );
    let mut z_data = vec![123.0f32; 1024];

    unsafe {
        let s = device.default_stream().scope();
        let z_data_ptr = z_data.as_mut_ptr();
        s.submit([
            z.copy_from_async(std::slice::from_raw_parts_mut(z_data_ptr, 1024)),
            kernel.dispatch_async([1024, 1, 1], &z),
            z.copy_to_async(std::slice::from_raw_parts_mut(z_data_ptr, 1024)),
            z.copy_from_async(std::slice::from_raw_parts_mut(z_data_ptr, 1024)),
            z.copy_to_buffer_async(&x)
        ]);
    }

    // this should produce the expected behavior
    // unsafe {
    //     let z_data_ptr = z_data.as_mut_ptr();

    //     z.copy_from(std::slice::from_raw_parts_mut(z_data_ptr, 1024));
    //     kernel.dispatch([1024, 1, 1], &z);
    //     z.copy_to(std::slice::from_raw_parts_mut(z_data_ptr, 1024));
    //     z.copy_from(std::slice::from_raw_parts_mut(z_data_ptr, 1024));
    //     z.copy_to_buffer(&x);
    // }

    println!("{:?}", &z_data[0..16]);
    let x_data = x.copy_to_vec();
    println!("{:?}", &x_data[0..16]);
}
