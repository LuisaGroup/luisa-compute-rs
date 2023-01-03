use luisa::prelude::*;
use luisa_compute as luisa;

#[derive(Clone, Copy, Value, Debug)]
#[repr(C)]
pub struct MyAddArgs {
    pub x: f32,
    pub y: f32,
    pub result: f32,
}

fn main() {
    init();
    let device = create_cpu_device().unwrap();
    let x = device.create_buffer::<f32>(1024).unwrap();
    let y = device.create_buffer::<f32>(1024).unwrap();
    let z = device.create_buffer::<f32>(1024).unwrap();
    x.view(..).fill_fn(|i| i as f32);
    y.view(..).fill_fn(|i| 1000.0 * i as f32);
    let my_add = CpuFn::new(|args: &mut MyAddArgs| {
        args.result = args.x + args.y;
    });
    let my_print = CpuFn::new(|tid: &mut u32| {
        if *tid == 0 {
            println!("Hello from thread 0!");
        }
    });
    let kernel = device
        .create_kernel(wrap_fn!(1, |buf_z: BufferVar<f32>| {
            // z is pass by arg
            let buf_x = x.var(); // x and y are captured
            let buf_y = y.var();
            let tid = dispatch_id().x();
            let x = buf_x.read(tid);
            let y = buf_y.read(tid);
            let args = MyAddArgsExpr::new(x, y, Float32::zero());
            let result = my_add.call(args);
            let _ = my_print.call(tid);
            if_!(tid.cmpeq(0), {
                cpu_dbg!(MyAddArgs, args);
            });
            buf_z.write(tid, result.result());
        }))
        .unwrap();
    kernel.dispatch([1024, 1, 1], &z).unwrap();
    let mut z_data = vec![0.0; 1024];
    z.view(..).copy_to(&mut z_data);
    println!("{:?}", &z_data[0..16]);
}
