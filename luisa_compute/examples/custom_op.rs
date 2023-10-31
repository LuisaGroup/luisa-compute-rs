use std::env::current_exe;

use luisa::lang::external::CpuFn;
use luisa::prelude::*;
use luisa_compute as luisa;

#[derive(Clone, Copy, Value, Debug)]
#[repr(C)]
#[value_new(pub)]
pub struct MyAddArgs {
    pub x: f32,
    pub y: f32,
    pub result: f32,
}

fn main() {
    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_device("cpu");
    let x = device.create_buffer::<f32>(1024);
    let y = device.create_buffer::<f32>(1024);
    let z = device.create_buffer::<f32>(1024);
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
    let shader = Kernel::<fn(Buffer<f32>)>::new(
        &device,
        &track!(|buf_z: BufferVar<f32>| {
            // z is pass by arg
            let buf_x = x.var(); // x and y are captured
            let buf_y = y.var();
            let tid = dispatch_id().x;
            let x = buf_x.read(tid);
            let y = buf_y.read(tid);
            let args = MyAddArgs::new_expr(x, y, 0.0f32.expr());
            let result = my_add.call(args);
            let _ = my_print.call(tid);
            if tid == 0 {
                cpu_dbg!(args);
            }
            buf_z.write(tid, result.result);
        }),
    );
    shader.dispatch([1024, 1, 1], &z);
    let mut z_data = vec![0.0; 1024];
    z.view(..).copy_to(&mut z_data);
    println!("{:?}", &z_data[0..16]);
}
