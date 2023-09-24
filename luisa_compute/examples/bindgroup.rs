use std::env::current_exe;

use luisa::prelude::*;
use luisa_compute as luisa;
#[derive(BindGroup)]
struct MyArgStruct<T: Value> {
    x: Buffer<T>,
    y: Buffer<T>,
    #[allow(dead_code)]
    #[luisa(exclude)]
    exclude: T,
}
fn main() {
    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_device("cpu");
    let x = device.create_buffer::<f32>(1024);
    let y = device.create_buffer::<f32>(1024);
    x.view(..).fill(1.0);
    y.view(..).fill(2.0);
    let my_args = MyArgStruct {
        x,
        y,
        exclude: 42.0,
    };
    let shader = Kernel::<fn(MyArgStruct<f32>)>::new(&device, &|_args| {});
    shader.dispatch([1024, 1, 1], &my_args);
}
