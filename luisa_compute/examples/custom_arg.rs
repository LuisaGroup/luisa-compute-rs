use luisa::prelude::*;
use luisa_compute as luisa;
#[derive(KernelArg)]
struct MyArgStruct<T: Value> {
    x: Buffer<T>,
    y: Buffer<T>,
    #[allow(dead_code)]
    #[luisa(exclude)]
    exclude: T,
}
fn main() {
    init();
    let device = create_cpu_device().unwrap();
    let x = device.create_buffer::<f32>(1024).unwrap();
    let y = device.create_buffer::<f32>(1024).unwrap();
    x.view(..).fill(1.0);
    y.view(..).fill(2.0);
    let my_args = MyArgStruct {
        x,
        y,
        exclude: 42.0,
    };
    let shader = device
        .create_kernel::<(MyArgStruct<f32>,)>(&|_args| {})
        .unwrap();
    shader.dispatch([1024, 1, 1], &my_args).unwrap();
}
