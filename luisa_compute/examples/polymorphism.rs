use std::f32::consts::PI;

use luisa::prelude::{poly::Polymorphic, *};
use luisa_compute as luisa;

trait Area {
    fn area(&self) -> Float32;
}
#[derive(Value, Clone, Copy)]
#[repr(C)]
pub struct Circle {
    radius: f32,
}
impl Area for CircleExpr {
    fn area(&self) -> Float32 {
        PI * self.radius() * self.radius()
    }
}
impl_polymorphic!(Area, Circle);
#[derive(Value, Clone, Copy)]
#[repr(C)]
pub struct Square {
    side: f32,
}
impl Area for SquareExpr {
    fn area(&self) -> Float32 {
        self.side() * self.side()
    }
}
impl_polymorphic!(Area, Square);
fn main() {
    init();
    let device = create_cpu_device().unwrap();
    let circles = device.create_buffer::<Circle>(2).unwrap();
    circles
        .view(..)
        .copy_from(&[Circle { radius: 1.0 }, Circle { radius: 2.0 }]);
    let squares = device.create_buffer::<Square>(2).unwrap();
    squares
        .view(..)
        .copy_from(&[Square { side: 1.0 }, Square { side: 2.0 }]);
    let mut poly_area: Polymorphic<dyn Area> = Polymorphic::new();
    poly_area.register(&circles);
    poly_area.register(&squares);
    let areas = device.create_buffer::<f32>(4).unwrap();
    let kernel = device.create_kernel::<()>(&|| {
        let tid = dispatch_id().x();
        let tag = tid / 2;
        let index = tid % 2;
        let area = poly_area.dispatch(tag, index, |obj|{
            obj.area()
        });
        areas.var().write(tid, area);
    }).unwrap();
    kernel.dispatch([4,1,1]).unwrap();
    let areas = areas.view(..).copy_to_vec();
    println!("{:?}", areas);
}
