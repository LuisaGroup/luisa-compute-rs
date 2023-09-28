use std::env::current_exe;
use std::f32::consts::PI;

use luisa::lang::poly::*;
use luisa::prelude::*;
use luisa_compute as luisa;

trait Area {
    fn area(&self) -> Expr<f32>;
}
#[derive(Value, Clone, Copy)]
#[repr(C)]
pub struct Circle {
    radius: f32,
}
impl Area for CircleExpr {
    #[tracked]
    fn area(&self) -> Expr<f32> {
        PI * self.radius * self.radius
    }
}
impl_polymorphic!(Area, Circle);
#[derive(Value, Clone, Copy)]
#[repr(C)]
pub struct Square {
    side: f32,
}
impl Area for SquareExpr {
    #[tracked]
    fn area(&self) -> Expr<f32> {
        self.side * self.side
    }
}
impl_polymorphic!(Area, Square);
fn main() {
    let ctx = Context::new(current_exe().unwrap());
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() <= 2,
        "Usage: {} <backend>. <backend>: cpu, cuda, dx, metal, remote",
        args[0]
    );
    let device = ctx.create_device(if args.len() == 2 {
        args[1].as_str()
    } else {
        "cpu"
    });
    let circles = device.create_buffer::<Circle>(2);
    circles
        .view(..)
        .copy_from(&[Circle { radius: 1.0 }, Circle { radius: 2.0 }]);
    let squares = device.create_buffer::<Square>(2);
    squares
        .view(..)
        .copy_from(&[Square { side: 1.0 }, Square { side: 2.0 }]);
    // Polymorphic<DevirtualizationKey, Trait>
    // Here we only need the type to devirtualize, so we use `()`.
    let mut poly_area: Polymorphic<(), dyn Area> = Polymorphic::new();
    // since we don't need a key, just supply `()`.
    poly_area.register((), &circles);
    poly_area.register((), &squares);
    let areas = device.create_buffer::<f32>(4);
    let shader = Kernel::<fn()>::new(
        &device,
        &track!(|| {
            let tid = dispatch_id().x;
            let tag = tid / 2;
            let index = tid % 2;
            let area = poly_area
                .get(TagIndex::new_expr(tag, index))
                .dispatch(|_tag, _key, obj| obj.area());
            areas.var().write(tid, area);
        }),
    );
    shader.dispatch([4, 1, 1]);
    let areas = areas.view(..).copy_to_vec();
    println!("{:?}", areas);
}
