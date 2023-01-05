use luisa::prelude::*;
use luisa_compute as luisa;

trait Area {
    fn area(&self) -> Float32;
}
fn main() {
    init();
    let device = create_cpu_device().unwrap();
}
