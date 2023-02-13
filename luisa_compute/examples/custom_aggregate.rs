use luisa::prelude::*;
use luisa_compute as luisa;

#[derive(Aggregate)]
pub struct Spectrum {
    samples: Vec<Float32>,
}

#[derive(Aggregate)]
pub enum Color {
    Rgb(Expr<Vec3>),
    Spectrum(Spectrum)
}

fn main() {

}