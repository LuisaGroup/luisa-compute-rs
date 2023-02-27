use luisa::prelude::*;
use luisa_compute as luisa;

#[derive(Aggregate)]
pub struct Spectrum {
    samples: Vec<Float>,
}

#[derive(Aggregate)]
pub enum Color {
    Rgb(Expr<Float3>),
    Spectrum(Spectrum)
}

fn main() {

}