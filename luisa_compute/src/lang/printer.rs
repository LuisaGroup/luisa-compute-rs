use crate::prelude::*;
use crate::*;

pub struct Printer {
    items: Buffer<u32>,
    data: Buffer<u32>,
}

pub struct PrinterArgs<'a> {
    printer: &'a Printer,
}
impl Printer {
    pub fn _log<'a>(&self, args:PrinterArgs<'a>){}
    pub fn retrieve(&self) {

    }
}