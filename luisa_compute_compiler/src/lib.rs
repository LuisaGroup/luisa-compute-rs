use proc_macro2::TokenStream;
use syn::{ItemFn, ItemStruct, Item};

pub struct Compiler {}
impl Compiler {
    pub fn new() -> Self {
        Self {}
    }
    pub fn compile_fn(&self, func: &ItemFn) -> TokenStream {
        todo!()
    }
    pub fn compile_kernel(&self, func: &ItemFn) -> TokenStream {
        todo!()
    }
    pub fn derive_value(&self, struct_: &ItemStruct) -> TokenStream {
        todo!()
    }
    pub fn derive_struct_of_nodes(&self, item: &Item) -> TokenStream {
        todo!()
    }
}
