use proc_macro::TokenStream;


#[proc_macro_derive(Value)]
pub fn derive_value(item: TokenStream) -> TokenStream {
    let item: syn::ItemStruct = syn::parse(item).unwrap();
    let compiler = luisa_compute_compiler::Compiler::new();
    compiler.derive_value(&item).into()
}