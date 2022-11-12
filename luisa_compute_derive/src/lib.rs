use proc_macro::TokenStream;

#[proc_macro_derive(Value)]
pub fn derive_value(item: TokenStream) -> TokenStream {
    let item: syn::ItemStruct = syn::parse(item).unwrap();
    let compiler = luisa_compute_compiler::Compiler::new(false);
    compiler.derive_value(&item).into()
}
#[proc_macro_derive(StructOfNodes)]
pub fn derive_struct_of_nodes(item: TokenStream) -> TokenStream {
    let item: syn::Item = syn::parse(item).unwrap();
    let compiler = luisa_compute_compiler::Compiler::new(false);
    compiler.derive_struct_of_nodes(&item).into()
}

#[proc_macro_attribute]
pub fn function(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item: syn::ItemFn = syn::parse(item).unwrap();
    let compiler = luisa_compute_compiler::Compiler::new(false);
    compiler.compile_fn(&item).into()
}
