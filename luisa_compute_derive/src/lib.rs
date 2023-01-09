use proc_macro::TokenStream;

#[proc_macro_derive(Value)]
pub fn derive_value(item: TokenStream) -> TokenStream {
    let item: syn::ItemStruct = syn::parse(item).unwrap();
    let compiler = luisa_compute_derive_impl::Compiler::new(false);
    compiler.derive_value(&item).into()
}

#[proc_macro_derive(KernelArg)]
pub fn derive_kernel_arg(item: TokenStream) -> TokenStream {
    let item: syn::ItemStruct = syn::parse(item).unwrap();
    let compiler = luisa_compute_derive_impl::Compiler::new(false);
    compiler.derive_kernel_arg(&item).into()
}

#[proc_macro_derive(Aggregate)]
pub fn derive_aggregate(item: TokenStream) -> TokenStream {
    let item: syn::Item = syn::parse(item).unwrap();
    let compiler = luisa_compute_derive_impl::Compiler::new(false);
    compiler.derive_aggregate(&item).into()
}

#[proc_macro_attribute]
pub fn function(attr: TokenStream, item: TokenStream) -> TokenStream {
    let item: syn::ItemFn = syn::parse(item).unwrap();
    let args = syn::parse_macro_input!(attr as syn::AttributeArgs);
    let compiler = luisa_compute_derive_impl::Compiler::new(false);
    compiler.compile_fn(&args, &item).into()
}

#[proc_macro_derive(__Value)]
pub fn _derive_value(item: TokenStream) -> TokenStream {
    let item: syn::ItemStruct = syn::parse(item).unwrap();
    let compiler = luisa_compute_derive_impl::Compiler::new(true);
    compiler.derive_value(&item).into()
}
#[proc_macro_derive(__Aggregate)]
pub fn _derive_aggregate(item: TokenStream) -> TokenStream {
    let item: syn::Item = syn::parse(item).unwrap();
    let compiler = luisa_compute_derive_impl::Compiler::new(true);
    compiler.derive_aggregate(&item).into()
}
