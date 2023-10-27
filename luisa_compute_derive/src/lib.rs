use proc_macro::TokenStream;
use syn::__private::quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;

/// Derives the `IoTexel` trait for a `#[repr(transparent)]` struct and a `Value` impl.
#[proc_macro_derive(IoTexel)]
pub fn derive_iotexel(item: TokenStream) -> TokenStream {
    let item: syn::Item = syn::parse(item).unwrap();
    let compiler = luisa_compute_derive_impl::Compiler;
    compiler.derive_iotexel(&item).into()
}

#[proc_macro_derive(Value, attributes(value_new))]
pub fn derive_value(item: TokenStream) -> TokenStream {
    let item: syn::Item = syn::parse(item).unwrap();
    let compiler = luisa_compute_derive_impl::Compiler;
    compiler.derive_value(&item).into()
}

#[proc_macro_derive(Soa)]
pub fn derive_soa(item: TokenStream) -> TokenStream {
    let item: syn::ItemStruct = syn::parse(item).unwrap();
    let compiler = luisa_compute_derive_impl::Compiler;
    compiler.derive_soa(&item).into()
}

#[proc_macro_derive(BindGroup, attributes(luisa))]
pub fn derive_kernel_arg(item: TokenStream) -> TokenStream {
    let item: syn::ItemStruct = syn::parse(item).unwrap();
    let compiler = luisa_compute_derive_impl::Compiler;
    compiler.derive_kernel_arg(&item).into()
}

#[proc_macro_derive(Aggregate)]
pub fn derive_aggregate(item: TokenStream) -> TokenStream {
    let item: syn::Item = syn::parse(item).unwrap();
    let compiler = luisa_compute_derive_impl::Compiler;
    compiler.derive_aggregate(&item).into()
}

struct LogInput {
    printer: syn::Expr,
    level: syn::Expr,
    fmt: syn::LitStr,
    args: Vec<syn::Expr>,
}
impl Parse for LogInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let printer = input.parse()?;
        input.parse::<syn::Token![,]>()?;
        let level = input.parse()?;
        input.parse::<syn::Token![,]>()?;
        let fmt = input.parse()?;
        let mut args = vec![];
        while !input.is_empty() {
            input.parse::<syn::Token![,]>()?;
            args.push(input.parse()?);
        }
        Ok(Self {
            printer,
            level,
            fmt,
            args,
        })
    }
}

#[proc_macro]
pub fn _log(item: TokenStream) -> TokenStream {
    // _log($printer:expr, $level:expr, $fmt:literal, $($arg:expr), *)
    let input = syn::parse_macro_input!(item as LogInput);
    let printer = &input.printer;
    let level = &input.level;
    let fmt = &input.fmt;
    let args = &input.args;
    let arg_idents = input
        .args
        .iter()
        .enumerate()
        .map(|(i, a)| syn::Ident::new(&format!("__log_priv_arg{}", i), a.span()))
        .collect::<Vec<_>>();
    quote! {
        {
            #(
                let #arg_idents = #args._type_tag();
            )*;
            let mut __log_priv_i = 0;
            let log_fn = Box::new(move |args: &[*const u32]| -> () {
                let mut i = 0;
                luisa_compute::printer::_log::log!(#level, #fmt , #(
                    {
                        let ret = luisa_compute::printer::_unpack_from_expr(args[i], #arg_idents);
                        i += 1;
                        ret
                    }
                ), *);
            });
            #(
                let #arg_idents = #args;
            )*;
            let mut printer_args = luisa_compute::printer::PrinterArgs::new();
            #(
                printer_args.append(#arg_idents);
            )*
            #printer._log(#level, printer_args, log_fn);
        }
    }
    .into()
}
