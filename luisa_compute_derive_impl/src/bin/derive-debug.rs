use luisa_compute_derive_impl::*;
use quote::ToTokens;
#[repr(C)]
struct Foo {
    a: u32,
    b: u32,
}

fn main() {
    let compiler = Compiler;
    let item: syn::ItemStruct = syn::parse_str(
        r#"
        #[derive(__Value)]
        #[repr(C)]
        struct Foo {
            a: u32,
            b: u32,
        }
        "#,
    )
    .unwrap();
    println!("{:?}", item.to_token_stream());
    let out = compiler.derive_value(&item);
    println!("{:?}", out.to_string());
}
