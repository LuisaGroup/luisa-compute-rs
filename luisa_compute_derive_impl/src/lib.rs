use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::{spanned::Spanned, Item, ItemEnum, ItemFn, ItemStruct};
pub struct Compiler {
    inside_crate: bool,
}
impl Compiler {
    fn crate_path(&self) -> TokenStream {
        if self.inside_crate {
            quote!(crate::lang)
        } else {
            quote!(luisa_compute::lang)
        }
    }
    pub fn new(inside_crate: bool) -> Self {
        Self { inside_crate }
    }
    pub fn compile_fn(&self, func: &ItemFn) -> TokenStream {
        quote!(#func)
    }
    pub fn compile_kernel(&self, func: &ItemFn) -> TokenStream {
        todo!()
    }
    pub fn derive_value(&self, struct_: &ItemStruct) -> TokenStream {
        let span = struct_.span();
        let crate_path = self.crate_path();
        let name = &struct_.ident;
        let vis = &struct_.vis;
        let fields: Vec<_> = struct_.fields.iter().map(|f| f).collect();
        let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();
        let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
        let proxy_name = syn::Ident::new(&format!("__{}Proxy", name), name.span());
        let proxy_fields: Vec<_> = fields
            .iter()
            .map(|f| {
                let ty = &f.ty;
                let name = &f.ident;
                let vis = &f.vis;
                quote_spanned!(span=> #vis #name : <#ty as #crate_path ::Value>::Proxy)
            })
            .collect();
        let proxy_def = quote_spanned!(span=>
            #[derive(Clone, Copy, Debug)]
            #vis struct #proxy_name {
                #( #proxy_fields ),*
            }
            impl #crate_path ::Aggregate for #proxy_name {
                fn to_nodes(&self, nodes: &mut Vec<#crate_path ::NodeRef>) {
                    #( self.#fields.to_nodes(nodes); )*
                }
                fn from_nodes<I: Iterator<Item = #crate_path ::NodeRef>>(iter: &mut I) -> Self {
                    Self {
                        #( #proxy_fields : <#field_types as #crate_path ::Value>::StructOfNodes::from_nodes(iter) ),*
                    }
                }
            }
            impl #crate_path ::VarProxy<#name> for #proxy_name {
                fn from_node(node: #crate_path ::NodeRef) -> Self {
                    let mut index = 0;
                    #(
                    let #field_names = {
                        let field = #crate_path ::__extract::<#field_types>(v.node(), &mut index);
                        <#field_types as #crate_path ::Value>::Proxy::from_node(field)
                    };
                    )*
                    Self{
                        #( #field_names ),*
                    }
                }
                fn node(&self) -> #crate_path ::NodeRef {
                    let mut nodes = Vec::new();
                    #(
                        nodes.append(
                            self.#field_names.node()
                        );
                    )*
                    __compose::<Self>(&nodes)
                }
            }
        );
        quote_spanned! {
            span=>
            #proxy_def
            impl #crate_path ::Value for #name {
                type Proxy = #proxy_name;
            }
        }
    }
    pub fn derive_struct_of_nodes_for_struct(&self, struct_: &ItemStruct) -> TokenStream {
        todo!()
    }
    pub fn derive_struct_of_nodes_for_enum(&self, enum_: &ItemEnum) -> TokenStream {
        todo!()
    }
    pub fn derive_struct_of_nodes(&self, item: &Item) -> TokenStream {
        match item {
            Item::Struct(struct_) => self.derive_struct_of_nodes_for_struct(struct_),
            Item::Enum(enum_) => self.derive_struct_of_nodes_for_enum(enum_),
            _ => todo!(),
        }
    }
}
