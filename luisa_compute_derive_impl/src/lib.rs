use std::collections::HashSet;

use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::{spanned::Spanned, Attribute, Item, ItemEnum, ItemFn, ItemStruct, NestedMeta};
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
    pub fn compile_fn(&self, args: &Vec<NestedMeta>, func: &ItemFn) -> TokenStream {
        quote!(#func)
    }
    pub fn compile_kernel(&self, func: &ItemFn) -> TokenStream {
        todo!()
    }
    fn check_repr_c(&self, attribtes: &Vec<Attribute>) {
        let mut has_repr_c = false;
        for attr in attribtes {
            let path = attr.path.get_ident().unwrap().to_string();
            if path == "repr" {
                let tokens = attr.tokens.to_string();
                if tokens == "(C)" {
                    has_repr_c = true;
                }
            }
        }
        if !has_repr_c {
            panic!("Struct must have repr(C) attribute");
        }
    }
    pub fn derive_value(&self, struct_: &ItemStruct) -> TokenStream {
        self.check_repr_c(&struct_.attrs);
        let span = struct_.span();
        let crate_path = self.crate_path();
        let name = &struct_.ident;
        let vis = &struct_.vis;
        let fields: Vec<_> = struct_.fields.iter().map(|f| f).collect();
        let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();
        let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
        let proxy_name = syn::Ident::new(&format!("{}Proxy", name), name.span());
        let proxy_fields: Vec<_> = fields
            .iter()
            .map(|f| {
                let ty = &f.ty;
                let name = &f.ident;
                let vis = &f.vis;
                quote_spanned!(span=> #vis #name : #crate_path ::Expr<#ty>)
            })
            .collect();
        let type_of_impl = quote_spanned!(span=>
            impl #crate_path ::TypeOf for #name {
                fn type_() ->  #crate_path ::Gc< #crate_path ::Type> {
                    use #crate_path ::*;
                    let size = std::mem::size_of::<#name>();
                    let alignment = std::mem::align_of::<#name>();
                    let struct_type = StructType {
                        fields: CBoxedSlice::new(vec![#(<#field_types as TypeOf>::type_(),)*]),
                        size,
                        alignment
                    };
                    let type_ = Type::Struct(struct_type);
                    register_type(type_)
                }
            }
        );
        let proxy_def = quote_spanned!(span=>
            #[derive(Clone, Copy, Debug)]
            #vis struct #proxy_name {
                #( #proxy_fields ),*
            }
            impl #crate_path ::Aggregate for #proxy_name {
                fn to_nodes(&self, nodes: &mut Vec<#crate_path ::NodeRef>) {
                    #( self.#field_names.to_nodes(nodes); )*
                }
                fn from_nodes<I: Iterator<Item = #crate_path ::NodeRef>>(iter: &mut I) -> Self {
                    Self {
                        #( #field_names : #crate_path ::Expr::<#field_types>::from_proxy(
                            <#field_types as #crate_path ::Value>::Proxy::from_nodes(iter)) ),*
                    }
                }
            }
            impl #crate_path ::Proxy<#name> for #proxy_name {
                #[allow(unused_assignments)]
                fn from_node(node: #crate_path ::NodeRef) -> Self {
                    let mut index = 0;
                    #(
                    let #field_names = {
                        let field = #crate_path ::__extract::<#field_types>(node, index);
                        index += 1;
                        #crate_path ::Expr::<#field_types>::from_proxy(<#field_types as #crate_path ::Value>::Proxy::from_node(field))
                    };
                    )*
                    Self{
                        #( #field_names ),*
                    }
                }
                fn node(&self) -> #crate_path ::NodeRef {
                    let mut nodes = Vec::new();
                    #(
                        nodes.push(
                            self.#field_names.node()
                        );
                    )*
                    #crate_path ::__compose::<#name>(&nodes)
                }
            }
        );
        quote_spanned! {
            span=>
            #proxy_def
            #type_of_impl
            impl #crate_path ::Value for #name {
                type Proxy = #proxy_name;
            }
        }
    }
    pub fn derive_aggregate_for_struct(&self, struct_: &ItemStruct) -> TokenStream {
        todo!()
    }
    pub fn derive_aggregate_for_enum(&self, enum_: &ItemEnum) -> TokenStream {
        todo!()
    }
    pub fn derive_aggregate(&self, item: &Item) -> TokenStream {
        match item {
            Item::Struct(struct_) => self.derive_aggregate_for_struct(struct_),
            Item::Enum(enum_) => self.derive_aggregate_for_enum(enum_),
            _ => todo!(),
        }
    }
}
