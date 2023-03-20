use std::collections::HashSet;

use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::{spanned::Spanned, Attribute, Item, ItemEnum, ItemFn, ItemStruct, ItemTrait, NestedMeta};
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
    pub fn derive_kernel_arg(&self, struct_: &ItemStruct) -> TokenStream {
        let span = struct_.span();
        let name = &struct_.ident;
        let vis = &struct_.vis;
        let generics = &struct_.generics;
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
        let fields: Vec<_> = struct_
            .fields
            .iter()
            .map(|f| f)
            .filter(|f| {
                let attrs = &f.attrs;
                for attr in attrs {
                    let meta = attr.parse_meta().unwrap();
                    match meta {
                        syn::Meta::List(list) => {
                            for attr in &list.nested {
                                match attr {
                                    NestedMeta::Meta(syn::Meta::Path(path)) => {
                                        if let Some(ident) = path.get_ident() {
                                            if ident == "exclude" || ident == "ignore" {
                                                return false;
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        _ => {}
                    }
                }
                true
            })
            .collect();
        let field_vis: Vec<_> = fields.iter().map(|f| &f.vis).collect();
        let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();
        let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
        let parameter_name = syn::Ident::new(&format!("{}Var", name), name.span());
        let parameter_def = quote!(
            #vis struct #parameter_name #generics {
                #(#field_vis #field_names: <#field_types as luisa_compute::runtime::KernelArg>::Parameter),*
            }
        );
        quote_spanned!(span=>
            #parameter_def

            impl #impl_generics luisa_compute::lang::KernelParameter for #parameter_name #ty_generics #where_clause{
                fn def_param(builder: &mut ShaderBuilder) -> Self {
                    Self{
                        #(#field_names:  luisa_compute::lang::KernelParameter::def_param(builder)),*
                    }
                }
            }
            impl #impl_generics luisa_compute::runtime::KernelArg for #name #ty_generics #where_clause{
                type Parameter = #parameter_name #ty_generics;
                fn encode(&self, encoder: &mut  luisa_compute::prelude::ArgEncoder) {
                    #(self.#field_names.encode(encoder);)*
                }
            }
            impl #impl_generics luisa_compute::runtime::AsKernelArg<#name #ty_generics> for #name #ty_generics #where_clause {
            }
        )
    }
    pub fn derive_value(&self, struct_: &ItemStruct) -> TokenStream {
        self.check_repr_c(&struct_.attrs);
        let span = struct_.span();
        let crate_path = self.crate_path();
        let generics = &struct_.generics;
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
        let marker_args = generics
            .params
            .iter()
            .map(|p| match p {
                syn::GenericParam::Type(ty) => {
                    let ident = &ty.ident;
                    quote!(#ident)
                }
                syn::GenericParam::Lifetime(lt) => {
                    let lt = &lt.lifetime;
                    quote!(& #lt u32)
                }
                syn::GenericParam::Const(_) => {
                    panic!("Const generic parameter is not supported")
                }
            })
            .collect::<Vec<_>>();
        let marker_args = quote!(#(#marker_args),*);
        let name = &struct_.ident;
        let vis = &struct_.vis;
        let fields: Vec<_> = struct_.fields.iter().map(|f| f).collect();
        let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();
        let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
        let expr_proxy_field_methods: Vec<_> = fields
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let ident = f.ident.as_ref().unwrap();
                let vis = &f.vis;
                let ty = &f.ty;
                let set_ident = syn::Ident::new(&format!("set_{}", ident), ident.span());
                quote_spanned!(span=>
                    #[allow(dead_code)]
                    #vis fn #ident (&self) -> Expr<#ty> {
                        <Expr::<#ty> as FromNode>::from_node(__extract::<#ty>(
                            self.node, #i,
                        ))
                    }
                    #[allow(dead_code)]
                    #vis fn #set_ident<T:Into<Expr<#ty>>>(&self, value: T) -> Self {
                        let value = value.into();
                        Self::from_node(#crate_path ::__insert::<#name #ty_generics>(self.node, #i, FromNode::node(&value)))
                    }
                )
            })
            .collect();
        let var_proxy_field_methods: Vec<_> = fields
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let ident = f.ident.as_ref().unwrap();
                let vis = &f.vis;
                let ty = &f.ty;
                let set_ident = syn::Ident::new(&format!("set_{}", ident), ident.span());
                quote_spanned!(span=>
                    #[allow(dead_code)]
                    #vis fn #ident (&self) -> Var<#ty> {
                        <Var::<#ty> as FromNode>::from_node(__extract::<#ty>(
                            self.node, #i,
                        ))
                    }
                    #[allow(dead_code)]
                    #vis fn #set_ident<T:Into<Expr<#ty>>>(&self, value: T) {
                        let value = value.into();
                        self.#ident().store(value);
                    }
                )
            })
            .collect();
        let expr_proxy_name = syn::Ident::new(&format!("{}Expr", name), name.span());
        let var_proxy_name = syn::Ident::new(&format!("{}Var", name), name.span());
        let type_of_impl = quote_spanned!(span=>
            impl #impl_generics #crate_path ::TypeOf for #name #ty_generics #where_clause {
                fn type_() ->  #crate_path ::CArc< #crate_path ::Type> {
                    use #crate_path ::*;
                    let size = std::mem::size_of::<#name #ty_generics>();
                    let alignment = std::mem::align_of::<#name #ty_generics>();
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
            #vis struct #expr_proxy_name #generics{
                node: #crate_path ::NodeRef,
                _marker: std::marker::PhantomData<(#marker_args)>,
            }
            #[derive(Clone, Copy, Debug)]
            #vis struct #var_proxy_name #generics{
                node: #crate_path ::NodeRef,
                _marker: std::marker::PhantomData<(#marker_args)>,
            }
            impl #impl_generics #crate_path ::Aggregate for #expr_proxy_name #ty_generics #where_clause {
                fn to_nodes(&self, nodes: &mut Vec<#crate_path ::NodeRef>) {
                    nodes.push(self.node);
                }
                fn from_nodes<I: Iterator<Item = #crate_path ::NodeRef>>(iter: &mut I) -> Self {
                    Self{
                        node: iter.next().unwrap(),
                        _marker:std::marker::PhantomData
                    }
                }
            }
            impl #impl_generics #crate_path ::Aggregate for #var_proxy_name #ty_generics #where_clause {
                fn to_nodes(&self, nodes: &mut Vec<#crate_path ::NodeRef>) {
                    nodes.push(self.node);
                }
                fn from_nodes<I: Iterator<Item = #crate_path ::NodeRef>>(iter: &mut I) -> Self {
                    Self{
                        node: iter.next().unwrap(),
                        _marker:std::marker::PhantomData
                    }
                }
            }
            impl #impl_generics #crate_path ::FromNode  for #expr_proxy_name #ty_generics #where_clause {
                #[allow(unused_assignments)]
                fn from_node(node: #crate_path ::NodeRef) -> Self {
                    Self { node, _marker:std::marker::PhantomData }
                }
                fn node(&self) -> #crate_path ::NodeRef {
                    self.node
                }
            }
            impl #impl_generics #crate_path ::ExprProxy for #expr_proxy_name #ty_generics #where_clause {
                type Value = #name #ty_generics;
            }
            impl #impl_generics #crate_path ::FromNode for #var_proxy_name #ty_generics #where_clause {
                #[allow(unused_assignments)]
                fn from_node(node: #crate_path ::NodeRef) -> Self {
                    Self { node, _marker:std::marker::PhantomData }
                }
                fn node(&self) -> #crate_path ::NodeRef {
                    self.node
                }
            }
            impl #impl_generics #crate_path ::VarProxy for #var_proxy_name #ty_generics #where_clause {
                type Value = #name #ty_generics;
            }
            impl #impl_generics From<#var_proxy_name #ty_generics> for #expr_proxy_name #ty_generics #where_clause {
                fn from(var: #var_proxy_name #ty_generics) -> Self {
                    var.load()
                }
            }

        );
        let test_name = syn::Ident::new(&format!("test_{}", name), name.span());
        let test = quote! {
            #[allow(non_snake_case)]
            #[cfg(test)]
            mod #test_name {
                #[test]
                fn test_size() {
                    use std::mem::size_of;
                    assert_eq!(size_of::<super:: #name #ty_generics>(), <super:: #name #ty_generics as #crate_path ::TypeOf>::type_().size());
                }
            }
        };
        quote_spanned! {
            span=>
            #proxy_def
            #type_of_impl
            impl #impl_generics #crate_path ::Value for #name #ty_generics #where_clause{
                type Expr = #expr_proxy_name #ty_generics;
                type Var = #var_proxy_name #ty_generics;
                fn fields() -> Vec<String> {
                    vec![#(stringify!(#field_names).into(),)*]
                }
            }
            impl #impl_generics #expr_proxy_name #ty_generics #where_clause {
                #(#expr_proxy_field_methods)*
                #vis fn new(#(#field_names: impl Into<Expr<#field_types>>),*) -> Self {
                    let node = #crate_path ::__compose::<#name #ty_generics>(&[ #( FromNode::node(&#field_names.into()) ),* ]);
                    Self { node, _marker:std::marker::PhantomData }
                }
            }
            impl #impl_generics  #var_proxy_name #ty_generics #where_clause {
                #(#var_proxy_field_methods)*
            }
            #test
        }
    }
    pub fn derive_aggregate_for_struct(&self, struct_: &ItemStruct) -> TokenStream {
        let span = struct_.span();
        let crate_path = self.crate_path();
        let name = &struct_.ident;
        let fields: Vec<_> = struct_.fields.iter().map(|f| f).collect();
        let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();
        let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
        quote_spanned!(span=>
            impl #crate_path ::Aggregate for #name {
                fn to_nodes(&self, nodes: &mut Vec<#crate_path ::NodeRef>) {
                    #(self.#field_names.to_nodes(nodes);)*
                }
                fn from_nodes<I: Iterator<Item = #crate_path ::NodeRef>>(iter: &mut I) -> Self {
                    #(let #field_names = <#field_types as #crate_path ::Aggregate>::from_nodes(iter);)*
                    Self{
                        #(#field_names,)*
                    }
                }
            }
        )
    }
    pub fn derive_aggregate_for_enum(&self, enum_: &ItemEnum) -> TokenStream {
        let span = enum_.span();
        let crate_path = self.crate_path();
        let name = &enum_.ident;
        let variants = &enum_.variants;
        let to_nodes = variants.iter().enumerate().map(|(i, v)|{
            let name = &v.ident;
            let field_span = v.span();
            match &v.fields {
                syn::Fields::Named(n) => {
                    let named = n
                        .named
                        .iter()
                        .map(|f| f.ident.clone().unwrap())
                        .collect::<Vec<_>>();

                    quote_spanned! {
                        field_span=>
                        Self::#name{#(#named),*}=>{
                            nodes.push(__new_user_node(#i));
                            #(#named.to_nodes(nodes);)*
                        }
                    }
                }
                syn::Fields::Unnamed(u) => {
                    let fields = u.unnamed.iter().enumerate().map(|(i, f)| syn::Ident::new(&format!("f{}", i), f.span())).collect::<Vec<_>>();
                    quote_spanned! {
                        field_span=>
                        Self::#name(#(#fields),*)=>{
                            nodes.push(__new_user_node(#i));
                            #(#fields.to_nodes(nodes);)*
                        }
                    }
                },
                syn::Fields::Unit => {
                    quote_spanned! { field_span=> Self::#name => {  nodes.push(#crate_path ::__new_user_node(#i)); } }
                }
            }
        }).collect::<Vec<_>>();
        let from_nodes = variants
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let name = &v.ident;
                let field_span = v.span();
                match &v.fields {
                    syn::Fields::Unnamed(u) => {
                        let field_types = u.unnamed.iter().map(|f| &f.ty).collect::<Vec<_>>();
                        let fields = u.unnamed.iter().enumerate().map(|(i, f)| syn::Ident::new(&format!("f{}", i), f.span())).collect::<Vec<_>>();
                        quote_spanned! { field_span=>
                            #i=> {
                                #(let #fields: #field_types = #crate_path :: Aggregate ::from_nodes(iter);)*
                                Self::#name(#(#fields),*)
                            },
                        }
                    }
                    syn::Fields::Unit => {
                        quote_spanned! {  field_span=> #i=>Self::#name, }
                    }
                    syn::Fields::Named(n) => {
                        let named = n
                            .named
                            .iter()
                            .map(|f| f.ident.clone().unwrap())
                            .collect::<Vec<_>>();
                        quote_spanned! { field_span=>
                            #i=> {
                                #(let #named = #named ::from_nodes(iter);)*
                                Self::#name{#(#named),*}
                            },
                        }
                    }
                }
            })
            .collect::<Vec<_>>();
        quote_spanned! {span=>
            impl #crate_path ::Aggregate for #name{
                #[allow(non_snake_case)]
                fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
                    let variant = iter.next().unwrap();
                    let variant = variant.unwrap_user_data::<usize>();
                    match variant{
                        #(#from_nodes)*
                        _=> panic!("invalid variant"),
                    }
                }
                #[allow(non_snake_case)]
                fn to_nodes(&self, nodes: &mut Vec<NodeRef>){
                    match self {
                        #(#to_nodes)*
                    }
                }
            }
        }
    }
    pub fn derive_aggregate(&self, item: &Item) -> TokenStream {
        match item {
            Item::Struct(struct_) => self.derive_aggregate_for_struct(struct_),
            Item::Enum(enum_) => self.derive_aggregate_for_enum(enum_),
            _ => todo!(),
        }
    }
}
