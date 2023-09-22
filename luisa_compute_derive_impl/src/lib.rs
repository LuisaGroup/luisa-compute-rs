use proc_macro2::{TokenStream, TokenTree};
use quote::{quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{Attribute, Item, ItemEnum, ItemFn, ItemStruct};
pub struct Compiler;
impl Compiler {
    fn lang_path(&self) -> TokenStream {
        quote!(::luisa_compute::lang)
    }
    fn runtime_path(&self) -> TokenStream {
        quote!(::luisa_compute::runtime)
    }
    pub fn compile_kernel(&self, _func: &ItemFn) -> TokenStream {
        todo!()
    }
    fn check_repr_c(&self, attribtes: &Vec<Attribute>) {
        let mut has_repr_c = false;
        for attr in attribtes {
            let meta = &attr.meta;
            match meta {
                syn::Meta::List(list) => {
                    let path = &list.path;
                    if path.is_ident("repr") {
                        for tok in list.tokens.clone().into_iter() {
                            match tok {
                                TokenTree::Ident(ident) => {
                                    if ident == "C" {
                                        has_repr_c = true;
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                _ => {}
            }
            // if path == "repr" {
            //     let tokens = attr.bracket_token.to_string();
            //     if tokens == "(C)" {
            //         has_repr_c = true;
            //     }
            // }
        }
        if !has_repr_c {
            panic!("Struct must have repr(C) attribute");
        }
    }
    pub fn derive_kernel_arg(&self, struct_: &ItemStruct) -> TokenStream {
        let runtime_path = self.runtime_path();
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
                    let meta = &attr.meta;
                    match meta {
                        syn::Meta::List(list) => {
                            for tok in list.tokens.clone().into_iter() {
                                match tok {
                                    TokenTree::Ident(ident) => {
                                        if ident == "exclude" || ident == "ignore" {
                                            return false;
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
                #(#field_vis #field_names: <#field_types as #runtime_path::KernelArg>::Parameter),*
            }
        );
        quote_spanned!(span=>
            #parameter_def

            impl #impl_generics #runtime_path::KernelParameter for #parameter_name #ty_generics #where_clause{
                fn def_param(builder: &mut #runtime_path::KernelBuilder) -> Self {
                    Self{
                        #(#field_names:  #runtime_path::KernelParameter::def_param(builder)),*
                    }
                }
            }
            impl #impl_generics #runtime_path::KernelArg for #name #ty_generics #where_clause{
                type Parameter = #parameter_name #ty_generics;
                fn encode(&self, encoder: &mut  #runtime_path::KernelArgEncoder) {
                    #(self.#field_names.encode(encoder);)*
                }
            }
            impl #impl_generics #runtime_path::AsKernelArg<#name #ty_generics> for #name #ty_generics #where_clause {
            }
        )
    }

    pub fn derive_value(&self, struct_: &ItemStruct) -> TokenStream {
        self.check_repr_c(&struct_.attrs);
        let span = struct_.span();
        let lang_path = self.lang_path();
        let runtime_path = self.runtime_path();
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
        let field_vis: Vec<_> = fields.iter().map(|f| &f.vis).collect();
        let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();
        let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();

        let extract_expr_fields: Vec<_> = fields
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let ident = f.ident.as_ref().unwrap();
                let ty = &f.ty;
                quote_spanned!(span=>
                    let #ident = < #lang_path::types::Expr::<#ty> as #lang_path::FromNode>::from_node(#lang_path::__extract::<#ty>(
                        __node, #i,
                    ));
                )
            })
            .collect();
        let extract_var_fields: Vec<_> = fields
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let ident = f.ident.as_ref().unwrap();
                let ty = &f.ty;
                quote_spanned!(span=>
                    let #ident = < #lang_path::types::Var::<#ty> as #lang_path::FromNode>::from_node(#lang_path::__extract::<#ty>(
                        __node, #i,
                    ));
                )
            })
            .collect();
        let expr_proxy_name = syn::Ident::new(&format!("{}Expr", name), name.span());
        let var_proxy_name = syn::Ident::new(&format!("{}Var", name), name.span());
        let ctor_proxy_name = syn::Ident::new(&format!("{}Init", name), name.span());
        let ctor_proxy = {
            let ctor_fields = fields
                .iter()
                .map(|f| {
                    let ident = f.ident.as_ref().unwrap();
                    let ty = &f.ty;
                    quote_spanned!(span=> #vis #ident: #lang_path::types::Expr<#ty>)
                })
                .collect::<Vec<_>>();
            quote_spanned!(span=>
                #vis struct #ctor_proxy_name #generics {
                    #(#ctor_fields),*
                }
                impl #impl_generics From<#ctor_proxy_name #ty_generics> for #lang_path::types::Expr<#name> {
                    fn from(ctor: #ctor_proxy_name #ty_generics) -> #lang_path::types::Expr<#name> {
                        #name::new_expr(#(ctor.#field_names,)*)
                    }
                }
            )
        };
        let type_of_impl = quote_spanned!(span=>
            impl #impl_generics #lang_path::ir::TypeOf for #name #ty_generics #where_clause {
                #[allow(unused_parens)]
                fn type_() ->  #lang_path::ir::CArc< #lang_path::ir::Type> {
                    use #lang_path::*;
                    let size = std::mem::size_of::<#name #ty_generics>();
                    let alignment = std::mem::align_of::<#name #ty_generics>();
                    let struct_type = ir::StructType {
                        fields: ir::CBoxedSlice::new(vec![#(<#field_types as ir::TypeOf>::type_(),)*]),
                        size,
                        alignment
                    };
                    let type_ = ir::Type::Struct(struct_type);
                    assert_eq!(std::mem::size_of::<#name #ty_generics>(), type_.size());
                    ir::register_type(type_)
                }
            }
        );
        let proxy_def = quote_spanned!(span=>
            #ctor_proxy
            #[derive(Clone, Copy, Debug)]
            #[allow(unused_parens)]
            #vis struct #expr_proxy_name #generics{
                _marker: std::marker::PhantomData<(#marker_args)>,
                #[allow(dead_code)]
                self_: #lang_path::types::Expr<#name>,
                #(#field_vis #field_names: #lang_path::types::Expr<#field_types>),*

            }
            #[derive(Clone, Copy, Debug)]
            #[allow(unused_parens)]
            #vis struct #var_proxy_name #generics{
                _marker: std::marker::PhantomData<(#marker_args)>,
                #[allow(dead_code)]
                self_: #lang_path::types::Var<#name>,
                #(#field_vis #field_names: #lang_path::types::Var<#field_types>),*,
            }
            // #[allow(unused_parens)]
            // impl #impl_generics #lang_path::Aggregate for #expr_proxy_name #ty_generics #where_clause {
            //     fn to_nodes(&self, nodes: &mut Vec<#lang_path::NodeRef>) {
            //         nodes.push(self.node);
            //     }
            //     fn from_nodes<__I: Iterator<Item = #lang_path::NodeRef>>(iter: &mut __I) -> Self {
            //         Self{
            //             node: iter.next().unwrap(),
            //             _marker:std::marker::PhantomData
            //         }
            //     }
            // }
            // #[allow(unused_parens)]
            // impl #impl_generics #lang_path::Aggregate for #var_proxy_name #ty_generics #where_clause {
            //     fn to_nodes(&self, nodes: &mut Vec<#lang_path::NodeRef>) {
            //         nodes.push(self.node);
            //     }
            //     fn from_nodes<__I: Iterator<Item = #lang_path::NodeRef>>(iter: &mut __I) -> Self {
            //         Self{
            //             node: iter.next().unwrap(),
            //             _marker:std::marker::PhantomData
            //         }
            //     }
            // }
            // #[allow(unused_parens)]
            // impl #impl_generics #lang_path::FromNode  for #expr_proxy_name #ty_generics #where_clause {
            //     #[allow(unused_assignments)]
            //     fn from_node(node: #lang_path::NodeRef) -> Self {
            //         Self { node, _marker:std::marker::PhantomData }
            //     }
            // }
            // #[allow(unused_parens)]
            // impl #impl_generics #lang_path::ToNode  for #expr_proxy_name #ty_generics #where_clause {
            //     fn node(&self) -> #lang_path::NodeRef {
            //         self.node
            //     }
            // }
            #[allow(unused_parens)]
            impl #impl_generics #lang_path::types::ExprProxy for #expr_proxy_name #ty_generics #where_clause {
                type Value = #name #ty_generics;
                fn from_expr(expr: #lang_path::types::Expr<#name #ty_generics>) -> Self {
                    use #lang_path::ToNode;
                    let __node = expr.node();
                    #(#extract_expr_fields)*
                    Self{
                        self_:expr,
                        _marker:std::marker::PhantomData,
                        #(#field_names),*
                        
                    }
                }
                fn as_expr_from_proxy(&self) -> &#lang_path::types::Expr<#name> {
                    &self.self_
                }
            }
            // #[allow(unused_parens)]
            // impl #impl_generics #lang_path::FromNode for #var_proxy_name #ty_generics #where_clause {
            //     #[allow(unused_assignments)]
            //     fn from_node(node: #lang_path::NodeRef) -> Self {
            //         Self { node, _marker:std::marker::PhantomData }
            //     }
            // }
            // impl #impl_generics #lang_path::ToNode for #var_proxy_name #ty_generics #where_clause {
            //     fn node(&self) -> #lang_path::NodeRef {
            //         self.self_.node()
            //     }
            // }
            #[allow(unused_parens)]
            impl #impl_generics #lang_path::types::VarProxy for #var_proxy_name #ty_generics #where_clause {
                type Value = #name #ty_generics;
                fn from_var(var: #lang_path::types::Var<#name #ty_generics>) -> Self {
                    use #lang_path::ToNode;
                    let __node = var.node();
                    #(#extract_var_fields)*
                    Self{
                        self_:var,
                        _marker:std::marker::PhantomData,
                        #(#field_names),*
                    }
                }
                fn as_var_from_proxy(&self) -> &#lang_path::types::Var<#name> {
                    &self.self_
                }
            }
            #[allow(unused_parens)]
            impl #impl_generics std::ops::Deref for #var_proxy_name #ty_generics #where_clause {
                type Target = #lang_path::types::Expr<#name> #ty_generics;
                fn deref(&self) -> &Self::Target {
                    #lang_path::types::_deref_proxy(self)
                }
            }
            // #[allow(unused_parens)]
            // impl #impl_generics From<#var_proxy_name #ty_generics> for #expr_proxy_name #ty_generics #where_clause {
            //     fn from(var: #var_proxy_name #ty_generics) -> Self {
            //         var.load()
            //     }
            // }
            // #[allow(unused_parens)]
            // impl #impl_generics #runtime_path::CallableParameter for #expr_proxy_name #ty_generics #where_clause {
            //     fn def_param(_:Option<std::rc::Rc<dyn std::any::Any>>, builder: &mut #runtime_path::KernelBuilder) -> #lang_path::types::Expr<#name> #ty_generics {
            //         builder.value::<#name #ty_generics>()
            //     }
            //     fn encode(&self, encoder: &mut #runtime_path::CallableArgEncoder) {
            //         encoder.var(*self)
            //     }
            // }
            // #[allow(unused_parens)]
            // impl #impl_generics #runtime_path::CallableParameter for #var_proxy_name #ty_generics #where_clause  {
            //     fn def_param(_:Option<std::rc::Rc<dyn std::any::Any>>, builder: &mut #runtime_path::KernelBuilder) -> #lang_path::types::Var<#name> #ty_generics {
            //         builder.var::<#name #ty_generics>()
            //     }
            //     fn encode(&self, encoder: &mut #runtime_path::CallableArgEncoder) {
            //         encoder.var(*self)
            //     }
            // }

        );
        quote_spanned! {
            span=>
            #proxy_def
            #type_of_impl
            impl #impl_generics #lang_path::types::Value for #name #ty_generics #where_clause{
                type Expr = #expr_proxy_name #ty_generics;
                type Var = #var_proxy_name #ty_generics;
            }
            impl #impl_generics #lang_path::StructInitiaizable for #name #ty_generics #where_clause{
                type Init = #ctor_proxy_name #ty_generics;
            }
            impl #impl_generics #name #ty_generics #where_clause {
                #vis fn new_expr(#(#field_names: impl Into<#lang_path::types::Expr<#field_types>>),*) -> #lang_path::types::Expr::<#name> {
                    use #lang_path::*;
                    let node = #lang_path::__compose::<#name #ty_generics>(&[ #( #lang_path::ToNode::node(&#field_names.into()) ),* ]);
                    let expr = <#lang_path::types::Expr::<#name> as #lang_path::FromNode>::from_node(node);
                    expr
                }
            }
        }
    }
    pub fn derive_aggregate_for_struct(&self, struct_: &ItemStruct) -> TokenStream {
        let span = struct_.span();
        let lang_path = self.lang_path();
        let name = &struct_.ident;
        let fields: Vec<_> = struct_.fields.iter().map(|f| f).collect();
        let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();
        let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
        quote_spanned!(span=>
            impl #lang_path::Aggregate for #name {
                fn to_nodes(&self, nodes: &mut Vec<#lang_path::NodeRef>) {
                    #(self.#field_names.to_nodes(nodes);)*
                }
                fn from_nodes<__I: Iterator<Item = #lang_path::NodeRef>>(iter: &mut __I) -> Self {
                    #(let #field_names = <#field_types as #lang_path::Aggregate>::from_nodes(iter);)*
                    Self{
                        #(#field_names,)*
                    }
                }
            }
        )
    }
    pub fn derive_aggregate_for_enum(&self, enum_: &ItemEnum) -> TokenStream {
        let span = enum_.span();
        let lang_path = self.lang_path();
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
                            nodes.push(#lang_path::__new_user_node(#i));
                            #(#named.to_nodes(nodes);)*
                        }
                    }
                }
                syn::Fields::Unnamed(u) => {
                    let fields = u.unnamed.iter().enumerate().map(|(i, f)| syn::Ident::new(&format!("f{}", i), f.span())).collect::<Vec<_>>();
                    quote_spanned! {
                        field_span=>
                        Self::#name(#(#fields),*)=>{
                            nodes.push(#lang_path::__new_user_node(#i));
                            #(#fields.to_nodes(nodes);)*
                        }
                    }
                },
                syn::Fields::Unit => {
                    quote_spanned! { field_span=> Self::#name => {  nodes.push(#lang_path::__new_user_node(#i)); } }
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
                                #(let #fields: #field_types = #lang_path:: Aggregate ::from_nodes(iter);)*
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
            impl #lang_path::Aggregate for #name{
                #[allow(non_snake_case)]
                fn from_nodes<I: Iterator<Item = #lang_path::NodeRef>>(iter: &mut I) -> Self {
                    let variant = iter.next().unwrap();
                    let variant = variant.unwrap_user_data::<usize>();
                    match variant{
                        #(#from_nodes)*
                        _=> panic!("invalid variant"),
                    }
                }
                #[allow(non_snake_case)]
                fn to_nodes(&self, nodes: &mut Vec<#lang_path::NodeRef>){
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
