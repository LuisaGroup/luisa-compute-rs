use proc_macro2::{Ident, TokenStream, TokenTree};
use quote::{quote, quote_spanned};
use syn::parse::Parse;
use syn::spanned::Spanned;
use syn::{Attribute, Item, ItemEnum, ItemFn, ItemStruct, Token, Visibility};

struct ValueNewOrdering {
    vis: Visibility,
    ordering: Vec<Ident>,
}

impl Parse for ValueNewOrdering {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let vis = input.parse()?;
        let ordering = if input.is_empty() {
            vec![]
        } else {
            let mut ordering = vec![];
            while !input.is_empty() {
                let ident = input.parse()?;
                ordering.push(ident);
                if !input.is_empty() {
                    let _ = input.parse::<Option<Token![,]>>()?;
                }
            }
            ordering
        };
        Ok(Self { vis, ordering })
    }
}

pub struct Compiler;
impl Compiler {
    fn lang_path(&self) -> TokenStream {
        quote!(::luisa_compute::lang)
    }
    fn runtime_path(&self) -> TokenStream {
        quote!(::luisa_compute::runtime)
    }
    fn value_attributes(&self, attribtes: &Vec<Attribute>) -> Option<ValueNewOrdering> {
        let mut has_repr_c = false;
        let mut ordering = None;
        for attr in attribtes {
            let meta = &attr.meta;
            match meta {
                syn::Meta::Path(path) => {
                    if path.is_ident("value_new") {
                        ordering = Some(ValueNewOrdering {
                            vis: Visibility::Inherited,
                            ordering: vec![],
                        });
                    }
                }
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
                    } else if path.is_ident("value_new") {
                        let value_new = syn::parse2::<ValueNewOrdering>(list.tokens.clone())
                            .expect("invalid value_new attribute");
                        ordering = Some(value_new);
                    }
                }
                _ => {}
            }
        }
        if !has_repr_c {
            panic!("Struct must have repr(C) attribute");
        }
        ordering
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
                type Arg = #name #ty_generics #where_clause;
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
            impl #impl_generics #runtime_path::AsKernelArg for #name #ty_generics #where_clause {
                type Output = #name #ty_generics;
            }
        )
    }
    pub fn derive_soa(&self, struct_: &ItemStruct) -> TokenStream {
        let span = struct_.span();
        let lang_path = self.lang_path();
        let generics = &struct_.generics;
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
        let name = &struct_.ident;
        let vis = &struct_.vis;
        let fields: Vec<_> = struct_.fields.iter().map(|f| f).collect();
        let field_vis: Vec<_> = fields.iter().map(|f| &f.vis).collect();
        let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();
        let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
        let soa_proxy_name = syn::Ident::new(&format!("{}Soa", name), name.span());
        quote_spanned!(span=>{
            #[repr(C)]
            #[derive(Copy, Clone)]
            pub struct #soa_proxy_name #generics #where_clause{
                #(#field_vis #field_names: <#field_types as #lang_path::SoaValue>::SoaBuffer),*
            }
            impl #impl_generics #lang_path::SoaValue for #name #ty_generics #where_clause{
                type SoaBuffer = #soa_proxy_name #ty_generics;
            }
            impl #impl_generics #lang_path::SoaBufferProxy for #soa_proxy_name #ty_generics #where_clause{
                type Value = #name #ty_generics;

                #[allow(unused_assignments)]
                fn from_soa_storage(
                    storage: ByteBufferVar,
                    meta: Expr<SoaMetadata>,
                    global_offset: usize,
                ) -> Self {
                    use #lang_path::SoaBufferProxy;
                    let mut i = 0;
                    #(
                        let $field_names = T::SoaBuffer::from_soa_storage(
                            storage.clone(),
                            meta.clone(),
                            global_offset + i,
                        );
                        i += <#field_types::SoaBuffer as SoaBufferProxy>::num_buffers();
                    )*
                    Self{
                        #(#field_names),*
                    }
                }
                fn num_buffers() -> usize {
                    [#( <#field_types as #lang_path::SoaValue>::SoaBuffer::num_buffers()),*].iter().sum()
                }
            }
            impl #impl_generics #lang_path::IndexRead for #soa_proxy_name #ty_generics #where_clause{
                type Element = #name #ty_generics;
                fn read<I:#lang_path::IntoIndex>(&self, i: I) -> Expr<Self::Element> {
                    let i = i.to_u64();
                    #(
                        let #field_names = self.#field_names.read(i);
                    )*
                    Self{
                        #(#field_names),*
                    }
                }
            }
            impl #impl_generics #lang_path::IndexWrite for #soa_proxy_name #ty_generics #where_clause{
                fn write<I:#lang_path::IntoIndex, V: #lang_path::AsExpr<Value = Self::Element>>(&self, i: I, value: V) {
                    let i = i.to_u64();
                    let v = value.as_expr();
                    #(
                        self.#field_names.write(i, v.#field_names);
                    )*
                }
            }
        })
    }
    pub fn derive_value(&self, struct_: &ItemStruct) -> TokenStream {
        let ordering = self.value_attributes(&struct_.attrs);
        let span = struct_.span();
        let lang_path = self.lang_path();
        // let runtime_path = self.runtime_path();
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
        let extract_atomic_ref_fields: Vec<_> = fields
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let ident = f.ident.as_ref().unwrap();
            let ty = &f.ty;
            quote_spanned!(span=>
                let #ident = < #lang_path::types::AtomicRef::<#ty> as #lang_path::FromNode>::from_node(#lang_path::__extract::<#ty>(
                    __node, #i,
                ));
            )
        })
        .collect();
        let expr_proxy_name = syn::Ident::new(&format!("{}Expr", name), name.span());
        let var_proxy_name = syn::Ident::new(&format!("{}Var", name), name.span());
        let atomic_ref_proxy_name = syn::Ident::new(&format!("{}AtomicRef", name), name.span());
        let ctor_proxy_name = syn::Ident::new(&format!("{}Comps", name), name.span());
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
                #[allow(dead_code)]
                #vis struct #ctor_proxy_name #generics #where_clause{
                    #(#ctor_fields),*
                }
                impl #impl_generics #name #ty_generics #where_clause{
                    #[allow(dead_code)]
                    #vis fn from_comps_expr(ctor: #ctor_proxy_name #ty_generics) -> #lang_path::types::Expr<#name #ty_generics> {
                        use #lang_path::*;
                        let node = #lang_path::__compose::<#name #ty_generics>(&[ #( #lang_path::ToNode::node(&ctor.#field_names.as_expr()) ),* ]);
                        let expr = <#lang_path::types::Expr::<#name> as #lang_path::FromNode>::from_node(node);
                        expr
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
            #[derive(Clone, Copy)]
            #[allow(unused_parens)]
            #[allow(dead_code)]
            #vis struct #expr_proxy_name #generics #where_clause{
                _marker: std::marker::PhantomData<(#marker_args)>,
                self_: #lang_path::types::Expr<#name>,
                #(#field_vis #field_names: #lang_path::types::Expr<#field_types>),*

            }
            #[derive(Clone, Copy)]
            #[allow(unused_parens)]
            #[allow(dead_code)]
            #vis struct #var_proxy_name #generics #where_clause{
                _marker: std::marker::PhantomData<(#marker_args)>,
                self_: #lang_path::types::Var<#name>,
                #(#field_vis #field_names: #lang_path::types::Var<#field_types>),*,
            }
            #[derive(Clone, Copy)]
            #[allow(unused_parens)]
            #[allow(dead_code)]
            #vis struct #atomic_ref_proxy_name #generics #where_clause{
                _marker: std::marker::PhantomData<(#marker_args)>,
                self_: #lang_path::types::AtomicRef<#name>,
                #(#field_vis #field_names: #lang_path::types::AtomicRef<#field_types>),*,
            }
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
            unsafe impl #impl_generics #lang_path::types::AtomicRefProxy for #atomic_ref_proxy_name #ty_generics #where_clause {
                type Value = #name #ty_generics;
                fn from_atomic_ref(var: #lang_path::types::AtomicRef<#name #ty_generics>) -> Self {
                    use #lang_path::ToNode;
                    let __node = var.node();
                    #(#extract_atomic_ref_fields)*
                    Self{
                        self_:var,
                        _marker:std::marker::PhantomData,
                        #(#field_names),*
                    }
                }
                fn as_atomic_ref_from_proxy(&self) -> &#lang_path::types::AtomicRef<#name> {
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
        );
        let new_expr = if let Some(ordering) = ordering {
            let (field_names, field_types) = if ordering.ordering.is_empty() {
                // fields
                //     .iter()
                //     .map(|f| f.ident.as_ref().unwrap())
                //     .collect::<Vec<_>>()
                let field_names = fields
                    .iter()
                    .map(|f| f.ident.as_ref().unwrap())
                    .collect::<Vec<_>>();
                let field_types = fields.iter().map(|f| &f.ty).collect::<Vec<_>>();
                (field_names, field_types)
            } else {
                let fields = ordering
                    .ordering
                    .iter()
                    .map(|ident| {
                        *fields
                            .iter()
                            .find(|f| f.ident.as_ref().unwrap() == ident)
                            .unwrap_or_else(|| panic!("field {} not found", ident))
                    })
                    .collect::<Vec<_>>();
                let fields_names = fields
                    .iter()
                    .map(|f| f.ident.as_ref().unwrap())
                    .collect::<Vec<_>>();
                let fields_types = fields.iter().map(|f| &f.ty).collect::<Vec<_>>();
                (fields_names, fields_types)
            };
            let vis = &ordering.vis;
            quote_spanned! {
                span =>
                impl #impl_generics #name #ty_generics #where_clause {
                    #vis fn new_expr(#(#field_names: impl #lang_path::types::AsExpr<Value = #field_types>),*) -> #lang_path::types::Expr::<#name> {
                        use #lang_path::*;
                        let node = #lang_path::__compose::<#name #ty_generics>(&[ #( #lang_path::ToNode::node(&#field_names.as_expr()) ),* ]);
                        let expr = <#lang_path::types::Expr::<#name> as #lang_path::FromNode>::from_node(node);
                        expr
                    }
                }
            }
        } else {
            quote!()
        };
        quote_spanned! {
            span=>
            #proxy_def
            #type_of_impl
            impl #impl_generics #lang_path::types::Value for #name #ty_generics #where_clause{
                type Expr = #expr_proxy_name #ty_generics;
                type Var = #var_proxy_name #ty_generics;
                type AtomicRef = #atomic_ref_proxy_name #ty_generics;
            }
            #new_expr
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
