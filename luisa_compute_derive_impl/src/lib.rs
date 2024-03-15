use std::collections::HashMap;

use proc_macro2::{Ident, TokenStream, TokenTree};
use quote::{quote, quote_spanned};
use syn::ext::IdentExt;
use syn::parse::Parse;
use syn::spanned::Spanned;
use syn::{Attribute, Item, ItemEnum, ItemStruct, Token, Visibility};

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

pub struct Compiler {
    crate_path: TokenStream,
}
impl Compiler {
    pub fn new() -> Self {
        Self {
            crate_path: quote!(::luisa_compute),
        }
    }
    fn lang_path(&self) -> TokenStream {
        let crate_path = &self.crate_path;
        quote!(#crate_path::lang)
    }
    fn runtime_path(&self) -> TokenStream {
        let crate_path = &self.crate_path;
        quote!(#crate_path::runtime)
    }
    fn resource_path(&self) -> TokenStream {
        let crate_path = &self.crate_path;
        quote!(#crate_path::resource)
    }
    fn parse_luisa_attributes(
        &self,
        attribtes: &Vec<Attribute>,
    ) -> HashMap<String, Option<TokenStream>> {
        // checks for luisa attribute

        struct AttrAssign {
            ident: Ident,
            #[allow(dead_code)]
            eq: Option<Token![=]>,
            string: Option<syn::LitStr>,
        }

        impl Parse for AttrAssign {
            fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
                let ident = input.call(Ident::parse_any)?;
                let eq: Option<Token![=]> = input.parse()?;
                if eq.is_none() {
                    return Ok(Self {
                        ident,
                        eq,
                        string: None,
                    });
                }
                let string = input.parse()?;
                Ok(Self {
                    ident,
                    eq,
                    string: Some(string),
                })
            }
        }

        struct VecAttrAssign(Vec<AttrAssign>);
        impl Parse for VecAttrAssign {
            fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
                let mut vec = vec![];
                while !input.is_empty() {
                    let attr = input.parse()?;
                    vec.push(attr);
                    if !input.is_empty() {
                        let _ = input.parse::<Option<Token![,]>>()?;
                    }
                }
                Ok(Self(vec))
            }
        }

        let mut map = HashMap::new();
        for attr in attribtes {
            let meta = &attr.meta;
            match meta {
                syn::Meta::List(list) => {
                    let path = &list.path;
                    if path.is_ident("luisa") {
                        let attr_assigns = syn::parse2::<VecAttrAssign>(list.tokens.clone())
                            .expect("invalid luisa attribute");
                        for attr_assign in attr_assigns.0 {
                            let ident = attr_assign.ident.to_string();
                            let value = if let Some(string) = attr_assign.string {
                                let string: TokenStream = string.value().parse().unwrap();
                                Some(quote!(#string))
                            } else {
                                None
                            };
                            map.insert(ident, value);
                        }
                    }
                }
                _ => {}
            }
        }
        map
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
                            if let TokenTree::Ident(ident) = tok {
                                if ident == "C" || ident == "transparent" {
                                    has_repr_c = true;
                                }
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
    fn set_crate_path_from_attrs(&mut self, attrs: &HashMap<String, Option<TokenStream>>) {
        if let Some(crate_path) = attrs.get("crate") {
            if let Some(crate_path) = crate_path {
                self.crate_path = crate_path.clone();
            }
        }
    }
    pub fn derive_kernel_arg(&mut self, struct_: &ItemStruct) -> TokenStream {
        let attrs = self.parse_luisa_attributes(&struct_.attrs);
        self.set_crate_path_from_attrs(&attrs);
        let runtime_path = self.runtime_path();
        let span = struct_.span();
        let name = &struct_.ident;
        let vis = &struct_.vis;
        let generics = &struct_.generics;
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
        let fields: Vec<_> = struct_
            .fields
            .iter()
            .filter(|f| {
                let attrs = &f.attrs;
                for attr in attrs {
                    let meta = &attr.meta;
                    if let syn::Meta::List(list) = meta {
                        for tok in list.tokens.clone().into_iter() {
                            if let TokenTree::Ident(ident) = tok {
                                if ident == "exclude" || ident == "ignore" {
                                    return false;
                                }
                            }
                        }
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
    pub fn derive_soa(&mut self, struct_: &ItemStruct) -> TokenStream {
        let attrs = self.parse_luisa_attributes(&struct_.attrs);
        self.set_crate_path_from_attrs(&attrs);
        let span = struct_.span();
        let lang_path = self.lang_path();
        let crate_path = &self.crate_path;
        let generics = &struct_.generics;
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
        let name = &struct_.ident;
        let vis = &struct_.vis;
        let fields: Vec<_> = struct_.fields.iter().collect();
        let field_vis: Vec<_> = fields.iter().map(|f| &f.vis).collect();
        let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();
        let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
        let soa_proxy_name = syn::Ident::new(&format!("{}Soa", name), name.span());
        quote_spanned!(span=>
            #[derive(Clone)]
            #vis struct #soa_proxy_name #generics #where_clause{
                #(#field_vis #field_names: <#field_types as #lang_path::types::SoaValue>::SoaBuffer),*
            }
            impl #impl_generics #lang_path::types::SoaValue for #name #ty_generics #where_clause{
                type SoaBuffer = #soa_proxy_name #ty_generics;
            }
            impl #impl_generics #lang_path::types::SoaBufferProxy for #soa_proxy_name #ty_generics #where_clause{
                type Value = #name #ty_generics;

                #[allow(unused_assignments)]
                fn from_soa_storage(
                    ___storage: #crate_path::resource::ByteBufferVar,
                    ___meta: Expr<#lang_path::soa::SoaMetadata>,
                    ___global_offset: usize,
                ) -> Self {
                    use #lang_path::types::SoaBufferProxy;
                    let mut ___i = 0usize;
                    #(
                        let #field_names = <#field_types as #lang_path::types::SoaValue>::SoaBuffer::from_soa_storage(
                            ___storage.clone(),
                            ___meta.clone(),
                            ___global_offset + ___i,
                        );
                        ___i += <<#field_types as #lang_path::types::SoaValue>::SoaBuffer as #lang_path::types::SoaBufferProxy>::num_buffers();
                    )*
                    Self{
                        #(#field_names),*
                    }
                }
                fn num_buffers() -> usize {
                    [#( <#field_types as #lang_path::types::SoaValue>::SoaBuffer::num_buffers()),*].iter().sum()
                }
            }
            impl #impl_generics #lang_path::index::IndexRead for #soa_proxy_name #ty_generics #where_clause{
                type Element = #name #ty_generics;
                fn read<I:#lang_path::index::IntoIndex>(&self, ___i: I) -> #lang_path::types::Expr<Self::Element> {
                    let ___i = ___i.to_u64();
                    use #lang_path::FromNode;
                    #(
                        let #field_names = self.#field_names.read(___i);
                    )*
                    Expr::<Self::Element>::from_node(#lang_path::__compose::<Self::Element>(&[ #( #lang_path::ToNode::node(&#field_names).get() ),* ]).into())
                }
            }
            impl #impl_generics #lang_path::index::IndexWrite for #soa_proxy_name #ty_generics #where_clause{
                fn write<I:#lang_path::index::IntoIndex, V: #lang_path::types::AsExpr<Value = Self::Element>>(&self, i: I, value: V) {
                    let i = i.to_u64();
                    let v = value.as_expr();
                    #(
                        self.#field_names.write(i, v.#field_names);
                    )*
                }
            }
        )
    }
    pub fn derive_iotexel(&mut self, item: &Item) -> TokenStream {
        match item {
            Item::Struct(struct_) => self.derive_iotexel_for_struct(struct_),
            _ => todo!(),
        }
    }
    pub fn derive_iotexel_for_struct(&mut self, struct_: &ItemStruct) -> TokenStream {
        let attrs = self.parse_luisa_attributes(&struct_.attrs);
        self.set_crate_path_from_attrs(&attrs);
        let span = struct_.span();
        let resource_path = self.resource_path();
        let lang_path = self.lang_path();
        // Make sure that the struct has repr(transparent).
        let mut has_repr_transparent = false;
        for Attribute { meta, .. } in &struct_.attrs {
            if let syn::Meta::List(list) = meta {
                let path = &list.path;
                if path.is_ident("repr") {
                    for tok in list.tokens.clone().into_iter() {
                        if let TokenTree::Ident(ident) = tok {
                            if ident == "transparent" {
                                has_repr_transparent = true;
                            }
                        }
                    }
                }
            }
        }
        if !has_repr_transparent {
            panic!("Struct must have #[repr(transparent)] attribute");
        }
        let generics = &struct_.generics;
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
        let syn::Fields::Named(syn::FieldsNamed { named, .. }) = &struct_.fields else {
            panic!("IoTexel derive currently only supports named fields.")
        };
        assert_eq!(named.len(), 1);
        let syn::Field { ident, ty, .. } = &named[0];
        let ident = ident.as_ref().unwrap();
        let struct_name = &struct_.ident;
        let struct_comps_name =
            syn::Ident::new(&format!("{}Comps", struct_name), struct_name.span());
        quote_spanned! {span=>
            impl #impl_generics #resource_path::IoTexel for #struct_name<#ty_generics> #where_clause {
                type RwType = <#ty as #resource_path::IoTexel>::RwType;
                fn pixel_format(storage: #resource_path::PixelStorage) -> #resource_path::PixelFormat {
                    <#ty as #resource_path::IoTexel>::pixel_format(storage)
                }
                fn convert_from_read(texel: #lang_path::types::Expr<Self::RwType>) -> #lang_path::types::Expr<Self> {
                    #struct_name::from_comps_expr(#struct_comps_name {
                        #ident: <#ty as #resource_path::IoTexel>::convert_from_read(texel),
                    })
                }
                fn convert_to_write(value: #lang_path::types::Expr<Self>) -> #lang_path::types::Expr<Self::RwType> {
                    <#ty as #resource_path::IoTexel>::convert_to_write(value.#ident)
                }
            }
        }
    }
    pub fn derive_value(&mut self, item: &Item) -> TokenStream {
        match item {
            Item::Struct(struct_) => self.derive_value_for_struct(struct_),
            Item::Enum(enum_) => self.derive_value_for_enum(enum_),
            _ => todo!(),
        }
    }
    pub fn derive_value_for_enum(&mut self, enum_: &ItemEnum) -> TokenStream {
        let attrs = self.parse_luisa_attributes(&enum_.attrs);
        self.set_crate_path_from_attrs(&attrs);
        let repr = enum_
            .attrs
            .iter()
            .find_map(|attr| {
                let meta = &attr.meta;
                match meta {
                    syn::Meta::List(list) => {
                        let path = &list.path;
                        if path.is_ident("repr") {
                            list.parse_args::<Ident>().ok()
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            })
            .expect("Enum must have repr attribute.");
        let span = enum_.span();
        let crate_path = &self.crate_path;
        let lang_path = self.lang_path();
        let name = &enum_.ident;
        let vis = &enum_.vis;
        let expr_proxy_name = syn::Ident::new(&format!("{}Expr", name), name.span());
        let var_proxy_name = syn::Ident::new(&format!("{}Var", name), name.span());
        let atomic_ref_proxy_name = syn::Ident::new(&format!("{}AtomicRef", name), name.span());
        let as_repr = syn::Ident::new(&format!("as_{}", repr), repr.span());
        if !(["bool", "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64"]
            .contains(&&*repr.to_string()))
        {
            panic!("Enum repr must be one of bool, u8, u16, u32, u64, i8, i16, i32, i64");
        }
        quote_spanned! {span=>
            impl #lang_path::types::Value for #name {
                type Expr = #expr_proxy_name;
                type Var = #var_proxy_name;
                type AtomicRef = #atomic_ref_proxy_name;

                fn expr(self) -> Expr<Self> {
                    let node = #lang_path::__current_scope(|s| s.const_(<#repr as #lang_path::types::core::Primitive>::const_(&(self as #repr))));
                    <Expr::<Self> as #lang_path::FromNode>::from_node(node.into())
                }
            }
            impl #lang_path::ir::TypeOf for #name {
                fn type_() -> #lang_path::ir::CArc<#lang_path::ir::Type> {
                    <#repr as #lang_path::ir::TypeOf>::type_()
                }
            }

            #crate_path::impl_simple_expr_proxy!(#expr_proxy_name for #name);
            #crate_path::impl_simple_var_proxy!(#var_proxy_name for #name);
            #crate_path::impl_simple_atomic_ref_proxy!(#atomic_ref_proxy_name for #name);

            impl #expr_proxy_name {
                #vis fn #as_repr(&self) -> #lang_path::types::Expr<#repr> {
                    use #lang_path::ToNode;
                    use #lang_path::types::ExprProxy;
                    #lang_path::FromNode::from_node(self.as_expr_from_proxy().node())
                }
            }
            impl #var_proxy_name {
                #vis fn #as_repr(&self) -> #lang_path::types::Var<#repr> {
                    use #lang_path::ToNode;
                    use #lang_path::types::VarProxy;
                    #lang_path::FromNode::from_node(self.as_var_from_proxy().node())
                }
            }
            impl #atomic_ref_proxy_name {
                #vis fn #as_repr(&self) -> #lang_path::types::AtomicRef<#repr> {
                    use #lang_path::ToNode;
                    use #lang_path::types::AtomicRefProxy;
                    #lang_path::FromNode::from_node(self.as_atomic_ref_from_proxy().node())
                }
            }
        }
    }
    pub fn derive_value_for_struct(&mut self, struct_: &ItemStruct) -> TokenStream {
        let attrs = self.parse_luisa_attributes(&struct_.attrs);
        self.set_crate_path_from_attrs(&attrs);
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
        let fields: Vec<_> = struct_.fields.iter().collect();
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
                        let node = #lang_path::__compose::<#name #ty_generics>(&[ #( #lang_path::ToNode::node(&ctor.#field_names.as_expr()).get() ),* ]);
                        <#lang_path::types::Expr::<#name> as #lang_path::FromNode>::from_node(node.into())
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
            impl #impl_generics #lang_path::types::AtomicRefProxy for #atomic_ref_proxy_name #ty_generics #where_clause {
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
                        let node = #lang_path::__compose::<#name #ty_generics>(&[ #( #lang_path::ToNode::node(&#field_names.as_expr()).get() ),* ]);
                        <#lang_path::types::Expr::<#name> as #lang_path::FromNode>::from_node(node.into())
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
    pub fn derive_aggregate_for_struct(&mut self, struct_: &ItemStruct) -> TokenStream {
        let attrs = self.parse_luisa_attributes(&struct_.attrs);
        self.set_crate_path_from_attrs(&attrs);
        let span = struct_.span();
        let lang_path = self.lang_path();
        let name = &struct_.ident;
        let fields: Vec<_> = struct_.fields.iter().collect();
        let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();
        let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
        quote_spanned!(span=>
            impl #lang_path::Aggregate for #name {
                fn to_nodes(&self, nodes: &mut Vec<#lang_path::SafeNodeRef>) {
                    #(self.#field_names.to_nodes(nodes);)*
                }
                fn from_nodes<__I: Iterator<Item = #lang_path::SafeNodeRef>>(iter: &mut __I) -> Self {
                    #(let #field_names = <#field_types as #lang_path::Aggregate>::from_nodes(iter);)*
                    Self{
                        #(#field_names,)*
                    }
                }
            }
        )
    }
    pub fn derive_aggregate_for_enum(&mut self, enum_: &ItemEnum) -> TokenStream {
        let attrs = self.parse_luisa_attributes(&enum_.attrs);
        self.set_crate_path_from_attrs(&attrs);
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
                fn from_nodes<I: Iterator<Item = #lang_path::SafeNodeRef>>(iter: &mut I) -> Self {
                    let variant = iter.next().unwrap().get();
                    let variant = variant.unwrap_user_data::<usize>();
                    match variant{
                        #(#from_nodes)*
                        _=> panic!("invalid variant"),
                    }
                }
                #[allow(non_snake_case)]
                fn to_nodes(&self, nodes: &mut Vec<#lang_path::SafeNodeRef>){
                    match self {
                        #(#to_nodes)*
                    }
                }
            }
        }
    }
    pub fn derive_aggregate(&mut self, item: &Item) -> TokenStream {
        match item {
            Item::Struct(struct_) => self.derive_aggregate_for_struct(struct_),
            Item::Enum(enum_) => self.derive_aggregate_for_enum(enum_),
            _ => todo!(),
        }
    }
}
