use proc_macro2::TokenStream;
use proc_macro_error::emit_error;
use quote::{quote, quote_spanned};
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;
use syn::visit_mut::*;
use syn::*;

// TODO: Impl let mut -> let = .var() <- Don't
// TODO: Impl x as f32 -> .cast()  <- Don't
// TOOD: Impl switch! macro.  <- Don't

struct TraceVisitor {
    trait_path: TokenStream,
    flow_path: TokenStream,
    debug_path: TokenStream,
    is_last_stmt: bool,
}

impl VisitMut for TraceVisitor {
    fn visit_block_mut(&mut self, node: &mut Block) {
        let len = node.stmts.len();
        if len > 0 {
            for stmt in node.stmts[0..len - 1].iter_mut() {
                let old = self.is_last_stmt;
                self.is_last_stmt = false;
                self.visit_stmt_mut(stmt);
                self.is_last_stmt = old;
            }
            let old = self.is_last_stmt;
            self.is_last_stmt = true;
            self.visit_stmt_mut(node.stmts.last_mut().unwrap());
            self.is_last_stmt = old;
        }
    }
    fn visit_stmt_mut(&mut self, node: &mut Stmt) {
        let span = node.span();
        match node {
            Stmt::Expr(_, semi) => {
                if !self.is_last_stmt && semi.is_none() {
                    *semi = Some(Token![;](span));
                }
            }
            Stmt::Macro(stmt) => {
                let path = &stmt.mac.path;
                if path.leading_colon.is_none()
                    && path.segments.len() == 1
                    && path.segments[0].arguments.is_none()
                {
                    let ident = &path.segments[0].ident;
                    if *ident == "escape" {
                        let tokens = &stmt.mac.tokens;
                        *node = parse_quote_spanned! {span=>
                            #tokens
                        };
                        return;
                    }
                }
            }
            Stmt::Item(Item::Const(c)) => match c.expr.as_ref() {
                Expr::Lit(_) => {}
                _ => {
                    panic!("Please wrap CONST statments with `escape!`");
                }
            },
            _ => {}
        }
        visit_stmt_mut(self, node);
    }
    fn visit_expr_mut(&mut self, node: &mut Expr) {
        let flow_path = &self.flow_path;
        let trait_path = &self.trait_path;
        let debug_path = &self.debug_path;
        let span = node.span();

        match node {
            Expr::Assign(expr) => {
                let left = &expr.left;
                let right = &expr.right;
                if let Expr::Unary(ExprUnary {
                    op: UnOp::Deref(_),
                    expr,
                    ..
                }) = &**left
                {
                    *node = parse_quote_spanned! {span=>
                        {
                            use #trait_path::StoreMaybeExpr;
                            (#expr).__store(#right)
                        }
                    }
                }
            }
            Expr::If(expr) => {
                let cond = &expr.cond;
                let then_branch = &expr.then_branch;
                let else_branch = &expr.else_branch;
                if let Expr::Let(_) = **cond {
                } else if let Some((_, else_branch)) = else_branch {
                    *node = parse_quote_spanned! {span=>
                        #debug_path::with_lineno("if",
                            file!(),
                            line!(),
                            column!(),
                            || <_ as #trait_path::SelectMaybeExpr<_>>::if_then_else(#cond, || #then_branch, || #else_branch))
                    }
                } else {
                    *node = parse_quote_spanned! {span=>
                        #debug_path::with_lineno("if",
                            file!(),
                            line!(),
                            column!(),
                            || <_ as #trait_path::ActivateMaybeExpr>::activate(#cond, || #then_branch))
                    }
                }
            }
            Expr::While(expr) => {
                let cond = &expr.cond;
                let body = &expr.body;
                *node = parse_quote_spanned! {span=>
                    #debug_path::with_lineno("while",
                            file!(),
                            line!(),
                            column!(), ||<_ as #trait_path::LoopMaybeExpr>::while_loop(|| #cond, || #body))
                }
            }
            Expr::Loop(expr) => {
                let body = &expr.body;
                *node = parse_quote_spanned! {span=>
                    #debug_path::with_lineno("loop",
                        file!(),
                        line!(),
                        column!(), || #flow_path::loop_(|| #body))
                }
            }
            Expr::ForLoop(expr) => {
                let pat = &expr.pat;
                let body = &expr.body;
                let expr = &expr.expr;
                if let Expr::Range(range) = &**expr {
                    let attrs = &range.attrs;
                    // check if #[unroll] is present
                    let unroll = attrs.iter().any(|attr| attr.path().is_ident("unroll"));
                    if unroll {
                        *node = parse_quote_spanned! {span=>
                            #debug_path::with_lineno("for range",
                                file!(),
                                line!(),
                                column!(), || #range.for_each(|#pat| #body))
                        }
                    } else {
                        *node = parse_quote_spanned! {span=>
                            #debug_path::with_lineno("for range",
                                file!(),
                                line!(),
                                column!(), ||#flow_path::for_range(#range, |#pat| #body))
                        }
                    }
                }
            }
            // Expr::Unary(op) => {
            //     if let ExprUnary {
            //         op: UnOp::Deref(_),
            //         expr,
            //         ..
            //     } = op
            //     {
            //         *node = parse_quote_spanned! {span=>
            //             #expr.load()
            //         }
            //     }
            // }
            Expr::Binary(expr) => {
                let left = &expr.left;
                let right = &expr.right;

                if let Expr::Unary(ExprUnary {
                    op: UnOp::Deref(_),
                    expr: left,
                    ..
                }) = &**left
                {
                    let op_fn_str = match &expr.op {
                        BinOp::AddAssign(_) => "__add_assign",
                        BinOp::SubAssign(_) => "__sub_assign",
                        BinOp::MulAssign(_) => "__mul_assign",
                        BinOp::DivAssign(_) => "__div_assign",
                        BinOp::RemAssign(_) => "__rem_assign",
                        BinOp::BitAndAssign(_) => "__bitand_assign",
                        BinOp::BitOrAssign(_) => "__bitor_assign",
                        BinOp::BitXorAssign(_) => "__bitxor_assign",
                        BinOp::ShlAssign(_) => "__shl_assign",
                        BinOp::ShrAssign(_) => "__shr_assign",
                        _ => "",
                    };
                    let op_trait_str = match &expr.op {
                        BinOp::AddAssign(_) => "AddAssignMaybeExpr",
                        BinOp::SubAssign(_) => "SubAssignMaybeExpr",
                        BinOp::MulAssign(_) => "MulAssignMaybeExpr",
                        BinOp::DivAssign(_) => "DivAssignMaybeExpr",
                        BinOp::RemAssign(_) => "RemAssignMaybeExpr",
                        BinOp::BitAndAssign(_) => "BitAndAssignMaybeExpr",
                        BinOp::BitOrAssign(_) => "BitOrAssignMaybeExpr",
                        BinOp::BitXorAssign(_) => "BitXorAssignMaybeExpr",
                        BinOp::ShlAssign(_) => "ShlAssignMaybeExpr",
                        BinOp::ShrAssign(_) => "ShrAssignMaybeExpr",
                        _ => "",
                    };
                    if !op_fn_str.is_empty() {
                        let op_fn = Ident::new(op_fn_str, expr.op.span());
                        let op_trait = Ident::new(op_trait_str, expr.op.span());
                        *node = parse_quote_spanned! {span=>
                            {
                                use #trait_path::#op_trait;
                                #[allow(unused_parens)]
                                (#left).#op_fn(#right)
                            }
                        };
                        visit_expr_mut(self, node);
                        return;
                    }
                }
                let left = if let Expr::Paren(ExprParen { expr, .. }) = &**left {
                    expr
                } else {
                    left
                };
                let right = if let Expr::Paren(ExprParen { expr, .. }) = &**right {
                    expr
                } else {
                    right
                };
                let op_fn_str = match &expr.op {
                    BinOp::Add(_) => "__add",
                    BinOp::Sub(_) => "__sub",
                    BinOp::Mul(_) => "__mul",
                    BinOp::Div(_) => "__div",
                    BinOp::Rem(_) => "__rem",
                    BinOp::BitAnd(_) => "__bitand",
                    BinOp::BitOr(_) => "__bitor",
                    BinOp::BitXor(_) => "__bitxor",
                    BinOp::Shl(_) => "__shl",
                    BinOp::Shr(_) => "__shr",

                    BinOp::And(_) => "and",
                    BinOp::Or(_) => "or",

                    BinOp::Eq(_) => "__eq",
                    BinOp::Ne(_) => "__ne",
                    BinOp::Lt(_) => "__lt",
                    BinOp::Le(_) => "__le",
                    BinOp::Ge(_) => "__ge",
                    BinOp::Gt(_) => "__gt",

                    _ => "",
                };
                let op_trait_str = match &expr.op {
                    BinOp::Add(_) => "AddMaybeExpr",
                    BinOp::Sub(_) => "SubMaybeExpr",
                    BinOp::Mul(_) => "MulMaybeExpr",
                    BinOp::Div(_) => "DivMaybeExpr",
                    BinOp::Rem(_) => "RemMaybeExpr",
                    BinOp::BitAnd(_) => "BitAndMaybeExpr",
                    BinOp::BitOr(_) => "BitOrMaybeExpr",
                    BinOp::BitXor(_) => "BitXorMaybeExpr",
                    BinOp::Shl(_) => "ShlMaybeExpr",
                    BinOp::Shr(_) => "ShrMaybeExpr",
                    BinOp::And(_) | BinOp::Or(_) => "LazyBoolMaybeExpr",
                    BinOp::Eq(_) | BinOp::Ne(_) => "EqMaybeExpr",
                    BinOp::Lt(_) | BinOp::Le(_) | BinOp::Ge(_) | BinOp::Gt(_) => "CmpMaybeExpr",
                    _ => "",
                };

                if !op_fn_str.is_empty() {
                    let op_fn = Ident::new(op_fn_str, expr.op.span());
                    let op_trait = Ident::new(op_trait_str, expr.op.span());
                    if let BinOp::And(_) | BinOp::Or(_) = &expr.op {
                        *node = parse_quote_spanned! {span=>
                            <_ as #trait_path::#op_trait<_, _>>::#op_fn(#left, || #right)
                        };
                    } else {
                        *node = parse_quote_spanned! {span=>
                            <_ as #trait_path::#op_trait<_, _>>::#op_fn(#left, #right)
                        };
                    }
                }
            }
            Expr::Return(expr) => {
                if let Some(expr) = &expr.expr {
                    *node = parse_quote_spanned! {span=>
                        #flow_path::return_v(#expr)
                    };
                } else {
                    *node = parse_quote_spanned! {span=>
                        #flow_path::return_()
                    };
                }
            }
            Expr::Continue(expr) => {
                if expr.label.is_some() {
                    emit_error!(
                        span,
                        "continue expression tracing with labels is not supported\nif this is intended to be a normal loop, use the `escape!` macro"
                    );
                } else {
                    *node = parse_quote_spanned! {span=>
                        #flow_path::continue_()
                    };
                }
            }
            Expr::Break(expr) => {
                if expr.label.is_some() {
                    emit_error!(
                        span,
                        "break expression tracing with labels is not supported\nif this is intended to be a normal loop, use the `escape!` macro"
                    );
                } else {
                    *node = parse_quote_spanned! {span=>
                        #flow_path::break_()
                    };
                }
            }
            Expr::Macro(expr) => {
                let path = &expr.mac.path;
                if path.leading_colon.is_none()
                    && path.segments.len() == 1
                    && path.segments[0].arguments.is_none()
                {
                    let ident = &path.segments[0].ident;
                    if *ident == "escape" {
                        let tokens = &expr.mac.tokens;
                        *node = parse_quote_spanned! {span=>
                            #tokens
                        };
                        return;
                    }
                }
            }
            _ => {}
        }
        visit_expr_mut(self, node);
    }
}
#[allow(dead_code)]
struct TrackInput {
    crate_path: OptCratePath,
    arrow: Option<Token![=>]>,
    body: TokenStream,
}
impl Parse for TrackInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let crate_path: OptCratePath = input.parse()?;
        let arrow = if crate_path.0.is_some() {
            Some(input.parse()?)
        } else {
            None
        };
        let body = input.parse()?;
        Ok(Self {
            crate_path,
            arrow,
            body,
        })
    }
}
#[proc_macro]
pub fn track(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let TrackInput {
        crate_path, body, ..
    } = parse_macro_input!(input as TrackInput);

    let crate_path = if let Some(crate_path) = crate_path.0 {
        let path = crate_path.path;
        quote!(#path)
    } else {
        quote!(::luisa_compute)
    };
    let input = quote!({ #body });
    let input = proc_macro::TokenStream::from(input);
    track_impl(parse_macro_input!(input as Expr), &crate_path).into()
}

// #[proc_macro]
// pub fn escape(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
//     input
// }
#[allow(dead_code)]
struct CratePath {
    crate_: Token![crate],
    eq: Token![=],
    path: TokenStream,
}
impl Parse for CratePath {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let crate_ = input.parse()?;
        let eq = input.parse()?;
        let path: LitStr = input.parse()?;
        let path = path.parse().unwrap();
        Ok(Self { crate_, eq, path })
    }
}
struct OptCratePath(Option<CratePath>);
impl Parse for OptCratePath {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let crate_path = if input.peek(Token![crate]) {
            Some(input.parse()?)
        } else {
            None
        };
        Ok(Self(crate_path))
    }
}
#[proc_macro_attribute]
pub fn tracked(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let crate_path = syn::parse_macro_input!(attr as OptCratePath);
    let crate_path = if let Some(crate_path) = crate_path.0 {
        let path = crate_path.path;
        quote!(#path)
    } else {
        quote!(luisa_compute)
    };

    let item = syn::parse_macro_input!(item as ItemFn);
    let body = &item.block;
    let body_span = body.span();
    let body = proc_macro::TokenStream::from(quote!({ #body }));
    let body = track_impl(parse_macro_input!(body as Expr), &crate_path);
    let body = quote_spanned! {body_span=>
        {
            let __fn_name = {
                fn f() {}
                fn type_name_of<T>(_: T) -> &'static str {
                    std::any::type_name::<T>()
                }
                let name = type_name_of(f);
                name.strip_suffix("::f").unwrap()
            };
            #crate_path::lang::debug::comment(&format!("begin fn {} at {}:{}:{}", __fn_name, file!(), line!(), column!()));
            #[allow(clippy::let_unit_value)]
            let __ret = #body;
            #[allow(unreachable_code)]
            {
                #crate_path::lang::debug::comment(&format!("end fn {} at {}:{}:{}", __fn_name, file!(), line!(), column!()));
                __ret
            }
        }
    };
    let attrs = &item.attrs;
    let sig = &item.sig;
    let vis = &item.vis;
    quote_spanned!(item.span()=> #(#attrs)* #vis #sig { #body }).into()
}

fn track_impl(mut ast: Expr, crate_path: &TokenStream) -> TokenStream {
    (TraceVisitor {
        is_last_stmt: false,
        flow_path: quote!(#crate_path::lang::control_flow),
        trait_path: quote!(#crate_path::lang::ops),
        debug_path: quote!(#crate_path::lang::debug),
    })
    .visit_expr_mut(&mut ast);

    quote!(#ast)
}
