use proc_macro2::TokenStream;
use proc_macro_error::emit_error;
use quote::{quote, quote_spanned};
use syn::spanned::Spanned;
use syn::visit_mut::*;
use syn::*;

// TODO: Impl let mut -> let = .var()
// TODO: Impl x as f32 -> .cast()
// TOOD: Impl switch! macro.

#[cfg(test)]
use pretty_assertions::assert_eq;

struct TraceVisitor {
    trait_path: TokenStream,
    flow_path: TokenStream,
}

impl VisitMut for TraceVisitor {
    fn visit_block_mut(&mut self, node: &mut Block) {
        let len = node.stmts.len();
        if len > 0 {
            for stmt in node.stmts[0..len - 1].iter_mut() {
                self.visit_stmt_mut(stmt);
            }
            visit_stmt_mut(self, node.stmts.last_mut().unwrap());
        }
    }
    fn visit_stmt_mut(&mut self, node: &mut Stmt) {
        let span = node.span();
        match node {
            Stmt::Expr(_, semi) => {
                if semi.is_none() {
                    *semi = Some(Token![;](span));
                }
            }
            _ => {}
        }
        visit_stmt_mut(self, node);
    }
    fn visit_expr_mut(&mut self, node: &mut Expr) {
        let flow_path = &self.flow_path;
        let trait_path = &self.trait_path;
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
                        <_ as #trait_path::StoreMaybeExpr>::store(#expr, #right)
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
                        <_ as #trait_path::SelectMaybeExpr<_>>::if_then_else(#cond, || #then_branch, || #else_branch)
                    }
                } else {
                    *node = parse_quote_spanned! {span=>
                        <_ as #trait_path::ActivateMaybeExpr>::activate(#cond, || #then_branch)
                    }
                }
            }
            Expr::While(expr) => {
                let cond = &expr.cond;
                let body = &expr.body;
                *node = parse_quote_spanned! {span=>
                    <_ as #trait_path::LoopMaybeExpr>::while_loop(|| #cond, || #body)
                }
            }
            Expr::Loop(expr) => {
                let body = &expr.body;
                *node = parse_quote_spanned! {span=>
                    #flow_path::loop_(|| #body)
                }
            }
            Expr::ForLoop(expr) => {
                let pat = &expr.pat;
                let body = &expr.body;
                let expr = &expr.expr;
                if let Expr::Range(range) = &**expr {
                    *node = parse_quote_spanned! {span=>
                        #flow_path::for_range(#range, |#pat| #body)
                    }
                }
            }
            Expr::Binary(expr) => {
                let op_fn_str = match &expr.op {
                    BinOp::Eq(_) => "eq",
                    BinOp::Ne(_) => "ne",

                    BinOp::And(_) => "and",
                    BinOp::Or(_) => "or",

                    BinOp::Lt(_) => "lt",
                    BinOp::Le(_) => "le",
                    BinOp::Ge(_) => "ge",
                    BinOp::Gt(_) => "gt",
                    _ => "",
                };

                if !op_fn_str.is_empty() {
                    let left = &expr.left;
                    let right = &expr.right;
                    let op_fn = Ident::new(op_fn_str, expr.op.span());
                    if op_fn_str == "eq" || op_fn_str == "ne" {
                        *node = parse_quote_spanned! {span=>
                            <_ as #trait_path::EqMaybeExpr<_, _>>::#op_fn(#left, #right)
                        }
                    } else if op_fn_str == "and" || op_fn_str == "or" {
                        *node = parse_quote_spanned! {span=>
                            <_ as #trait_path::LazyBoolMaybeExpr<_>>::#op_fn(#left, || #right)
                        }
                    } else {
                        *node = parse_quote_spanned! {span=>
                            <_ as #trait_path::CmpMaybeExpr<_, _>>::#op_fn(#left, #right)
                        }
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

#[proc_macro]
pub fn track(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = TokenStream::from(input);
    let input = quote!({ #input });
    let input = proc_macro::TokenStream::from(input);
    track_impl(parse_macro_input!(input as Expr)).into()
}

#[proc_macro_attribute]
pub fn tracked(
    _attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let item = syn::parse_macro_input!(item as ItemFn);
    let body = &item.block;
    let body = proc_macro::TokenStream::from(quote!({ #body }));
    let body = track_impl(parse_macro_input!(body as Expr));
    let attrs = &item.attrs;
    let sig = &item.sig;
    let vis = &item.vis;
    quote_spanned!(item.span()=> #(#attrs)* #vis #sig { #body }).into()
}

fn track_impl(mut ast: Expr) -> TokenStream {
    (TraceVisitor {
        flow_path: quote!(::luisa_compute::lang::control_flow),
        trait_path: quote!(::luisa_compute::lang::ops),
    })
    .visit_expr_mut(&mut ast);

    quote!(#ast)
}

#[test]
fn test_macro() {
    #[rustfmt::skip]
    assert_eq!(
        track_impl(parse_quote!(|x: Expr<f32>, y: Expr<f32>| {
            if x > y {
                x * y
            } else {
                y * x + (x / 32.0 * PI).sin()
            }
        }))
        .to_string(),
        quote!(|x: Expr<f32>, y: Expr<f32>| {
            <_ as ::luisa_compute::lang::maybe_expr::BoolIfElseMaybeExpr<_> >::if_then_else(
                <_ as ::luisa_compute::lang::maybe_expr::PartialOrdMaybeExpr<_> >::gt(x, y),
                | | { x * y },
                | | { y * x + (x / 32.0 * PI).sin() }
            )
        })
        .to_string()
    );
}
