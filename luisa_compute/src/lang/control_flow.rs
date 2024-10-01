use crate::internal_prelude::*;
use ir::SwitchCase;

use super::debug::__unreachable_typed;

/**
 * If you want rustfmt to format your code, use if_!(cond, { .. }, { .. })
 * or if_!(cond, { .. }, else, {...}) instead of if_!(cond, { .. }, else
 * {...}).
 *
 */
#[macro_export]
macro_rules! if_ {
    ($cond:expr, $then:block, else $else_:block) => {
        <_ as $crate::lang::ops::SelectMaybeExpr<_>>::if_then_else($cond, || $then, || $else_)
    };
    ($cond:expr, $then:block, else, $else_:block) => {
        <_ as $crate::lang::ops::SelectMaybeExpr<_>>::if_then_else($cond, || $then, || $else_)
    };
    ($cond:expr, $then:block, $else_:block) => {
        <_ as $crate::lang::ops::SelectMaybeExpr<_>>::if_then_else($cond, || $then, || $else_)
    };
    ($cond:expr, $then:block) => {
        <_ as $crate::lang::ops::ActivateMaybeExpr>::activate($cond, || $then)
    };
}
#[macro_export]
macro_rules! while_ {
    ($cond:expr,$body:block) => {
        <_ as $crate::lang::ops::LoopMaybeExpr>::while_loop(|| $cond, || $body)
    };
}
#[macro_export]
macro_rules! loop_ {
    ($body:block) => {
        $crate::while_!(true.expr(), $body)
    };
}

#[inline]
pub fn break_() {
    __current_scope(|b| {
        b.break_();
    });
}

#[inline]
pub fn continue_() {
    __current_scope(|b| {
        b.continue_();
    });
}

pub fn return_v<T: NodeLike>(v: T) {
    let v = v.node().get();
    with_recorder(|r| {
        if r.building_kernel {
            panic!("cannot return value from kernel!");
        }
        if r.callable_ret_type.is_none() {
            r.callable_ret_type = Some(v.type_().clone());
        } else {
            assert!(
                luisa_compute_ir::context::is_type_equal(
                    r.callable_ret_type.as_ref().unwrap(),
                    v.type_()
                ),
                "return type mismatch"
            );
        }
    });
    __current_scope(|b| {
        b.return_(v);
    });
}
pub fn return_() {
    with_recorder(|r| {
        if !r.building_kernel {
            if r.callable_ret_type.is_none() {
                r.callable_ret_type = Some(Type::void());
            } else {
                assert!(luisa_compute_ir::context::is_type_equal(
                    r.callable_ret_type.as_ref().unwrap(),
                    &Type::void()
                ));
            }
        }
    });
    __current_scope(|b| {
        b.return_(INVALID_REF);
    });
}

pub fn if_then_else<R: Aggregate>(
    cond: Expr<bool>,
    then: impl Fn() -> R,
    else_: impl Fn() -> R,
) -> R {
    let cond = cond.node().get();
    with_recorder(|r| {
        let pools = r.pools.clone();
        let s = &mut r.scopes;
        s.push(IrBuilder::new(pools));
    });
    let then = then();
    let then_nodes = then
        .to_vec_nodes()
        .into_iter()
        .map(|x| x.get())
        .collect::<Vec<_>>();
    let then_block = with_recorder(|r| {
        let pools = r.pools.clone();
        let s = &mut r.scopes;
        let then_block = s.pop().unwrap().finish();
        s.push(IrBuilder::new(pools));
        r.add_block_to_inaccessible(&then_block);
        then_block
    });
    let else_ = else_();
    let else_nodes = else_
        .to_vec_nodes()
        .into_iter()
        .map(|x| x.get())
        .collect::<Vec<_>>();
    let else_block = with_recorder(|r| {
        let s = &mut r.scopes;
        let else_block = s.pop().unwrap().finish();
        r.add_block_to_inaccessible(&else_block);
        else_block
    });
    __current_scope(|b| {
        b.if_(cond, then_block, else_block);
    });
    assert_eq!(then_nodes.len(), else_nodes.len());
    let phis = then_nodes
        .iter()
        .zip(else_nodes.iter())
        .map(|(then, else_)| {
            let incomings = vec![
                PhiIncoming {
                    value: *then,
                    block: then_block,
                },
                PhiIncoming {
                    value: *else_,
                    block: else_block,
                },
            ];
            assert_eq!(then.type_(), else_.type_());
            let phi = __current_scope(|b| b.phi(&incomings, then.type_().clone()));
            phi.into()
        })
        .collect::<Vec<_>>();
    R::from_vec_nodes(phis)
}

pub fn select<A: Aggregate>(mask: Expr<bool>, a: A, b: A) -> A {
    let a_nodes = a
        .to_vec_nodes()
        .into_iter()
        .map(|x| x.get())
        .collect::<Vec<_>>();
    let b_nodes = b
        .to_vec_nodes()
        .into_iter()
        .map(|x| x.get())
        .collect::<Vec<_>>();
    let mask = mask.node().get();
    assert_eq!(a_nodes.len(), b_nodes.len());
    let mut ret = vec![];
    __current_scope(|b| {
        for (a_node, b_node) in a_nodes.into_iter().zip(b_nodes.into_iter()) {
            assert_eq!(a_node.type_(), b_node.type_());
            assert!(!a_node.is_local(), "cannot select local variables");
            assert!(!b_node.is_local(), "cannot select local variables");
            if a_node.is_user_data() || b_node.is_user_data() {
                assert!(
                    a_node.is_user_data() && b_node.is_user_data(),
                    "cannot select user data and non-user data"
                );
                let a_data = a_node.get_user_data();
                let b_data = b_node.get_user_data();
                if a_data != b_data {
                    panic!("cannot select different user data");
                }
                ret.push(a_node);
            } else {
                ret.push(b.call(
                    Func::Select,
                    &[mask, a_node, b_node],
                    a_node.type_().clone(),
                ));
            }
        }
    });
    let ret = ret.into_iter().map(|x| x.into()).collect::<Vec<_>>();
    A::from_vec_nodes(ret)
}

pub fn generic_loop(
    mut cond: impl FnMut() -> Expr<bool>,
    mut body: impl FnMut(),
    mut update: impl FnMut(),
) {
    with_recorder(|r| {
        let pools = r.pools.clone();
        let s = &mut r.scopes;
        s.push(IrBuilder::new(pools));
    });
    let cond_v = cond().node().get();
    let prepare = with_recorder(|r| {
        let pools = r.pools.clone();
        let s = &mut r.scopes;
        let prepare = s.pop().unwrap().finish();
        s.push(IrBuilder::new(pools));
        r.add_block_to_inaccessible(&prepare);
        prepare
    });
    body();
    let body = with_recorder(|r| {
        let pools = r.pools.clone();
        let s = &mut r.scopes;
        let body = s.pop().unwrap().finish();
        s.push(IrBuilder::new(pools));
        r.add_block_to_inaccessible(&body);
        body
    });
    update();
    let update = with_recorder(|r| {
        let s = &mut r.scopes;
        let update_block = s.pop().unwrap().finish();
        r.add_block_to_inaccessible(&update_block);
        update_block
    });
    __current_scope(|b| {
        b.generic_loop(prepare, cond_v, body, update);
    });
}

pub trait ForLoopRange {
    type Element: Value;
    fn start(&self) -> SafeNodeRef;
    fn end(&self) -> SafeNodeRef;
    fn end_inclusive(&self) -> bool;
}
macro_rules! impl_range {
    ($t:ty) => {
        impl ForLoopRange for std::ops::RangeInclusive<$t> {
            type Element = $t;
            fn start(&self) -> SafeNodeRef {
                (*self.start()).expr().node()
            }
            fn end(&self) -> SafeNodeRef {
                (*self.end()).expr().node()
            }
            fn end_inclusive(&self) -> bool {
                true
            }
        }
        impl ForLoopRange for std::ops::RangeInclusive<Expr<$t>> {
            type Element = $t;
            fn start(&self) -> SafeNodeRef {
                self.start().node()
            }
            fn end(&self) -> SafeNodeRef {
                self.end().node()
            }
            fn end_inclusive(&self) -> bool {
                true
            }
        }
        impl ForLoopRange for std::ops::Range<$t> {
            type Element = $t;
            fn start(&self) -> SafeNodeRef {
                (self.start).expr().node()
            }
            fn end(&self) -> SafeNodeRef {
                (self.end).expr().node()
            }
            fn end_inclusive(&self) -> bool {
                false
            }
        }
        impl ForLoopRange for std::ops::Range<Expr<$t>> {
            type Element = $t;
            fn start(&self) -> SafeNodeRef {
                self.start.node()
            }
            fn end(&self) -> SafeNodeRef {
                self.end.node()
            }
            fn end_inclusive(&self) -> bool {
                false
            }
        }
    };
}
impl_range!(i32);
impl_range!(i64);
impl_range!(u32);
impl_range!(u64);

pub fn loop_(body: impl Fn()) {
    while_!(true.expr(), {
        body();
    });
}

pub fn for_unrolled<I: IntoIterator>(iter: I, body: impl Fn(I::Item)) {
    for i in iter {
        body(i);
    }
}

pub fn for_range<R: ForLoopRange>(r: R, body: impl Fn(Expr<R::Element>)) {
    let start = r.start().get();
    let end = r.end().get();
    let inc = |v: NodeRef| {
        __current_scope(|b| {
            let one = b.const_(Const::One(v.type_().clone()));
            b.call(Func::Add, &[v, one], v.type_().clone())
        })
    };
    let i = __current_scope(|b| b.local(start));
    generic_loop(
        || {
            Expr::<bool>::from_node(
                __current_scope(|b| {
                    let i = b.call(Func::Load, &[i], i.type_().clone());
                    b.call(
                        if r.end_inclusive() {
                            Func::Le
                        } else {
                            Func::Lt
                        },
                        &[i, end],
                        <bool as TypeOf>::type_(),
                    )
                })
                .into(),
            )
        },
        move || {
            let i = __current_scope(|b| b.call(Func::Load, &[i], i.type_().clone()));
            body(Expr::<R::Element>::from_node(i.into()));
        },
        || {
            let i_old = __current_scope(|b| b.call(Func::Load, &[i], i.type_().clone()));
            let i_new = inc(i_old);
            __current_scope(|b| b.update(i, i_new));
        },
    )
}

pub struct SwitchBuilder<R: Aggregate> {
    cases: Vec<(i32, Pooled<BasicBlock>, Vec<NodeRef>)>,
    default: Option<(Pooled<BasicBlock>, Vec<NodeRef>)>,
    value: NodeRef,
    _marker: PhantomData<R>,
    depth: usize,
}

pub fn switch<R: Aggregate>(node: Expr<i32>) -> SwitchBuilder<R> {
    SwitchBuilder::new(node)
}

impl<R: Aggregate> SwitchBuilder<R> {
    pub fn new(node: Expr<i32>) -> Self {
        SwitchBuilder {
            cases: vec![],
            default: None,
            value: node.node().get(),
            _marker: PhantomData,
            depth: with_recorder(|r| r.scopes.len()),
        }
    }
    pub fn case(mut self, value: i32, then: impl Fn() -> R) -> Self {
        with_recorder(|r| {
            let pools = r.pools.clone();
            let s = &mut r.scopes;
            assert_eq!(s.len(), self.depth);
            s.push(IrBuilder::new(pools));
        });
        let then = then().to_vec_nodes().into_iter().map(|x| x.get()).collect();
        let block = __pop_scope();
        self.cases.push((value, block, then));
        self
    }
    pub fn default(mut self, then: impl Fn() -> R) -> Self {
        with_recorder(|r| {
            let pools = r.pools.clone();
            let s = &mut r.scopes;
            assert_eq!(s.len(), self.depth);
            s.push(IrBuilder::new(pools));
        });
        let then = then().to_vec_nodes().into_iter().map(|x| x.get()).collect();
        let block = __pop_scope();
        self.default = Some((block, then));
        self
    }
    pub fn finish(self) -> R {
        with_recorder(|r| {
            let s = &mut r.scopes;
            assert_eq!(s.len(), self.depth);
        });
        let cases = self
            .cases
            .iter()
            .map(|(v, b, _)| SwitchCase {
                value: *v,
                block: *b,
            })
            .collect::<Vec<_>>();
        let case_phis = self
            .cases
            .iter()
            .map(|(_, _, nodes)| nodes.clone())
            .collect::<Vec<_>>();
        let phi_count = case_phis[0].len();
        let mut default_nodes = vec![];
        let default_block = if self.default.is_none() {
            with_recorder(|r| {
                let pools = r.pools.clone();
                let s = &mut r.scopes;
                assert_eq!(s.len(), self.depth);
                s.push(IrBuilder::new(pools));
            });
            for i in 0..phi_count {
                // let msg = CString::new("unreachable code in switch statement!").unwrap();
                // let default_node = __current_scope(|b| {
                //     b.call(
                //         Func::Unreachable(CBoxedSlice::from(msg)),
                //         &[],
                //         case_phis[0][i].type_().clone(),
                //     )
                // });
                let default_node = __unreachable_typed(
                    case_phis[0][i].type_().clone(),
                    file!(),
                    line!(),
                    column!(),
                );
                default_nodes.push(default_node);
            }
            __pop_scope()
        } else {
            default_nodes = self.default.as_ref().unwrap().1.clone();
            self.default.as_ref().unwrap().0
        };
        __current_scope(|b| {
            b.switch(self.value, &cases, default_block);
        });
        let mut phis = vec![];
        for i in 0..phi_count {
            let mut incomings = vec![];
            for (j, nodes) in case_phis.iter().enumerate() {
                incomings.push(PhiIncoming {
                    value: nodes[i],
                    block: self.cases[j].1,
                });
            }
            incomings.push(PhiIncoming {
                value: default_nodes[i],
                block: default_block,
            });
            let phi = __current_scope(|b| b.phi(&incomings, case_phis[0][i].type_().clone()));
            phis.push(phi);
        }
        let phis = phis.into_iter().map(|x| x.into()).collect::<Vec<_>>();
        R::from_vec_nodes(phis)
    }
}
