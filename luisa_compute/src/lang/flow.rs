use super::*;

/**
 * If you want rustfmt to format your code, use if_!(cond, { .. }, { .. }) or if_!(cond, { .. }, else, {...})
 * instead of if_!(cond, { .. }, else {...}).
 *
 */
#[macro_export]
macro_rules! if_ {
    ($cond:expr, $then:block, else $else_:block) => {
        $crate::lang::if_then_else($cond, || $then, || $else_)
    };
    ($cond:expr, $then:block, else, $else_:block) => {
        $crate::lang::if_then_else($cond, || $then, || $else_)
    };
    ($cond:expr, $then:block, $else_:block) => {
        $crate::lang::if_then_else($cond, || $then, || $else_)
    };
    ($cond:expr, $then:block) => {
        $crate::lang::if_then_else($cond, || $then, || {})
    };
}
#[macro_export]
macro_rules! while_ {
    ($cond:expr,$body:block) => {
        $crate::lang::generic_loop(|| $cond, || $body, || {})
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

// #[inline]
// pub fn return_v<T: FromNode>(v: T) {
//     __current_scope(|b| {
//         b.return_(Some(v.node()));
//     });
// }
#[inline]
pub fn return_() {
    __current_scope(|b| {
        b.return_(INVALID_REF);
    });
}

pub fn if_then_else<R: Aggregate>(
    cond: impl Mask,
    then: impl FnOnce() -> R,
    else_: impl FnOnce() -> R,
) -> R {
    let cond = cond.node();
    RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let pools = r.pools.clone().unwrap();
        let s = &mut r.scopes;
        s.push(IrBuilder::new(pools));
    });
    let then = then();
    let then_block = RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let pools = r.pools.clone().unwrap();
        let s = &mut r.scopes;
        let then_block = s.pop().unwrap().finish();
        s.push(IrBuilder::new(pools));
        then_block
    });
    let else_ = else_();
    let else_block = RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let s = &mut r.scopes;
        s.pop().unwrap().finish()
    });
    let then_nodes = then.to_vec_nodes();
    let else_nodes = else_.to_vec_nodes();
    __current_scope(|b| {
        b.if_(cond, then_block, else_block);
    });
    assert_eq!(then_nodes.len(), else_nodes.len());
    let phis = __current_scope(|b| {
        then_nodes
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
                let phi = b.phi(&incomings, then.type_().clone());
                phi
            })
            .collect::<Vec<_>>()
    });
    R::from_vec_nodes(phis)
}

pub fn generic_loop(
    mut cond: impl FnMut() -> Expr<bool>,
    mut body: impl FnMut(),
    mut update: impl FnMut(),
) {
    RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let pools = r.pools.clone().unwrap();
        let s = &mut r.scopes;
        s.push(IrBuilder::new(pools));
    });
    let cond_v = cond().node();
    let prepare = RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let pools = r.pools.clone().unwrap();
        let s = &mut r.scopes;
        let prepare = s.pop().unwrap().finish();
        s.push(IrBuilder::new(pools));
        prepare
    });
    body();
    let body = RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let pools = r.pools.clone().unwrap();
        let s = &mut r.scopes;
        let body = s.pop().unwrap().finish();
        s.push(IrBuilder::new(pools));
        body
    });
    update();
    let update = RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let s = &mut r.scopes;
        s.pop().unwrap().finish()
    });
    __current_scope(|b| {
        b.generic_loop(prepare, cond_v, body, update);
    });
}

pub trait ForLoopRange {
    type Element: Value;
    fn start(&self) -> NodeRef;
    fn end(&self) -> NodeRef;
    fn end_inclusive(&self) -> bool;
}
macro_rules! impl_range {
    ($t:ty) => {
        impl ForLoopRange for std::ops::RangeInclusive<$t> {
            type Element = $t;
            fn start(&self) -> NodeRef {
                (*self.start()).expr().node()
            }
            fn end(&self) -> NodeRef {
                (*self.end()).expr().node()
            }
            fn end_inclusive(&self) -> bool {
                true
            }
        }
        impl ForLoopRange for std::ops::RangeInclusive<Expr<$t>> {
            type Element = $t;
            fn start(&self) -> NodeRef {
                self.start().node()
            }
            fn end(&self) -> NodeRef {
                self.end().node()
            }
            fn end_inclusive(&self) -> bool {
                true
            }
        }
        impl ForLoopRange for std::ops::Range<$t> {
            type Element = $t;
            fn start(&self) -> NodeRef {
                (self.start).expr().node()
            }
            fn end(&self) -> NodeRef {
                (self.end).expr().node()
            }
            fn end_inclusive(&self) -> bool {
                false
            }
        }
        impl ForLoopRange for std::ops::Range<Expr<$t>> {
            type Element = $t;
            fn start(&self) -> NodeRef {
                self.start.node()
            }
            fn end(&self) -> NodeRef {
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

pub fn for_range<R: ForLoopRange>(r: R, body: impl Fn(Expr<R::Element>)) {
    let start = r.start();
    let end = r.end();
    let inc = |v: NodeRef| {
        __current_scope(|b| {
            let one = b.const_(Const::One(v.type_().clone()));
            b.call(Func::Add, &[v, one], v.type_().clone())
        })
    };
    let i = __current_scope(|b| b.local(start));
    generic_loop(
        || {
            __current_scope(|b| {
                let i = b.call(Func::Load, &[i], i.type_().clone());
                Expr::<bool>::from_node(b.call(
                    if r.end_inclusive() {
                        Func::Le
                    } else {
                        Func::Lt
                    },
                    &[i, end],
                    <bool as TypeOf>::type_(),
                ))
            })
        },
        move || {
            let i = __current_scope(|b| b.call(Func::Load, &[i], i.type_().clone()));
            body(Expr::<R::Element>::from_node(i));
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
            value: node.node(),
            _marker: PhantomData,
            depth: RECORDER.with(|r| r.borrow().scopes.len()),
        }
    }
    pub fn case(mut self, value: i32, then: impl Fn() -> R) -> Self {
        RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            let pools = r.pools.clone().unwrap();
            let s = &mut r.scopes;
            assert_eq!(s.len(), self.depth);
            s.push(IrBuilder::new(pools));
        });
        let then = then();
        let block = __pop_scope();
        self.cases.push((value, block, then.to_vec_nodes()));
        self
    }
    pub fn default(mut self, then: impl Fn() -> R) -> Self {
        RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            let pools = r.pools.clone().unwrap();
            let s = &mut r.scopes;
            assert_eq!(s.len(), self.depth);
            s.push(IrBuilder::new(pools));
        });
        let then = then();
        let block = __pop_scope();
        self.default = Some((block, then.to_vec_nodes()));
        self
    }
    pub fn finish(self) -> R {
        RECORDER.with(|r| {
            let mut r = r.borrow_mut();
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
            RECORDER.with(|r| {
                let mut r = r.borrow_mut();
                let pools = r.pools.clone().unwrap();
                let s = &mut r.scopes;
                assert_eq!(s.len(), self.depth);
                s.push(IrBuilder::new(pools));
            });
            for i in 0..phi_count {
                let msg = CString::new("unreachable code in switch statement!").unwrap();
                let default_node = __current_scope(|b| {
                    b.call(
                        Func::Unreachable(CBoxedSlice::from(msg)),
                        &[],
                        case_phis[0][i].type_().clone(),
                    )
                });
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
        R::from_vec_nodes(phis)
    }
}
