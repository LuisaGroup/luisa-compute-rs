use std::cell::RefCell;

use crate::internal_prelude::*;

use super::with_recorder;

struct AdContext {
    started: bool,
    backward_called: bool,
    is_forward_mode: bool,
    n_forward_grads: usize,
    // forward: Option<Pooled<BasicBlock>>,
}

impl AdContext {
    fn new_rev() -> Self {
        Self {
            started: false,
            backward_called: false,
            is_forward_mode: false,
            n_forward_grads: 0,
        }
    }
    fn new_fwd(n: usize) -> Self {
        Self {
            started: false,
            backward_called: false,
            is_forward_mode: true,
            n_forward_grads: n,
        }
    }
    fn reset(&mut self) {
        self.started = false;
    }
}
thread_local! {
    static AD_CONTEXT:RefCell<AdContext> = RefCell::new(AdContext::new_rev());
}
pub fn requires_grad<V: Value>(var: Expr<V>) {
    AD_CONTEXT.with(|c| {
        let c = c.borrow();
        assert!(c.started, "autodiff section is not started");
        assert!(
            !c.is_forward_mode,
            "requires_grad() is called in forward mode"
        );
        assert!(!c.backward_called, "backward is already called");
    });
    let var = var.node().get();
    __current_scope(|b| {
        b.call(Func::RequiresGradient, &[var], Type::void());
    });
}

pub fn backward<V: Value>(out: Expr<V>) {
    backward_with_grad(
        out,
        FromNode::from_node(__current_scope(|b| {
            let one = new_node(
                b.pools(),
                Node::new(
                    CArc::new(Instruction::Const(Const::One(V::type_()))),
                    V::type_(),
                ),
            );
            b.append(one);
            one.into()
        })),
    );
}

pub fn backward_with_grad<V: Value>(out: Expr<V>, grad: Expr<V>) {
    AD_CONTEXT.with(|c| {
        let mut c = c.borrow_mut();
        assert!(c.started, "autodiff section is not started");
        assert!(!c.is_forward_mode, "backward() is called in forward mode");
        assert!(!c.backward_called, "backward is already called");
        c.backward_called = true;
    });
    let out = out.node().get();
    let grad = grad.node().get();
    __current_scope(|b| {
        b.call(Func::GradientMarker, &[out, grad], Type::void());
        b.call(Func::Backward, &[], Type::void());
    });
}

/// Gradient of a value in *Reverse mode* AD
pub fn gradient<V: Value>(var: Expr<V>) -> Expr<V> {
    AD_CONTEXT.with(|c| {
        let c = c.borrow();
        assert!(c.started, "autodiff section is not started");
        assert!(!c.is_forward_mode, "gradient() is called in forward mode");
        assert!(c.backward_called, "backward is not called");
    });
    let var = var.node().get();
    Expr::<V>::from_node(
        __current_scope(|b| b.call(Func::Gradient, &[var], var.type_().clone()))
            .into(),
    )
}
/// Gradient of a value in *Reverse mode* AD
pub fn grad<V: Value>(var: Expr<V>) -> Expr<V> {
    gradient(var)
}

// pub fn detach<R: Aggregate>(body: impl FnOnce() -> R) -> R {
//     RECORDER.with(|r| {
//         let mut r = r.borrow_mut();
//         let s = &mut r.scopes;
//         s.push(IrBuilder::new());
//     });
//     let ret = body();
//     let fwd = pop_scope();
//     __current_scope(|b| {
//         let node = new_node(Node::new(CArc::new(Instruction::AdDetach(fwd)),
// Type::void()));         b.append(node);
//     });
//     let nodes = ret.to_vec_nodes();
//     let nodes: Vec<_> = nodes
//         .iter()
//         .map(|n| __current_scope(|b| b.call(Func::Detach, &[*n], n.type_())))
//         .collect();
//     R::from_vec_nodes(nodes)
// }
pub fn detach<T: NodeLike>(v: T) -> T {
    let v = v.node().get();
    let node = __current_scope(|b| b.call(Func::Detach, &[v], v.type_().clone()));
    T::from_node(node.into())
}

/// Start a *Forward mode* AD section that propagates N gradients w.r.t to input
/// variable
pub fn forward_autodiff(n_grads: usize, body: impl Fn()) {
    AD_CONTEXT.with(|c| {
        let mut c = c.borrow_mut();
        assert!(!c.started, "autodiff section already started");
        *c = AdContext::new_fwd(n_grads);
        c.started = true;
    });
    with_recorder(|r| {
        let pools = r.pools.clone();
        let s = &mut r.scopes;
        s.push(IrBuilder::new(pools));
    });
    body();
    let n_grads = AD_CONTEXT.with(|c| {
        let mut c = c.borrow_mut();
        let n_grads = c.n_forward_grads;
        c.reset();
        n_grads
    });
    let body = __pop_scope();
    __current_scope(|b| {
        b.fwd_ad_scope(body, n_grads);
    });
}

/// Propagate N gradients w.r.t to input variable using *Forward mode* AD
pub fn propagate_gradient<V: Value>(v: Expr<V>, grads: &[Expr<V>]) {
    AD_CONTEXT.with(|c| {
        let c = c.borrow();
        assert_eq!(grads.len(), c.n_forward_grads);
        assert!(c.started, "autodiff section is not started");
        assert!(
            c.is_forward_mode,
            "propagate_gradient() is called in backward mode"
        );
    });
    let mut nodes = vec![v.node().get()];
    nodes.extend(grads.iter().map(|g| g.node().get()));
    __current_scope(|b| {
        b.call(Func::PropagateGrad, &nodes, Type::void());
    });
}

pub fn output_gradients<V: Value>(v: Expr<V>) -> Vec<Expr<V>> {
    let n = AD_CONTEXT.with(|c| {
        let c = c.borrow();
        assert!(c.started, "autodiff section is not started");
        assert!(
            c.is_forward_mode,
            "output_gradients() is called in backward mode"
        );
        c.n_forward_grads
    });
    let mut grads = vec![];
    let v = v.node().get();
    for i in 0..n {
        grads.push(Expr::<V>::from_node(__current_scope(|b| {
            let idx = b.const_(Const::Int32(i as i32));
            b.call(Func::OutputGrad, &[v, idx], v.type_().clone())
        }).into()));
    }
    grads
}

pub fn autodiff(body: impl Fn()) {
    AD_CONTEXT.with(|c| {
        let mut c = c.borrow_mut();
        assert!(!c.started, "autodiff section is already started");
        *c = AdContext::new_rev();
        c.started = true;
    });
    with_recorder(|r| {
        let s = &mut r.scopes;
        s.push(IrBuilder::new(r.pools.clone()));
    });
    body();
    AD_CONTEXT.with(|c| {
        let mut c = c.borrow_mut();
        assert!(c.started, "autodiff section is not started");
        assert!(c.backward_called, "backward is not called");
        c.reset();
    });
    let body = __pop_scope();
    __current_scope(|b| {
        b.ad_scope(body);
    });
}
