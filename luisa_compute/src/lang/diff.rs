use std::cell::RefCell;

use crate::internal_prelude::*;

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
pub fn requires_grad(var: impl ExprProxy) {
    AD_CONTEXT.with(|c| {
        let c = c.borrow();
        assert!(c.started, "autodiff section is not started");
        assert!(
            !c.is_forward_mode,
            "requires_grad() is called in forward mode"
        );
        assert!(!c.backward_called, "backward is already called");
    });
    __current_scope(|b| {
        b.call(Func::RequiresGradient, &[var.node()], Type::void());
    });
}

pub fn backward<T: ExprProxy>(out: T) {
    backward_with_grad(
        out,
        FromNode::from_node(__current_scope(|b| {
            let one = new_node(
                b.pools(),
                Node::new(
                    CArc::new(Instruction::Const(Const::One(<T::Value>::type_()))),
                    <T::Value>::type_(),
                ),
            );
            b.append(one);
            one
        })),
    );
}

pub fn backward_with_grad<T: ExprProxy>(out: T, grad: T) {
    AD_CONTEXT.with(|c| {
        let mut c = c.borrow_mut();
        assert!(c.started, "autodiff section is not started");
        assert!(!c.is_forward_mode, "backward() is called in forward mode");
        assert!(!c.backward_called, "backward is already called");
        c.backward_called = true;
    });
    let out = out.node();
    let grad = grad.node();
    __current_scope(|b| {
        b.call(Func::GradientMarker, &[out, grad], Type::void());
        b.call(Func::Backward, &[], Type::void());
    });
}

/// Gradient of a value in *Reverse mode* AD
pub fn gradient<T: ExprProxy>(var: T) -> T {
    AD_CONTEXT.with(|c| {
        let c = c.borrow();
        assert!(c.started, "autodiff section is not started");
        assert!(!c.is_forward_mode, "gradient() is called in forward mode");
        assert!(c.backward_called, "backward is not called");
    });
    T::from_node(__current_scope(|b| {
        b.call(Func::Gradient, &[var.node()], var.node().type_().clone())
    }))
}
/// Gradient of a value in *Reverse mode* AD
pub fn grad<T: ExprProxy>(var: T) -> T {
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
//         let node = new_node(Node::new(CArc::new(Instruction::AdDetach(fwd)), Type::void()));
//         b.append(node);
//     });
//     let nodes = ret.to_vec_nodes();
//     let nodes: Vec<_> = nodes
//         .iter()
//         .map(|n| __current_scope(|b| b.call(Func::Detach, &[*n], n.type_())))
//         .collect();
//     R::from_vec_nodes(nodes)
// }
pub fn detach<T: NodeLike>(v: T) -> T {
    let v = v.node();
    let node = __current_scope(|b| b.call(Func::Detach, &[v], v.type_().clone()));
    T::from_node(node)
}

/// Start a *Forward mode* AD section that propagates N gradients w.r.t to input variable
pub fn forward_autodiff(n_grads: usize, body: impl Fn()) {
    AD_CONTEXT.with(|c| {
        let mut c = c.borrow_mut();
        assert!(!c.started, "autodiff section already started");
        *c = AdContext::new_fwd(n_grads);
        c.started = true;
    });
    RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let pools = r.pools.clone().unwrap();
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
pub fn propagate_gradient<T: ExprProxy>(v: T, grads: &[T]) {
    AD_CONTEXT.with(|c| {
        let c = c.borrow();
        assert_eq!(grads.len(), c.n_forward_grads);
        assert!(c.started, "autodiff section is not started");
        assert!(
            c.is_forward_mode,
            "propagate_gradient() is called in backward mode"
        );
    });
    __current_scope(|b| {
        let mut nodes = vec![v.node()];
        nodes.extend(grads.iter().map(|g| g.node()));
        b.call(Func::PropagateGrad, &nodes, Type::void());
    });
}

pub fn output_gradients<T: ExprProxy>(v: T) -> Vec<T> {
    let n = AD_CONTEXT.with(|c| {
        let c = c.borrow();
        assert!(c.started, "autodiff section is not started");
        assert!(
            c.is_forward_mode,
            "output_gradients() is called in backward mode"
        );
        c.n_forward_grads
    });
    __current_scope(|b| {
        let mut grads = vec![];
        for i in 0..n {
            let idx = b.const_(Const::Int32(i as i32));
            grads.push(T::from_node(b.call(
                Func::OutputGrad,
                &[v.node(), idx],
                v.node().type_().clone(),
            )));
        }
        grads
    })
}

pub fn autodiff(body: impl Fn()) {
    AD_CONTEXT.with(|c| {
        let mut c = c.borrow_mut();
        assert!(!c.started, "autodiff section is already started");
        *c = AdContext::new_rev();
        c.started = true;
    });
    RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let pools = r.pools.clone().unwrap();
        let s = &mut r.scopes;
        s.push(IrBuilder::new(pools));
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
