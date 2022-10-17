use std::{any::Any, ops::Deref};

use crate::*;
pub use ir::ir::NodeRef;
use ir::{
    ir::{BasicBlock, Const, IrBuilder},
    CBoxedSlice,
};
use luisa_compute_ir as ir;
use std::cell::RefCell;
pub mod traits;

pub trait Value: Copy + ir::TypeOf {
    type Proxy: VarProxy<Self>;
    type SoN: StructOfNodes;

    fn destruct(v: Var<Self>) -> Self::SoN;
    fn construct(s: Self::SoN) -> Var<Self>;
}

/* Struct of Nodes
*  Similar to how enoki, Dr.Jit handles structs
*  Makes every field of the struct a node
*  Most concise way to manipulate structs
*/
pub trait StructOfNodes: Sized {
    fn to_vec_nodes(&self) -> Vec<NodeRef> {
        let mut nodes = vec![];
        Self::to_nodes(&self, &mut nodes);
        nodes
    }
    fn from_vec_nodes(nodes: Vec<NodeRef>) -> Self {
        let mut iter = nodes.into_iter();
        let ret = Self::from_nodes(&mut iter);
        assert!(iter.next().is_none());
        ret
    }
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>);
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self;
}
pub trait Selectable {
    fn select(mask: Mask, lhs: Self, rhs: Self) -> Self;
}
pub trait VarProxy<T>: Copy + From<T> {
    fn from_node(node: NodeRef) -> Self;
    fn node(&self) -> NodeRef;
}
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Var<T: Value> {
    pub(crate) proxy: T::Proxy,
}
pub type Mask = Var<bool>;

impl<T: Value> Var<T> {
    pub fn set(&self, value: Self) {
        unimplemented!()
    }
    pub fn expand(&self) -> T::SoN {
        unimplemented!()
    }
    pub fn collect(value: T::SoN) -> Self {
        unimplemented!()
    }
    pub fn from_node(node: NodeRef) -> Self {
        Self {
            proxy: T::Proxy::from_node(node),
        }
    }
}
impl<T: Value> From<T> for Var<T> {
    fn from(t: T) -> Self {
        Self {
            proxy: T::Proxy::from(t),
        }
    }
}

impl<T: Value> Deref for Var<T> {
    type Target = T::Proxy;
    fn deref(&self) -> &Self::Target {
        &self.proxy
    }
}
#[derive(Clone, Copy, Debug)]
pub struct PrimProxy<T> {
    pub(crate) node: NodeRef,
    pub(crate) _phantom: std::marker::PhantomData<T>,
}

impl<T> StructOfNodes for PrimProxy<T> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        nodes.push(self.node);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        Self {
            node: iter.next().unwrap(),
            _phantom: std::marker::PhantomData,
        }
    }
}
macro_rules! impl_prim {
    ($t:ty) => {
        impl From<$t> for PrimProxy<$t> {
            fn from(v: $t) -> Self {
                const_(v).proxy
            }
        }
        impl VarProxy<$t> for PrimProxy<$t> {
            fn from_node(node: NodeRef) -> Self {
                Self {
                    node,
                    _phantom: std::marker::PhantomData,
                }
            }
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        impl Value for $t {
            type Proxy = PrimProxy<$t>;
            type SoN = PrimProxy<$t>;

            fn destruct(v: Var<Self>) -> Self::SoN {
                v.proxy
            }

            fn construct(s: Self::SoN) -> Var<Self> {
                Var { proxy: s }
            }
        }
    };
}

impl_prim!(bool);
impl_prim!(u32);
impl_prim!(u64);
impl_prim!(i32);
impl_prim!(i64);
impl_prim!(f32);
impl_prim!(f64);

pub type Bool = Var<bool>;
pub type Float = Var<f32>;
pub type Double = Var<f64>;
pub type Int = Var<i32>;
pub type Int32 = Var<i32>;
pub type Int64 = Var<i64>;
pub type Uint = Var<u32>;
pub type Uint32 = Var<u32>;
pub type Uint64 = Var<u64>;

pub(crate) struct Recorder {
    scopes: Vec<IrBuilder>,
}

thread_local! {
    pub(crate) static RECORDER: RefCell<Recorder> = RefCell::new(Recorder {
        scopes: vec![],
    });
}

pub(crate) fn current_scope<F: FnOnce(&mut IrBuilder) -> R, R>(f: F) -> R {
    RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let s = &mut r.scopes;
        if s.is_empty() {
            s.push(IrBuilder::new());
        }
        f(s.last_mut().unwrap())
    })
}
pub(crate) fn pop_scope() -> &'static BasicBlock {
    RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let s = &mut r.scopes;
        s.pop().unwrap().finish()
    })
}
pub fn const_<T: Value + Copy + 'static>(value: T) -> Var<T> {
    let node = current_scope(|s| -> NodeRef {
        let any = &value as &dyn Any;
        if let Some(value) = any.downcast_ref::<bool>() {
            s.const_(Const::Bool(*value))
        } else if let Some(value) = any.downcast_ref::<i32>() {
            s.const_(Const::Int32(*value))
        } else if let Some(value) = any.downcast_ref::<u32>() {
            s.const_(Const::Uint32(*value))
        } else if let Some(value) = any.downcast_ref::<i64>() {
            s.const_(Const::Int64(*value))
        } else if let Some(value) = any.downcast_ref::<u64>() {
            s.const_(Const::Uint64(*value))
        } else if let Some(value) = any.downcast_ref::<f32>() {
            s.const_(Const::Float32(*value))
        } else if let Some(value) = any.downcast_ref::<f64>() {
            s.const_(Const::Float64(*value))
        } else {
            let mut buf = vec![0u8; std::mem::size_of::<T>()];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &value as *const T as *const u8,
                    buf.as_mut_ptr(),
                    buf.len(),
                );
            }
            s.const_(Const::Generic(CBoxedSlice::new(buf), T::type_()))
        }
    });
    Var::from_node(node)
}
