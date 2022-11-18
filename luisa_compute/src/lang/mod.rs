use std::{any::Any, collections::HashMap, ops::Deref, sync::Arc};

use crate::resource::{BindlessArrayHandle, Buffer, BufferHandle, TextureHandle};
pub use ir::ir::NodeRef;
use ir::ir::{new_node, BasicBlock, Const, Func, Instruction, IrBuilder, Node};
use luisa_compute_ir as ir;

pub use luisa_compute_ir::{
    context::register_type,
    ffi::CBoxedSlice,
    ir::{StructType, Type},
    Gc, TypeOf,
};
use std::cell::RefCell;
pub mod math;
pub mod math_impl;
pub mod traits;
pub mod traits_impl;

pub trait Value: Copy + ir::TypeOf {
    type Proxy: Proxy<Self>;
}

pub trait Aggregate: Sized {
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
fn _store<T: Aggregate>(var: &T, value: &T) {
    let value_nodes = value.to_vec_nodes();
    let self_nodes = var.to_vec_nodes();
    assert_eq!(value_nodes.len(), self_nodes.len());
    current_scope(|b| {
        for (value_node, self_node) in value_nodes.into_iter().zip(self_nodes.into_iter()) {
            b.store(self_node, value_node);
        }
    })
}
pub trait Selectable {
    fn select(mask: Mask, lhs: Self, rhs: Self) -> Self;
}
pub trait Proxy<T>: Copy + Aggregate {
    fn from_node(node: NodeRef) -> Self;
    fn node(&self) -> NodeRef;
}
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Expr<T: Value> {
    pub(crate) proxy: T::Proxy,
}
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Var<T: Value> {
    pub(crate) proxy: T::Proxy,
}

impl<T: Value> Var<T> {
    pub fn store(&self, value: &Expr<T>) {
        _store(&self.proxy, &value.proxy);
    }
    pub fn load(&self) -> Expr<T> {
        Expr {
            proxy: current_scope(|b| {
                let nodes = self.proxy.to_vec_nodes();
                let mut ret = vec![];
                for node in nodes {
                    ret.push(b.call(Func::Load, &[node], node.type_()));
                }
                T::Proxy::from_nodes(&mut ret.into_iter())
            }),
        }
    }
}
pub type Mask = Expr<bool>;
impl<T: Value + 'static> From<T> for Expr<T> {
    fn from(value: T) -> Self {
        const_(value)
    }
}
impl<T: Value> Expr<T> {
    pub(crate) fn expand(&self) -> Vec<NodeRef> {
        self.proxy.to_vec_nodes()
    }
    pub(crate) fn collect(nodes: &[NodeRef]) -> Self {
        let proxy = T::Proxy::from_nodes(&mut nodes.iter().cloned());
        Self { proxy }
    }
    pub fn from_proxy(proxy: T::Proxy) -> Self {
        Self { proxy }
    }
    pub(crate) fn node(&self) -> NodeRef {
        self.proxy.node()
    }
    pub(crate) fn from_node(node: NodeRef) -> Self {
        Self {
            proxy: T::Proxy::from_node(node),
        }
    }
}

impl<T: Value> Deref for Expr<T> {
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

impl<T> Aggregate for PrimProxy<T> {
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
        impl Proxy<$t> for PrimProxy<$t> {
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

pub(crate) struct Recorder {
    scopes: Vec<IrBuilder>,
    buffer_to_node: HashMap<u64, NodeRef>,
}

thread_local! {
    pub(crate) static RECORDER: RefCell<Recorder> = RefCell::new(Recorder {
        scopes: vec![],
        buffer_to_node: HashMap::new(),
    });
}

// Don't call this function directly unless you know what you are doing
pub fn current_scope<F: FnOnce(&mut IrBuilder) -> R, R>(f: F) -> R {
    RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let s = &mut r.scopes;
        if s.is_empty() {
            s.push(IrBuilder::new());
        }
        f(s.last_mut().unwrap())
    })
}
// Don't call this function directly unless you know what you are doing
pub fn pop_scope() -> Gc<BasicBlock> {
    RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let s = &mut r.scopes;
        s.pop().unwrap().finish()
    })
}
pub fn __extract<T: Value>(node: NodeRef, index: usize) -> NodeRef {
    current_scope(|b| {
        let i = b.const_(Const::Int32(index as i32));
        let node = b.call(Func::ExtractElement, &[node, i], <T as TypeOf>::type_());
        node
    })
}
pub fn __compose<T: Value>(nodes: &[NodeRef]) -> NodeRef {
    current_scope(|b| b.call(Func::Struct, nodes, <T as TypeOf>::type_()))
}
pub fn const_<T: Value + Copy + 'static>(value: T) -> Expr<T> {
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
    Expr::from_node(node)
}

pub struct BufferVar<T: Value> {
    marker: std::marker::PhantomData<T>,
    #[allow(dead_code)]
    handle: Arc<BufferHandle>,
    node: NodeRef,
}

impl<T: Value> Drop for BufferVar<T> {
    fn drop(&mut self) {
        todo!()
    }
}
pub struct BindlessArrayVar {
    #[allow(dead_code)]
    handle: Arc<BindlessArrayHandle>,
    node: NodeRef,
}
impl BindlessArrayVar {
    pub fn buffer_read<T: Value, BI: Into<Expr<u32>>, EI: Into<Expr<u32>>>(
        &self,
        buffer_index: BI,
        element_index: EI,
    ) -> Expr<T> {
        Expr::from_node(current_scope(|b| {
            b.call(
                Func::BindlessBufferRead,
                &[
                    self.node,
                    buffer_index.into().node(),
                    element_index.into().node(),
                ],
                T::type_(),
            )
        }))
    }
    pub fn buffer_length<I: Into<Expr<u32>>>(&self, buffer_index: I) -> Expr<u32> {
        Expr::<u32>::from_node(current_scope(|b| {
            b.call(
                Func::BindlessBufferSize,
                &[self.node, buffer_index.into().node()],
                u32::type_(),
            )
        }))
    }
}
impl<T: Value> BufferVar<T> {
    pub fn new(buffer: &Buffer<T>) -> Self {
        let node = RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            let handle: u64 = buffer.handle().0;
            if let Some(node) = r.buffer_to_node.get(&handle) {
                *node
            } else {
                let node = new_node(Node::new(Gc::new(Instruction::Buffer), T::type_()));
                r.buffer_to_node.insert(handle, node);
                node
            }
        });
        Self {
            node,
            marker: std::marker::PhantomData,
            handle: buffer.handle.clone(),
        }
    }
    pub fn len(&self) -> Expr<u32> {
        Expr::from_node(
            current_scope(|b| b.call(Func::BufferSize, &[self.node], u32::type_())).into(),
        )
    }
    pub fn read<I: Into<Expr<u32>>>(&self, i: I) -> Expr<T> {
        current_scope(|b| {
            Expr::from_node(b.call(Func::BufferRead, &[self.node, i.into().node()], T::type_()))
        })
    }
    pub fn write<I: Into<Expr<u32>>, V: Into<Expr<T>>>(&self, i: I, v: V) {
        current_scope(|b| {
            b.call(
                Func::BufferWrite,
                &[self.node, i.into().node(), v.into().node()],
                Type::void(),
            )
        });
    }
    pub fn atomic_exchange<I: Into<Expr<u32>>, V: Into<Expr<T>>>(&self, i: I, v: V) -> Expr<T> {
        todo!()
    }
}

pub struct ImageVar<T: Value> {
    #[allow(dead_code)]
    handle: Arc<TextureHandle>,
    _marker: std::marker::PhantomData<T>,
}

pub struct VolumeVar<T: Value> {
    #[allow(dead_code)]
    handle: Arc<TextureHandle>,
    _marker: std::marker::PhantomData<T>,
}
pub type Tex2DVar<T> = ImageVar<T>;
pub type Tex3DVar<T> = VolumeVar<T>;

#[macro_export]
macro_rules! struct_ {
    ($name:ident $fields:tt) => {
        {
            type P = <$name as Value>::Proxy;
            Expr::from_proxy(P $fields)
        }
    };
}