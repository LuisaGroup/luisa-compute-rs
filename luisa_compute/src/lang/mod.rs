use std::{any::Any, collections::HashMap, ops::Deref, sync::Arc};

use crate::{
    backend,
    prelude::{ArgEncoder, Kernel, KernelArg, RawKernel},
    resource::{
        BindlessArray, BindlessArrayHandle, Buffer, BufferHandle, Tex2D, Tex3D, Texel,
        TextureHandle,
    },
};
pub use ir::ir::NodeRef;
use ir::ir::{
    new_node, BasicBlock, Binding, BufferBinding, Capture, Const, Func, Instruction, IrBuilder,
    KernelModule, Module, ModuleKind, Node, PhiIncoming,
};
use luisa_compute_ir as ir;

pub use luisa_compute_ir::{
    context::register_type,
    ffi::CBoxedSlice,
    ir::{StructType, Type},
    Gc, TypeOf,
};
use math::{BVec2Expr, BVec3Expr, BVec4Expr, UVec3};
use std::cell::RefCell;

// use self::math::UVec3;
pub mod math;
pub mod math_impl;
pub mod traits;

pub trait Value: Copy + ir::TypeOf {
    type Expr: ExprProxy<Self>;
    type Var: VarProxy<Self>;
    fn fields() -> Vec<String>;
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
pub trait Selectable: Aggregate {}
pub trait FromNode {
    fn from_node(node: NodeRef) -> Self;
    fn node(&self) -> NodeRef;
}
fn _store<T1: Aggregate, T2: Aggregate>(var: &T1, value: &T2) {
    let value_nodes = value.to_vec_nodes();
    let self_nodes = var.to_vec_nodes();
    assert_eq!(value_nodes.len(), self_nodes.len());
    current_scope(|b| {
        for (value_node, self_node) in value_nodes.into_iter().zip(self_nodes.into_iter()) {
            b.store(self_node, value_node);
        }
    })
}
macro_rules! impl_aggregate_for_tuple {
    ()=>{
        impl Aggregate for () {
            fn to_nodes(&self, _: &mut Vec<NodeRef>) {}
            fn from_nodes<I: Iterator<Item = NodeRef>>(_: &mut I) -> Self{}
        }
    };
    ($first:ident  $($rest:ident) *) => {
        impl<$first:Aggregate, $($rest: Aggregate),*> Aggregate for ($first, $($rest,)*) {
            #[allow(non_snake_case)]
            fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
                let ($first, $($rest,)*) = self;
                $first.to_nodes(nodes);
                $($rest.to_nodes(nodes);)*
            }
            #[allow(non_snake_case)]
            fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
                let $first = Aggregate::from_nodes(iter);
                $(let $rest = Aggregate::from_nodes(iter);)*
                ($first, $($rest,)*)
            }
        }
        impl_aggregate_for_tuple!($($rest)*);
    };

}
impl_aggregate_for_tuple!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);

pub unsafe trait _Mask: FromNode {}

pub fn select<M: _Mask, A: Selectable>(mask: M, a: A, b: A) -> A {
    let a_nodes = a.to_vec_nodes();
    let b_nodes = b.to_vec_nodes();
    assert_eq!(a_nodes.len(), b_nodes.len());
    let mut ret = vec![];
    current_scope(|b| {
        for (a_node, b_node) in a_nodes.into_iter().zip(b_nodes.into_iter()) {
            assert_eq!(a_node.type_(), b_node.type_());
            assert!(!a_node.is_local(), "cannot select local variables");
            assert!(!b_node.is_local(), "cannot select local variables");
            ret.push(b.call(Func::Select, &[mask.node(), a_node, b_node], a_node.type_()));
        }
    });
    A::from_vec_nodes(ret)
}
impl FromNode for bool {
    fn from_node(_: NodeRef) -> Self {
        panic!("don't call this")
    }
    fn node(&self) -> NodeRef {
        const_(*self).node()
    }
}
unsafe impl _Mask for bool {}
unsafe impl _Mask for Bool {}

pub trait ExprProxy<T>: Copy + Aggregate + FromNode {}

pub trait VarProxy<T: Value>: Copy + Aggregate + FromNode {
    fn store(&self, value: Expr<T>) {
        _store(self, &value);
    }
    fn load(&self) -> Expr<T> {
        current_scope(|b| {
            let nodes = self.to_vec_nodes();
            let mut ret = vec![];
            for node in nodes {
                ret.push(b.call(Func::Load, &[node], node.type_()));
            }
            Expr::<T>::from_nodes(&mut ret.into_iter())
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PrimExpr<T> {
    pub(crate) node: NodeRef,
    pub(crate) _phantom: std::marker::PhantomData<T>,
}
#[derive(Clone, Copy, Debug)]
pub struct PrimVar<T> {
    pub(crate) node: NodeRef,
    pub(crate) _phantom: std::marker::PhantomData<T>,
}
impl<T> Selectable for PrimExpr<T> {}
impl<T> Aggregate for PrimExpr<T> {
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
impl<T> Aggregate for PrimVar<T> {
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
        impl From<$t> for PrimExpr<$t> {
            fn from(v: $t) -> Self {
                const_(v)
            }
        }
        impl FromNode for PrimExpr<$t> {
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
        impl FromNode for PrimVar<$t> {
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
        impl ExprProxy<$t> for PrimExpr<$t> {}
        impl VarProxy<$t> for PrimVar<$t> {}
        impl Value for $t {
            type Expr = PrimExpr<$t>;
            type Var = PrimVar<$t>;
            fn fields() -> Vec<String> {
                vec![]
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

pub type Bool = PrimExpr<bool>;
pub type Float32 = PrimExpr<f32>;
pub type Float64 = PrimExpr<f64>;
pub type Int32 = PrimExpr<i32>;
pub type Int64 = PrimExpr<i64>;
pub type Uint32 = PrimExpr<u32>;
pub type Uint64 = PrimExpr<u64>;

pub type BoolVar = PrimVar<bool>;
pub type Float32Var = PrimVar<f32>;
pub type Float64Var = PrimVar<f64>;
pub type Int32Var = PrimVar<i32>;
pub type Int64Var = PrimVar<i64>;
pub type Uint32Var = PrimVar<u32>;
pub type Uint64Var = PrimVar<u64>;

pub(crate) struct Recorder {
    scopes: Vec<IrBuilder>,
    lock: bool,
    captured_buffer: HashMap<u64, (NodeRef, BufferBinding, Arc<BufferHandle>)>,
}
impl Recorder {
    fn reset(&mut self) {
        self.scopes.clear();
        self.lock = false;
    }
}
thread_local! {
    pub(crate) static RECORDER: RefCell<Recorder> = RefCell::new(Recorder {
        scopes: vec![],
        lock:false,
        captured_buffer: HashMap::new(),
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
    let inst = &node.get().instruction;
    current_scope(|b| {
        let i = b.const_(Const::Int32(index as i32));
        let op = match inst.as_ref() {
            Instruction::Local { .. } => Func::GetElementPtr,
            _ => Func::ExtractElement,
        };
        let node = b.call(op, &[node, i], <T as TypeOf>::type_());
        node
    })
}
pub fn __insert<T: Value>(node: NodeRef, index: usize, value: NodeRef) -> NodeRef {
    let inst = &node.get().instruction;
    current_scope(|b| {
        let i = b.const_(Const::Int32(index as i32));
        let op = match inst.as_ref() {
            Instruction::Local { .. } => panic!("Can't insert into local variable"),
            _ => Func::InsertElement,
        };
        let node = b.call(op, &[node, value, i], <T as TypeOf>::type_());
        node
    })
}
pub fn __compose<T: Value>(nodes: &[NodeRef]) -> NodeRef {
    let ty = <T as TypeOf>::type_();
    match ty.as_ref() {
        Type::Struct(st) => {
            assert_eq!(st.fields.as_ref().len(), nodes.len());
            current_scope(|b| b.call(Func::Struct, nodes, <T as TypeOf>::type_()))
        }
        Type::Primitive(_) => panic!("Can't compose primitive type"),
        Type::Vector(vt) => {
            let length = vt.length;
            let func = match length {
                2 => Func::Vec2,
                3 => Func::Vec3,
                4 => Func::Vec4,
                _ => panic!("Can't compose vector with length {}", length),
            };
            current_scope(|b| b.call(func, nodes, <T as TypeOf>::type_()))
        }
        _ => todo!(),
    }
}
pub fn local<T: Value>(init: Expr<T>) -> Var<T> {
    Var::<T>::from_node(current_scope(|b| b.local(init.node())))
}
pub fn local_zeroed<T: Value>() -> Var<T> {
    Var::<T>::from_node(current_scope(|b| b.local_zero_init(<T as TypeOf>::type_())))
}
pub fn thread_id() -> Expr<UVec3> {
    Expr::<UVec3>::from_node(current_scope(|b| {
        b.call(Func::ThreadId, &[], UVec3::type_())
    }))
}

pub fn block_id() -> Expr<UVec3> {
    Expr::<UVec3>::from_node(current_scope(|b| {
        b.call(Func::BlockId, &[], UVec3::type_())
    }))
}
pub fn dispatch_id() -> Expr<UVec3> {
    Expr::<UVec3>::from_node(current_scope(|b| {
        b.call(Func::DispatchId, &[], UVec3::type_())
    }))
}
pub fn dispatch_size() -> Expr<UVec3> {
    Expr::<UVec3>::from_node(current_scope(|b| {
        b.call(Func::DispatchSize, &[], UVec3::type_())
    }))
}
pub type Expr<T> = <T as Value>::Expr;
pub type Var<T> = <T as Value>::Var;

pub fn const_<T: Value + Copy + 'static>(value: T) -> T::Expr {
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
    FromNode::from_node(node)
}

pub struct BufferVar<T: Value> {
    marker: std::marker::PhantomData<T>,
    #[allow(dead_code)]
    handle: Option<Arc<BufferHandle>>,
    node: NodeRef,
}

impl<T: Value> Drop for BufferVar<T> {
    fn drop(&mut self) {}
}
pub struct BindlessArrayVar {
    node: NodeRef,
}
impl BindlessArrayVar {
    pub fn buffer_read<T: Value, BI: Into<Expr<u32>>, EI: Into<Expr<u32>>>(
        &self,
        buffer_index: BI,
        element_index: EI,
    ) -> Expr<T> {
        Expr::<T>::from_node(current_scope(|b| {
            b.call(
                Func::BindlessBufferRead,
                &[
                    self.node,
                    FromNode::node(&buffer_index.into()),
                    FromNode::node(&element_index.into()),
                ],
                T::type_(),
            )
        }))
    }
    pub fn buffer_length<I: Into<Expr<u32>>>(&self, buffer_index: I) -> Expr<u32> {
        <Expr<u32> as FromNode>::from_node(current_scope(|b| {
            b.call(
                Func::BindlessBufferSize,
                &[self.node, FromNode::node(&buffer_index.into())],
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
            if let Some((node, _, _)) = r.captured_buffer.get(&handle) {
                *node
            } else {
                let node = new_node(Node::new(Gc::new(Instruction::Buffer), T::type_()));
                r.captured_buffer.insert(
                    handle,
                    (
                        node,
                        BufferBinding {
                            handle: buffer.handle().0,
                            size: buffer.size_bytes(),
                            offset: 0,
                        },
                        buffer.handle.clone(),
                    ),
                );
                node
            }
        });
        Self {
            node,
            marker: std::marker::PhantomData,
            handle: Some(buffer.handle.clone()),
        }
    }
    pub fn len(&self) -> Expr<u32> {
        FromNode::from_node(
            current_scope(|b| b.call(Func::BufferSize, &[self.node], u32::type_())).into(),
        )
    }
    pub fn read<I: Into<Expr<u32>>>(&self, i: I) -> Expr<T> {
        let i = i.into();
        current_scope(|b| {
            FromNode::from_node(b.call(
                Func::BufferRead,
                &[self.node, FromNode::node(&i)],
                T::type_(),
            ))
        })
    }
    pub fn write<I: Into<Expr<u32>>, V: Into<Expr<T>>>(&self, i: I, v: V) {
        let i = i.into();
        let v = v.into();
        current_scope(|b| {
            b.call(
                Func::BufferWrite,
                &[self.node, FromNode::node(&i), v.node()],
                Type::void(),
            )
        });
    }
    pub fn atomic_exchange<I: Into<Expr<u32>>, V: Into<Expr<T>>>(&self, i: I, v: V) -> Expr<T> {
        todo!()
    }
}

pub struct ImageVar<T: Texel> {
    node: NodeRef,
    #[allow(dead_code)]
    handle: Option<Arc<TextureHandle>>,
    _marker: std::marker::PhantomData<T>,
}

pub struct VolumeVar<T: Texel> {
    node: NodeRef,
    #[allow(dead_code)]
    handle: Option<Arc<TextureHandle>>,
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

// Not recommended to use this directly
pub struct KernelBuilder {
    device: crate::runtime::Device,
    args: Vec<NodeRef>,
}
pub trait KernelParameter {
    type Arg: KernelArg;
    fn def_param(builder: &mut KernelBuilder) -> Self;
}
impl<T: Value> KernelParameter for BufferVar<T> {
    type Arg = Buffer<T>;
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.buffer()
    }
}
impl<T: Texel> KernelParameter for Tex2DVar<T> {
    type Arg = Tex2D<T>;
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.tex2d()
    }
}
impl<T: Texel> KernelParameter for Tex3DVar<T> {
    type Arg = Tex3D<T>;
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.tex3d()
    }
}
impl KernelParameter for BindlessArrayVar {
    type Arg = BindlessArray;
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.bindless_array()
    }
}
macro_rules! impl_kernel_param_for_tuple {
    ($first:ident  $($rest:ident)*) => {
        impl<$first:KernelParameter, $($rest: KernelParameter),*> KernelParameter for ($first, $($rest,)*) {
            type Arg = ($first::Arg, $($rest::Arg),*);
            #[allow(non_snake_case)]
            fn def_param(builder: &mut KernelBuilder) -> Self {
                ($first::def_param(builder), $($rest::def_param(builder)),*)
            }
        }
        impl_kernel_param_for_tuple!($($rest)*);
    };
    ()=>{
        impl KernelParameter for () {
            type Arg = ();
            fn def_param(_: &mut KernelBuilder) -> Self {
            }
        }
    }
}
impl_kernel_param_for_tuple!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);
impl KernelBuilder {
    pub fn new(device: crate::runtime::Device) -> Self {
        RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            assert!(!r.lock, "Cannot record multiple kernels at the same time");
            assert!(
                r.scopes.is_empty(),
                "Cannot record multiple kernels at the same time"
            );
            r.lock = true;
            r.scopes.clear();
        });
        Self {
            device,
            args: vec![],
        }
    }
    pub fn buffer<T: Value>(&mut self) -> BufferVar<T> {
        let node = new_node(Node::new(Gc::new(Instruction::Buffer), T::type_()));
        self.args.push(node);
        BufferVar {
            node,
            marker: std::marker::PhantomData,
            handle: None,
        }
    }
    pub fn tex2d<T: Texel>(&mut self) -> ImageVar<T> {
        todo!()
    }
    pub fn tex3d<T: Texel>(&mut self) -> VolumeVar<T> {
        todo!()
    }
    pub fn bindless_array(&mut self) -> BindlessArrayVar {
        let node = new_node(Node::new(Gc::new(Instruction::Bindless), u32::type_()));
        self.args.push(node);
        BindlessArrayVar { node }
    }
    pub(crate) fn build(
        device: crate::runtime::Device,
        f: impl FnOnce(&mut Self),
    ) -> Result<crate::runtime::RawKernel, crate::backend::BackendError> {
        let mut builder = Self::new(device);

        builder.build_(f)
    }
    fn build_(
        &mut self,
        body: impl FnOnce(&mut Self),
    ) -> Result<crate::runtime::RawKernel, crate::backend::BackendError> {
        body(self);
        RECORDER.with(
            |r| -> Result<crate::runtime::RawKernel, crate::backend::BackendError> {
                let mut resource_tracker: Vec<Box<dyn Any>> = Vec::new();
                let mut r = r.borrow_mut();
                assert!(r.lock);
                r.lock = false;
                assert_eq!(r.scopes.len(), 1);
                let scope = r.scopes.pop().unwrap();
                let entry = scope.finish();
                let mut captured: Vec<Capture> = Vec::new();
                for (_, (node, binding, handle)) in r.captured_buffer.iter() {
                    captured.push(Capture {
                        node: *node,
                        binding: Binding::Buffer(binding.clone()),
                    });
                    resource_tracker.push(Box::new(handle.clone()));
                }
                let module = KernelModule {
                    module: Module {
                        entry,
                        kind: ModuleKind::Kernel,
                    },
                    captures: CBoxedSlice::new(captured),
                    shared: CBoxedSlice::new(vec![]),
                    args: CBoxedSlice::new(self.args.clone()),
                };
                // build kernel here
                let shader = self.device.inner.create_shader(&module, "")?;
                //
                r.reset();
                Ok(RawKernel {
                    shader,
                    device: self.device.clone(),
                    resource_tracker,
                })
            },
        )
    }
}

pub trait KernelBuildFn {
    type Output;
    fn build(&self, builder: &mut KernelBuilder) -> Self::Output;
}
macro_rules! impl_kernel_build_for_fn {
    ()=>{
        impl KernelBuildFn for Box<dyn Fn()> {
            type Output = backend::Result<Kernel<()>>;
            fn build(&self, builder: &mut KernelBuilder) -> backend::Result<Kernel<()>> {
                let kernel = builder.build_(|_| {
                    self()
                })?;
                let kernel = Kernel {
                    inner: kernel,
                    _marker: std::marker::PhantomData,
                };
                Ok(kernel)
            }
        }
    };
    ($first:ident  $($rest:ident)*) => {
        impl<$first:KernelParameter, $($rest: KernelParameter),*> KernelBuildFn for Box<dyn Fn($first, $($rest,)*)> {
            type Output = backend::Result<Kernel<($first::Arg, $($rest::Arg),*)>>;
            #[allow(non_snake_case)]
            fn build(&self, builder: &mut KernelBuilder) -> backend::Result<Kernel<($first::Arg, $($rest::Arg),*)>> {
                let kernel = builder.build_(|builder| {
                    let $first = $first::def_param(builder);
                    $(let $rest = $rest::def_param(builder);)*
                    self($first, $($rest,)*)
                })?;
                let kernel = Kernel {
                    inner: kernel,
                    _marker: std::marker::PhantomData,
                };
                Ok(kernel)
            }
        }
        impl_kernel_build_for_fn!($($rest)*);
    };
}
impl_kernel_build_for_fn!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);

pub fn if_<R: Aggregate>(
    cond: impl _Mask,
    then: impl FnOnce() -> R,
    else_: impl FnOnce() -> R,
) -> R {
    RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let s = &mut r.scopes;
        s.push(IrBuilder::new());
    });
    let then = then();
    let then_block = RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let s = &mut r.scopes;
        let then_block = s.pop().unwrap().finish();
        s.push(IrBuilder::new());
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
    current_scope(|b| {
        b.if_(cond.node(), then_block, else_block);
    });
    assert_eq!(then_nodes.len(), else_nodes.len());
    let phis = current_scope(|b| {
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
                let phi = new_node(Node::new(
                    Gc::new(Instruction::Phi(CBoxedSlice::new(incomings))),
                    then.type_(),
                ));
                b.append(phi);
                phi
            })
            .collect::<Vec<_>>()
    });
    R::from_vec_nodes(phis)
}
pub fn generic_loop(cond: impl FnOnce() -> Bool, body: impl FnOnce(), update: impl FnOnce()) {
    RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let s = &mut r.scopes;
        s.push(IrBuilder::new());
    });
    let cond_v = cond().node();
    let prepare = RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let s = &mut r.scopes;
        let prepare = s.pop().unwrap().finish();
        s.push(IrBuilder::new());
        prepare
    });
    body();
    let body = RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let s = &mut r.scopes;
        let body = s.pop().unwrap().finish();
        s.push(IrBuilder::new());
        body
    });
    update();
    let update = RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let s = &mut r.scopes;
        s.pop().unwrap().finish()
    });
    current_scope(|b| {
        b.generic_loop(prepare, cond_v, body, update);
    });
}
#[macro_export]
macro_rules! if_ {
    ($cond:expr, $then:block, else $else_:block) => {
        if_($cond, || $then, || $else_)
    };
    ($cond:expr, $then:block) => {
        if_($cond, || $then, || {})
    };
}
#[macro_export]
macro_rules! while_ {
    ($cond:expr,$body:block) => {
        generic_loop(|| $cond, || $body, || {})
    };
}

pub fn break_() {
    current_scope(|b| {
        b.break_();
    });
}
pub fn continue_() {
    current_scope(|b| {
        b.continue_();
    });
}