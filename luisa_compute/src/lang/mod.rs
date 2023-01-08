use std::io::stderr;
use std::marker::PhantomData;
use std::process::abort;
use std::{any::Any, collections::HashMap, fmt::Debug, ops::Deref, sync::Arc};

use crate::lang::traits::VarCmp;
use crate::{
    backend,
    prelude::{Device, Kernel, KernelArg, RawKernel},
    resource::{
        BindlessArray, BindlessArrayHandle, Buffer, BufferHandle, Tex2D, Tex3D, Texel,
        TextureHandle,
    },
};
pub use ir::ir::NodeRef;
use ir::ir::{BindlessArrayBinding, SwitchCase};
use ir::{
    ir::{
        new_node, BasicBlock, Binding, BufferBinding, Capture, Const, CpuCustomOp, Func,
        Instruction, IrBuilder, KernelModule, Module, ModuleKind, Node, PhiIncoming,
    },
    transform::{self, Transform},
    CRc,
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
pub mod swizzle;
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
    fn store<U: Into<Expr<T>>>(&self, value: U) {
        let value = value.into();
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

pub struct CpuFn<T: Value> {
    op: CRc<CpuCustomOp>,
    _marker: std::marker::PhantomData<T>,
}
#[macro_export]
macro_rules! cpu_dbg {
    ($t:ty, $arg:expr) => {{
        __cpu_dbg::<$t>($arg, file!(), line!())
    }};
}
pub fn __cpu_dbg<T: Value + Debug>(arg: Expr<T>, file: &'static str, line: u32) {
    let f = CpuFn::new(move |x: &mut T| {
        println!("[{}:{}] {:?}", file, line, x);
    });
    let _ = f.call(arg);
}
extern "C" fn _trampoline<T, F: FnMut(&mut T)>(data: *mut u8, args: *mut u8) {
    unsafe {
        let container = &*(data as *const ClosureContainer<T>);
        let f = &container.f;
        let args = &mut *(args as *mut T);
        f(args);
    }
}
extern "C" fn _drop<T>(data: *mut u8) {
    unsafe {
        let _ = Box::from_raw(data as *mut T);
    }
}
/*
Interestingly, Box::into_raw(Box<Closure>) does not give a valid pointer.
*/
struct ClosureContainer<T> {
    f: Arc<dyn Fn(&mut T) + 'static + Send + Sync>,
}
impl<T: Value> CpuFn<T> {
    pub fn new<F: Fn(&mut T) + 'static + Send + Sync>(f: F) -> Self {
        let f_ptr = Box::into_raw(Box::new(ClosureContainer::<T> { f: Arc::new(f) }));
        let op = CpuCustomOp {
            data: f_ptr as *mut u8,
            func: _trampoline::<T, F>,
            destructor: _drop::<F>,
            arg_type: T::type_(),
        };
        Self {
            op: CRc::new(op),
            _marker: std::marker::PhantomData,
        }
    }
    pub fn call(&self, arg: Expr<T>) -> Expr<T> {
        RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            assert!(r.lock);
            assert!(
                r.device.as_ref().unwrap().inner.is_cpu_backend(),
                "CpuFn can only be used in cpu backend"
            );
            let addr = CRc::as_ptr(&self.op) as u64;
            if let Some((_, op)) = r.cpu_custom_ops.get(&addr) {
                assert_eq!(CRc::as_ptr(op), CRc::as_ptr(&self.op));
            } else {
                let i = r.cpu_custom_ops.len();
                r.cpu_custom_ops.insert(addr, (i, self.op.clone()));
            }
        });
        Expr::<T>::from_node(current_scope(|b| {
            b.call(
                Func::CpuCustomOp(self.op.clone()),
                &[arg.node()],
                T::type_(),
            )
        }))
    }
}
pub(crate) struct Recorder {
    scopes: Vec<IrBuilder>,
    lock: bool,
    captured_buffer: HashMap<u64, (usize, NodeRef, Binding, Arc<dyn Any>)>,
    cpu_custom_ops: HashMap<u64, (usize, CRc<CpuCustomOp>)>,
    device: Option<Device>,
    block_size: Option<[u32; 3]>,
}
impl Recorder {
    fn reset(&mut self) {
        self.scopes.clear();
        self.captured_buffer.clear();
        self.cpu_custom_ops.clear();
        self.lock = false;
        self.device = None;
        self.block_size = None;
    }
}
thread_local! {
    pub(crate) static RECORDER: RefCell<Recorder> = RefCell::new(Recorder {
        scopes: vec![],
        lock:false,
        captured_buffer: HashMap::new(),
        cpu_custom_ops: HashMap::new(),
        device:None,
        block_size: None,
    });
}

// Don't call this function directly unless you know what you are doing
pub fn current_scope<F: FnOnce(&mut IrBuilder) -> R, R>(f: F) -> R {
    RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        assert!(r.lock, "current_scope must be called within a kernel");
        let s = &mut r.scopes;
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
        Type::Matrix(vt) => {
            let length = vt.dimension;
            let func = match length {
                2 => Func::Mat2,
                3 => Func::Mat3,
                4 => Func::Mat4,
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
pub fn set_block_size(size: [u32; 3]) {
    RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        assert!(r.block_size.is_none(), "Block size already set");
        r.block_size = Some(size);
    });
}
pub fn block_size() -> Expr<UVec3> {
    RECORDER.with(|r| {
        let r = r.borrow();
        let s = r.block_size.unwrap_or_else(|| panic!("Block size not set"));
        const_::<UVec3>(UVec3::new(s[0], s[1], s[2]))
    })
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
pub fn bitcast<From: Value, To: Value>(expr: Expr<From>) -> Expr<To> {
    assert_eq!(std::mem::size_of::<From>(), std::mem::size_of::<To>());
    Expr::<To>::from_node(current_scope(|b| {
        b.call(Func::Bitcast, &[expr.node()], <To as TypeOf>::type_())
    }))
}

impl<T: Value, const N: usize> Value for [T; N] {
    type Expr = ArrayExpr<T, N>;
    type Var = ArrayVar<T, N>;
    fn fields() -> Vec<String> {
        todo!("why this method exists?")
    }
}
#[derive(Clone, Copy, Debug)]
pub struct ArrayExpr<T: Value, const N: usize> {
    marker: std::marker::PhantomData<T>,
    node: NodeRef,
}
#[derive(Clone, Copy, Debug)]
pub struct ArrayVar<T: Value, const N: usize> {
    marker: std::marker::PhantomData<T>,
    node: NodeRef,
}
impl<T: Value, const N: usize> FromNode for ArrayExpr<T, N> {
    fn from_node(node: NodeRef) -> Self {
        Self {
            marker: std::marker::PhantomData,
            node,
        }
    }
    fn node(&self) -> NodeRef {
        self.node
    }
}
impl<T: Value, const N: usize> Aggregate for ArrayExpr<T, N> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        nodes.push(self.node);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        Self::from_node(iter.next().unwrap())
    }
}
impl<T: Value, const N: usize> FromNode for ArrayVar<T, N> {
    fn from_node(node: NodeRef) -> Self {
        Self {
            marker: std::marker::PhantomData,
            node,
        }
    }
    fn node(&self) -> NodeRef {
        self.node
    }
}
impl<T: Value, const N: usize> Aggregate for ArrayVar<T, N> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        nodes.push(self.node);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        Self::from_node(iter.next().unwrap())
    }
}
impl<T: Value, const N: usize> ExprProxy<[T; N]> for ArrayExpr<T, N> {}
impl<T: Value, const N: usize> VarProxy<[T; N]> for ArrayVar<T, N> {}
impl<T: Value, const N: usize> ArrayVar<T, N> {
    pub fn read<I: Into<Expr<u32>>>(&self, i: I) -> Expr<T> {
        let i = i.into();
        if __env_need_backtrace() {
            assert(i.cmplt(const_(N as u32)));
        }
        Expr::<T>::from_node(current_scope(|b| {
            let gep = b.call(Func::GetElementPtr, &[self.node, i.node()], T::type_());
            b.call(Func::Load, &[gep], T::type_())
        }))
    }
    pub fn len(&self) -> Expr<u32> {
        const_(N as u32)
    }
    pub fn write<I: Into<Expr<u32>>, V: Into<Expr<T>>>(&self, i: I, value: V) {
        let i = i.into();
        let value = value.into();
        if __env_need_backtrace() {
            assert(i.cmplt(const_(N as u32)));
        }
        current_scope(|b| {
            let gep = b.call(Func::GetElementPtr, &[self.node, i.node()], T::type_());
            b.update(gep, value.node());
        });
    }
}
impl<T: Value, const N: usize> ArrayExpr<T, N> {
    pub fn zero() -> Self {
        let node = current_scope(|b| b.call(Func::ZeroInitializer, &[], <[T; N]>::type_()));
        Self::from_node(node)
    }
    pub fn read<I: Into<Expr<u32>>>(&self, i: I) -> Expr<T> {
        let i = i.into();
        if __env_need_backtrace() {
            assert(i.cmplt(const_(N as u32)));
        }
        Expr::<T>::from_node(current_scope(|b| {
            let gep = b.call(Func::GetElementPtr, &[self.node, i.node()], T::type_());
            b.call(Func::Load, &[gep], T::type_())
        }))
    }
    pub fn len(&self) -> Expr<u32> {
        const_(N as u32)
    }
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
    #[allow(dead_code)]
    handle: Option<Arc<BindlessArrayHandle>>,
}
pub struct BindlessBufferVar<T> {
    array: NodeRef,
    buffer_index: Expr<u32>,
    _marker: std::marker::PhantomData<T>,
}
impl<T: Value> BindlessBufferVar<T> {
    pub fn read<I: Into<Expr<u32>>>(&self, i: I) -> Expr<T> {
        let i = i.into();
        if __env_need_backtrace() {
            assert(i.cmplt(self.len()));
        }
        Expr::<T>::from_node(current_scope(|b| {
            b.call(
                Func::BindlessBufferRead,
                &[self.array, self.buffer_index.node(), FromNode::node(&i)],
                T::type_(),
            )
        }))
    }
    pub fn len(&self) -> Expr<u32> {
        Expr::<u32>::from_node(current_scope(|b| {
            b.call(
                Func::BindlessBufferSize(T::type_()),
                &[self.array, self.buffer_index.node()],
                T::type_(),
            )
        }))
    }
    pub fn __type(&self) -> Expr<u64> {
        Expr::<u64>::from_node(current_scope(|b| {
            b.call(
                Func::BindlessBufferType,
                &[self.array, self.buffer_index.node()],
                u64::type_(),
            )
        }))
    }
}
impl BindlessArrayVar {
    pub fn buffer<T: Value>(&self, buffer_index: Expr<u32>) -> BindlessBufferVar<T> {
        let v = BindlessBufferVar {
            array: self.node,
            buffer_index,
            _marker: std::marker::PhantomData,
        };
        let vt = v.__type();
        if __env_need_backtrace() {
            let backtrace = backtrace::Backtrace::new();
            let check_type = CpuFn::new(move |t: &mut u64| {
                let expected = T::type_();
                if *t != Gc::as_ptr(expected) as u64 {
                    let t = unsafe { &*(*t as *const Type) };
                    eprintln!(
                            "Bindless buffer type mismatch: expected {:?}, got {:?}; host backtrace:\n {:?}",
                            expected, t, backtrace
                        );
                    abort();
                }
            });
            let _ = check_type.call(vt);
        } else {
            let check_type = CpuFn::new(move |t: &mut u64| {
                let expected = T::type_();
                if *t != Gc::as_ptr(expected) as u64 {
                    let t = unsafe { &*(*t as *const Type) };
                    eprintln!(
                        "Bindless buffer type mismatch: expected {:?}, got {:?}; set LUISA_BACKTRACE=1 for more info",
                        expected, t,
                    );
                    abort();
                }
            });
            let _ = check_type.call(vt);
        }
        v
    }

    pub fn new(array: &BindlessArray) -> Self {
        let node = RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            assert!(r.lock, "BufferVar must be created from within a kernel");
            let handle: u64 = array.handle().0;
            if let Some((_, node, _, _)) = r.captured_buffer.get(&handle) {
                *node
            } else {
                let node = new_node(Node::new(Gc::new(Instruction::Bindless), Type::void()));
                let i = r.captured_buffer.len();
                r.captured_buffer.insert(
                    handle,
                    (
                        i,
                        node,
                        Binding::BindlessArray(BindlessArrayBinding { handle }),
                        Arc::new(array.handle.clone()),
                    ),
                );
                node
            }
        });
        Self {
            node,
            handle: Some(array.handle.clone()),
        }
    }
}
impl<T: Value> BufferVar<T> {
    pub fn new(buffer: &Buffer<T>) -> Self {
        let node = RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            assert!(r.lock, "BufferVar must be created from within a kernel");
            let handle: u64 = buffer.handle().0;
            if let Some((_, node, _, _)) = r.captured_buffer.get(&handle) {
                *node
            } else {
                let node = new_node(Node::new(Gc::new(Instruction::Buffer), T::type_()));
                let i = r.captured_buffer.len();
                r.captured_buffer.insert(
                    handle,
                    (
                        i,
                        node,
                        Binding::Buffer(BufferBinding {
                            handle: buffer.handle().0,
                            size: buffer.size_bytes(),
                            offset: 0,
                        }),
                        Arc::new(buffer.handle.clone()),
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
        if __env_need_backtrace() {
            assert(i.cmplt(self.len()));
        }
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
        if __env_need_backtrace() {
            assert(i.cmplt(self.len()));
        }
        current_scope(|b| {
            b.call(
                Func::BufferWrite,
                &[self.node, FromNode::node(&i), v.node()],
                Type::void(),
            )
        });
    }
}

macro_rules! impl_atomic {
    ($t:ty) => {
        impl BufferVar<$t> {
            pub fn atomic_exchange<I: Into<Expr<u32>>, V: Into<Expr<$t>>>(
                &self,
                i: I,
                v: V,
            ) -> Expr<$t> {
                let i = i.into();
                let v = v.into();
                Expr::<$t>::from_node(current_scope(|b| {
                    b.call(
                        Func::AtomicExchange,
                        &[self.node, FromNode::node(&i), v.node()],
                        <$t>::type_(),
                    )
                }))
            }
            pub fn atomic_compare_exchange<
                I: Into<Expr<u32>>,
                V0: Into<Expr<$t>>,
                V1: Into<Expr<$t>>,
            >(
                &self,
                i: I,
                expected: V0,
                desired: V1,
            ) -> Expr<$t> {
                let i = i.into();
                let expected = expected.into();
                let desired = desired.into();
                Expr::<$t>::from_node(current_scope(|b| {
                    b.call(
                        Func::AtomicCompareExchange,
                        &[
                            self.node,
                            FromNode::node(&i),
                            expected.node(),
                            desired.node(),
                        ],
                        <$t>::type_(),
                    )
                }))
            }
            pub fn atomic_fetch_add<I: Into<Expr<u32>>, V: Into<Expr<$t>>>(
                &self,
                i: I,
                v: V,
            ) -> Expr<$t> {
                let i = i.into();
                let v = v.into();
                Expr::<$t>::from_node(current_scope(|b| {
                    b.call(
                        Func::AtomicFetchAdd,
                        &[self.node, FromNode::node(&i), v.node()],
                        <$t>::type_(),
                    )
                }))
            }
            pub fn atomic_fetch_sub<I: Into<Expr<u32>>, V: Into<Expr<$t>>>(
                &self,
                i: I,
                v: V,
            ) -> Expr<$t> {
                let i = i.into();
                let v = v.into();
                Expr::<$t>::from_node(current_scope(|b| {
                    b.call(
                        Func::AtomicFetchSub,
                        &[self.node, FromNode::node(&i), v.node()],
                        <$t>::type_(),
                    )
                }))
            }
        }
    };
}
macro_rules! impl_atomic_bit {
    ($t:ty) => {
        impl BufferVar<$t> {
            pub fn atomic_fetch_and<I: Into<Expr<u32>>, V: Into<Expr<$t>>>(
                &self,
                i: I,
                v: V,
            ) -> Expr<$t> {
                let i = i.into();
                let v = v.into();
                Expr::<$t>::from_node(current_scope(|b| {
                    b.call(
                        Func::AtomicFetchAnd,
                        &[self.node, FromNode::node(&i), v.node()],
                        <$t>::type_(),
                    )
                }))
            }
            pub fn atomic_fetch_or<I: Into<Expr<u32>>, V: Into<Expr<$t>>>(
                &self,
                i: I,
                v: V,
            ) -> Expr<$t> {
                let i = i.into();
                let v = v.into();
                Expr::<$t>::from_node(current_scope(|b| {
                    b.call(
                        Func::AtomicFetchOr,
                        &[self.node, FromNode::node(&i), v.node()],
                        <$t>::type_(),
                    )
                }))
            }
            pub fn atomic_fetch_xor<I: Into<Expr<u32>>, V: Into<Expr<$t>>>(
                &self,
                i: I,
                v: V,
            ) -> Expr<$t> {
                let i = i.into();
                let v = v.into();
                Expr::<$t>::from_node(current_scope(|b| {
                    b.call(
                        Func::AtomicFetchXor,
                        &[self.node, FromNode::node(&i), v.node()],
                        <$t>::type_(),
                    )
                }))
            }

            pub fn atomic_fetch_min<I: Into<Expr<u32>>, V: Into<Expr<$t>>>(
                &self,
                i: I,
                v: V,
            ) -> Expr<$t> {
                let i = i.into();
                let v = v.into();
                Expr::<$t>::from_node(current_scope(|b| {
                    b.call(
                        Func::AtomicFetchMin,
                        &[self.node, FromNode::node(&i), v.node()],
                        <$t>::type_(),
                    )
                }))
            }
            pub fn atomic_fetch_max<I: Into<Expr<u32>>, V: Into<Expr<$t>>>(
                &self,
                i: I,
                v: V,
            ) -> Expr<$t> {
                let i = i.into();
                let v = v.into();
                Expr::<$t>::from_node(current_scope(|b| {
                    b.call(
                        Func::AtomicFetchMax,
                        &[self.node, FromNode::node(&i), v.node()],
                        <$t>::type_(),
                    )
                }))
            }
        }
    };
}
impl_atomic!(i32);
impl_atomic!(u32);
impl_atomic!(i64);
impl_atomic!(u64);
impl_atomic!(f32);
impl_atomic_bit!(u32);
impl_atomic_bit!(u64);
impl_atomic_bit!(i32);
impl_atomic_bit!(i64);
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
            r.device = Some(device.clone());
            r.scopes.clear();
            r.scopes.push(IrBuilder::new());
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
        BindlessArrayVar { node, handle: None }
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
                let mut resource_tracker: Vec<Arc<dyn Any>> = Vec::new();
                let mut r = r.borrow_mut();
                assert!(r.lock);
                r.lock = false;
                assert_eq!(r.scopes.len(), 1);
                let scope = r.scopes.pop().unwrap();
                let entry = scope.finish();
                let mut captured: Vec<Capture> = Vec::new();
                let mut captured_buffers: Vec<_> = r.captured_buffer.values().cloned().collect();
                captured_buffers.sort_by_key(|(i, _, _, _)| *i);
                for (j, (i, node, binding, handle)) in captured_buffers.into_iter().enumerate() {
                    assert_eq!(j, i);
                    captured.push(Capture {
                        node: node,
                        binding: binding,
                    });
                    resource_tracker.push(handle);
                }
                let mut cpu_custom_ops: Vec<_> = r.cpu_custom_ops.values().cloned().collect();
                cpu_custom_ops.sort_by_key(|(i, _)| *i);
                let cpu_custom_ops = cpu_custom_ops
                    .iter()
                    .enumerate()
                    .map(|(j, (i, op))| {
                        assert_eq!(j, *i);
                        op.clone()
                    })
                    .collect::<Vec<_>>();
                let module = KernelModule {
                    module: Module {
                        entry,
                        kind: ModuleKind::Kernel,
                    },
                    cpu_custom_ops: CBoxedSlice::new(cpu_custom_ops),
                    captures: CBoxedSlice::new(captured),
                    shared: CBoxedSlice::new(vec![]),
                    args: CBoxedSlice::new(self.args.clone()),
                    block_size: r.block_size.unwrap_or([1, 1, 1]),
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
pub trait KernelSigature<'a> {
    type Fn: KernelBuildFn;
}

macro_rules! impl_kernel_signature {
    ()=>{
        impl<'a> KernelSigature<'a> for () {
            type Fn = &'a dyn Fn();
        }
    };
    ($first:ident  $($rest:ident)*) => {
        impl<'a, $first:KernelArg +'static, $($rest: KernelArg +'static),*> KernelSigature<'a> for ($first, $($rest,)*) {
            type Fn = &'a dyn Fn($first::Parameter, $($rest::Parameter),*);
        }
        impl_kernel_signature!($($rest)*);
    };
}
impl_kernel_signature!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);

macro_rules! impl_kernel_build_for_fn {
    ()=>{
        impl KernelBuildFn for &dyn Fn() {
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
        impl<$first:KernelParameter, $($rest: KernelParameter),*> KernelBuildFn for &dyn Fn($first, $($rest,)*) {
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
                let phi = b.phi(&incomings, then.type_());
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
pub struct SwitchBuilder<R: Aggregate> {
    cases: Vec<(i32, Gc<BasicBlock>, Vec<NodeRef>)>,
    default: Option<(Gc<BasicBlock>, Vec<NodeRef>)>,
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
    pub fn case(mut self, value: i32, then: impl FnOnce() -> R) -> Self {
        RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            let s = &mut r.scopes;
            assert_eq!(s.len(), self.depth);
            s.push(IrBuilder::new());
        });
        let then = then();
        let block = pop_scope();
        self.cases.push((value, block, then.to_vec_nodes()));
        self
    }
    pub fn default(mut self, then: impl FnOnce() -> R) -> Self {
        RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            let s = &mut r.scopes;
            assert_eq!(s.len(), self.depth);
            s.push(IrBuilder::new());
        });
        let then = then();
        let block = pop_scope();
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
                let s = &mut r.scopes;
                assert_eq!(s.len(), self.depth);
                s.push(IrBuilder::new());
            });
            for i in 0..phi_count {
                let default_node =
                    current_scope(|b| b.call(Func::Unreachable, &[], case_phis[0][i].type_()));
                default_nodes.push(default_node);
            }
            pop_scope()
        } else {
            default_nodes = self.default.as_ref().unwrap().1.clone();
            self.default.as_ref().unwrap().0
        };
        current_scope(|b| {
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
            let phi = current_scope(|b| b.phi(&incomings, case_phis[0][i].type_()));
            phis.push(phi);
        }
        R::from_vec_nodes(phis)
    }
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
// pub fn return_v<T: FromNode>(v: T) {
//     current_scope(|b| {
//         b.return_(Some(v.node()));
//     });
// }
pub fn return_() {
    current_scope(|b| {
        b.return_(None);
    });
}
struct AdContext {
    started: bool,
    backward_called: bool,
    forward: Option<Gc<BasicBlock>>,
}
impl AdContext {
    fn new() -> Self {
        Self {
            started: false,
            backward_called: false,
            forward: None,
        }
    }
    fn reset(&mut self) {
        *self = Self::new();
    }
}
thread_local! {
    static AD_CONTEXT:RefCell<AdContext> = RefCell::new(AdContext::new());
}
pub fn requires_grad<T: Value>(var: impl ExprProxy<T>) {
    current_scope(|b| {
        b.call(Func::RequiresGradient, &[var.node()], Type::void());
    });
}
pub fn backward<T: Value>(out: impl ExprProxy<T>) {
    backward_with_grad(
        out,
        FromNode::from_node(current_scope(|b| {
            let one = new_node(Node::new(
                Gc::new(Instruction::Const(Const::One(T::type_()))),
                T::type_(),
            ));
            b.append(one);
            one
        })),
    );
}
pub fn backward_with_grad<T: Value, U: ExprProxy<T>>(out: U, grad: U) {
    AD_CONTEXT.with(|c| {
        let mut c = c.borrow_mut();
        assert!(c.started, "autodiff section is not started");
        assert!(!c.backward_called, "backward is already called");
        c.backward_called = true;
    });
    let out = out.node();
    let grad = grad.node();
    current_scope(|b| {
        b.call(Func::GradientMarker, &[out, grad], Type::void());
    });
    let fwd = pop_scope();
    AD_CONTEXT.with(|c| {
        let mut c = c.borrow_mut();
        c.forward = Some(fwd);
    });
    RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let s = &mut r.scopes;
        s.push(IrBuilder::new());
    });
}
pub fn gradient<T: Value, U: ExprProxy<T>>(var: U) -> U {
    U::from_node(current_scope(|b| {
        b.call(Func::Gradient, &[var.node()], var.node().type_())
    }))
}
pub fn grad<T: Value, U: ExprProxy<T>>(var: U) -> U {
    gradient(var)
}
pub fn autodiff(body: impl FnOnce()) {
    AD_CONTEXT.with(|c| {
        let mut c = c.borrow_mut();
        assert!(!c.started, "autodiff section is already started");
        c.started = true;
    });
    RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let s = &mut r.scopes;
        s.push(IrBuilder::new());
    });
    body();
    let fwd = AD_CONTEXT.with(|c| {
        let mut c = c.borrow_mut();
        assert!(c.started, "autodiff section is not started");
        assert!(c.backward_called, "backward is not called");
        c.forward.take().unwrap()
    });
    let fwd_module = Module {
        kind: ModuleKind::Block,
        entry: fwd,
    };
    let ad_transform = transform::autodiff::Autodiff;
    let ad_module = ad_transform.transform(fwd_module);
    let epilogue = RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let s = &mut r.scopes;
        s.pop().unwrap().finish()
    });
    let fwd_bwd = ad_module.entry;

    AD_CONTEXT.with(|c| {
        let mut c = c.borrow_mut();
        assert!(c.started, "autodiff section is not started");
        assert!(c.backward_called, "backward is not called");
        c.reset();
    });
    current_scope(|b| {
        b.append_block(fwd_bwd);
        b.append_block(epilogue);
    })
}
pub fn is_cpu_backend() -> bool {
    RECORDER.with(|r| {
        let r = r.borrow();
        r.device.as_ref().unwrap().inner.is_cpu_backend()
    })
}
pub fn __env_need_backtrace() -> bool {
    match std::env::var("LUISA_BACKTRACE") {
        Ok(s) => s == "1" || s == "ON",
        Err(_) => false,
    }
}
pub fn assert(cond: Expr<bool>) {
    if is_cpu_backend() && __env_need_backtrace() {
        let backtrace = backtrace::Backtrace::new();
        let assert_fn = CpuFn::new(move |b: &mut bool| {
            if !*b {
                eprintln!("assertion failed with host backtrace:\n {:?}", backtrace);
            }
        });
        let _ = assert_fn.call(cond);
    }
    current_scope(|b| {
        b.call(Func::Assert, &[cond.node()], Type::void());
    });
}
