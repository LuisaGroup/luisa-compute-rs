use std::backtrace::Backtrace;
use std::collections::HashSet;
use std::marker::PhantomData;
use std::{any::Any, collections::HashMap, fmt::Debug, rc::Rc, sync::Arc};
use std::{env, unreachable};

use crate::lang::traits::VarCmp;
pub use crate::runtime::CallableArgEncoder;
use crate::runtime::{AsyncShaderArtifact, ShaderArtifact};
use crate::*;
use crate::{rtx, ResourceTracker};
use bumpalo::Bump;
use indexmap::IndexMap;
pub use ir::ir::NodeRef;
use ir::ir::{
    ArrayType, CallableModule, CallableModuleRef, ModulePools, SwitchCase, UserNodeData,
    INVALID_REF,
};
pub use ir::CArc;
use ir::Pooled;
use ir::{
    ir::{
        new_node, BasicBlock, Binding, Capture, Const, CpuCustomOp, Func, Instruction, IrBuilder,
        KernelModule, Module, ModuleKind, Node, PhiIncoming,
    },
    transform::{self, Transform},
};

use luisa_compute_ir as ir;

pub use luisa_compute_ir::{
    context::register_type,
    ffi::CBoxedSlice,
    ir::{StructType, Type},
    TypeOf,
};
use math::Uint3;
use std::cell::{Cell, RefCell, UnsafeCell};
use std::ffi::CString;
use std::ops::{Bound, Deref, DerefMut, RangeBounds};
use std::sync::atomic::AtomicUsize;
// use self::math::Uint3;
pub mod math;
pub mod poly;
pub mod printer;
pub mod swizzle;
pub mod traits;

pub use math::*;
pub use poly::*;
pub use printer::*;

pub(crate) static KERNEL_ID: AtomicUsize = AtomicUsize::new(0);
// prevent node being shared across kernels
// TODO: replace NodeRef with SafeNodeRef
#[derive(Clone, Copy, Debug)]
pub(crate) struct SafeNodeRef {
    pub(crate) node: NodeRef,
    pub(crate) kernel_id: usize,
}
pub trait Value: Copy + ir::TypeOf + 'static {
    type Expr: ExprProxy<Value = Self>;
    type Var: VarProxy<Value = Self>;
    fn fields() -> Vec<String>;
}

pub trait StructInitiaizable: Value {
    type Init: Into<Self::Expr>;
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

pub trait ToNode {
    fn node(&self) -> NodeRef;
}

pub trait FromNode: ToNode {
    fn from_node(node: NodeRef) -> Self;
}

fn _store<T1: Aggregate, T2: Aggregate>(var: &T1, value: &T2) {
    let value_nodes = value.to_vec_nodes();
    let self_nodes = var.to_vec_nodes();
    assert_eq!(value_nodes.len(), self_nodes.len());
    __current_scope(|b| {
        for (value_node, self_node) in value_nodes.into_iter().zip(self_nodes.into_iter()) {
            b.store(self_node, value_node);
        }
    })
}

#[inline(always)]
pub fn __new_user_node<T: UserNodeData>(data: T) -> NodeRef {
    use luisa_compute_ir::ir::new_user_node;
    new_user_node(__module_pools(), data)
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

pub unsafe trait Mask: ToNode {}
pub trait IntoIndex {
    fn to_u64(&self) -> Expr<u64>;
}
impl IntoIndex for i32 {
    fn to_u64(&self) -> Expr<u64> {
        const_(*self as u64)
    }
}
impl IntoIndex for i64 {
    fn to_u64(&self) -> Expr<u64> {
        const_(*self as u64)
    }
}
impl IntoIndex for u32 {
    fn to_u64(&self) -> Expr<u64> {
        const_(*self as u64)
    }
}
impl IntoIndex for u64 {
    fn to_u64(&self) -> Expr<u64> {
        const_(*self)
    }
}
impl IntoIndex for PrimExpr<u32> {
    fn to_u64(&self) -> Expr<u64> {
        self.ulong()
    }
}
impl IntoIndex for PrimExpr<u64> {
    fn to_u64(&self) -> Expr<u64> {
        *self
    }
}

pub trait IndexRead: ToNode {
    type Element: Value;
    fn read<I: IntoIndex>(&self, i: I) -> Expr<Self::Element>;
}

pub trait IndexWrite: IndexRead {
    fn write<I: IntoIndex, V: Into<Expr<Self::Element>>>(&self, i: I, value: V);
}

pub fn select<A: Aggregate>(mask: impl Mask, a: A, b: A) -> A {
    let a_nodes = a.to_vec_nodes();
    let b_nodes = b.to_vec_nodes();
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
                    &[mask.node(), a_node, b_node],
                    a_node.type_().clone(),
                ));
            }
        }
    });
    A::from_vec_nodes(ret)
}

impl ToNode for bool {
    fn node(&self) -> NodeRef {
        const_(*self).node()
    }
}

unsafe impl Mask for bool {}

unsafe impl Mask for Bool {}

pub trait ExprProxy: Copy + Aggregate + FromNode {
    type Value: Value;
}

pub struct VarDerefProxy<P, T: Value>
where
    P: VarProxy<Value = T>,
{
    pub(crate) var: P,
    pub(crate) dirty: Cell<bool>,
    pub(crate) assigned: Expr<T>,
    pub(crate) _phantom: std::marker::PhantomData<T>,
}

impl<P, T: Value> Deref for VarDerefProxy<P, T>
where
    P: VarProxy<Value = T>,
{
    type Target = Expr<T>;
    fn deref(&self) -> &Self::Target {
        &self.assigned
    }
}

impl<P, T: Value> DerefMut for VarDerefProxy<P, T>
where
    P: VarProxy<Value = T>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.dirty.set(true);
        &mut self.assigned
    }
}

impl<P, T: Value> Drop for VarDerefProxy<P, T>
where
    P: VarProxy<Value = T>,
{
    fn drop(&mut self) {
        if self.dirty.get() {
            self.var.store(self.assigned)
        }
    }
}
macro_rules! impl_assign_ops {
    ($ass:ident, $ass_m:ident, $o:ident, $o_m:ident) => {
        impl<P, T: Value, Rhs> std::ops::$ass<Rhs> for VarDerefProxy<P, T>
        where
            P: VarProxy<Value = T>,
            <T as lang::Value>::Expr: std::ops::$o<Rhs, Output = <T as lang::Value>::Expr>,
        {
            fn $ass_m(&mut self, rhs: Rhs) {
                *self.deref_mut() = std::ops::$o::$o_m(**self, rhs);
            }
        }
    };
}
impl_assign_ops!(AddAssign, add_assign, Add, add);
impl_assign_ops!(SubAssign, sub_assign, Sub, sub);
impl_assign_ops!(MulAssign, mul_assign, Mul, mul);
impl_assign_ops!(DivAssign, div_assign, Div, div);
impl_assign_ops!(RemAssign, rem_assign, Rem, rem);
impl_assign_ops!(BitAndAssign, bitand_assign, BitAnd, bitand);
impl_assign_ops!(BitOrAssign, bitor_assign, BitOr, bitor);
impl_assign_ops!(BitXorAssign, bitxor_assign, BitXor, bitxor);
impl_assign_ops!(ShlAssign, shl_assign, Shl, shl);
impl_assign_ops!(ShrAssign, shr_assign, Shr, shr);

pub trait VarProxy: Copy + Aggregate + FromNode {
    type Value: Value;
    fn store<U: Into<Expr<Self::Value>>>(&self, value: U) {
        let value = value.into();
        _store(self, &value);
    }
    fn load(&self) -> Expr<Self::Value> {
        __current_scope(|b| {
            let nodes = self.to_vec_nodes();
            let mut ret = vec![];
            for node in nodes {
                ret.push(b.call(Func::Load, &[node], node.type_().clone()));
            }
            Expr::<Self::Value>::from_nodes(&mut ret.into_iter())
        })
    }
    fn get_mut(&self) -> VarDerefProxy<Self, Self::Value> {
        VarDerefProxy {
            var: *self,
            dirty: Cell::new(false),
            assigned: self.load(),
            _phantom: std::marker::PhantomData,
        }
    }
    fn _deref<'a>(&'a self) -> &'a Expr<Self::Value> {
        RECORDER.with(|r| {
            let v: Expr<Self::Value> = self.load();
            let r = r.borrow();
            let v: &Expr<Self::Value> = r.arena.alloc(v);
            unsafe {
                let v: &'a Expr<Self::Value> = std::mem::transmute(v);
                v
            }
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
#[macro_export]
macro_rules! impl_callable_param {
    ($t:ty, $e:ty, $v:ty) => {
        impl CallableParameter for $e {
            fn def_param(_: Option<std::rc::Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self {
                builder.value::<$t>()
            }
            fn encode(&self, encoder: &mut CallableArgEncoder) {
                encoder.var(*self)
            }
        }
        impl CallableParameter for $v {
            fn def_param(_: Option<std::rc::Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self {
                builder.var::<$t>()
            }
            fn encode(&self, encoder: &mut CallableArgEncoder) {
                encoder.var(*self)
            }
        }
    };
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
        }
        impl ToNode for PrimVar<$t> {
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        impl ExprProxy for PrimExpr<$t> {
            type Value = $t;
        }
        impl VarProxy for PrimVar<$t> {
            type Value = $t;
        }
        impl Deref for PrimVar<$t> {
            type Target = PrimExpr<$t>;
            fn deref(&self) -> &Self::Target {
                self._deref()
            }
        }
        impl Value for $t {
            type Expr = PrimExpr<$t>;
            type Var = PrimVar<$t>;
            fn fields() -> Vec<String> {
                vec![]
            }
        }
        impl_callable_param!($t, PrimExpr<$t>, PrimVar<$t>);
    };
}

impl_prim!(bool);
impl_prim!(u32);
impl_prim!(u64);
impl_prim!(i32);
impl_prim!(i64);
impl_prim!(i16);
impl_prim!(u16);
impl_prim!(f16);
impl_prim!(f32);
impl_prim!(f64);

pub type Bool = PrimExpr<bool>;
pub type F16 = PrimExpr<f16>;
pub type F32 = PrimExpr<f32>;
pub type F64 = PrimExpr<f64>;
pub type I16 = PrimExpr<i16>;
pub type I32 = PrimExpr<i32>;
pub type I64 = PrimExpr<i64>;
pub type U16 = PrimExpr<u16>;
pub type U32 = PrimExpr<u32>;
pub type U64 = PrimExpr<u64>;

pub type F16Var = PrimVar<f16>;
pub type F32Var = PrimVar<f32>;
pub type F64Var = PrimVar<f64>;
pub type I16Var = PrimVar<i16>;
pub type I32Var = PrimVar<i32>;
pub type I64Var = PrimVar<i64>;
pub type U16Var = PrimVar<u16>;
pub type U32Var = PrimVar<u32>;
pub type U64Var = PrimVar<u64>;

pub type Half = PrimExpr<f16>;
pub type Float = PrimExpr<f32>;
pub type Double = PrimExpr<f64>;
pub type Int = PrimExpr<i32>;
pub type Long = PrimExpr<i64>;
pub type Uint = PrimExpr<u32>;
pub type Ulong = PrimExpr<u64>;
pub type Short = PrimExpr<i16>;
pub type Ushort = PrimExpr<u16>;

pub type BoolVar = PrimVar<bool>;
pub type HalfVar = PrimVar<f16>;
pub type FloatVar = PrimVar<f32>;
pub type DoubleVar = PrimVar<f64>;
pub type IntVar = PrimVar<i32>;
pub type LongVar = PrimVar<i64>;
pub type UintVar = PrimVar<u32>;
pub type UlongVar = PrimVar<u64>;
pub type ShortVar = PrimVar<i16>;
pub type UshortVar = PrimVar<u16>;

pub struct CpuFn<T: Value> {
    op: CArc<CpuCustomOp>,
    _marker: std::marker::PhantomData<T>,
}
#[macro_export]
macro_rules! cpu_dbg {
    ($arg:expr) => {{
        __cpu_dbg($arg, file!(), line!())
    }};
}
#[macro_export]
macro_rules! lc_dbg {
    ($arg:expr) => {{
        __cpu_dbg($arg, file!(), line!())
    }};
}
#[macro_export]
macro_rules! lc_unreachable {
    () => {
        __unreachable(file!(), line!(), column!())
    };
}
#[macro_export]
macro_rules! lc_assert {
    ($arg:expr) => {
        __assert($arg, stringify!($arg), file!(), line!(), column!())
    };
    ($arg:expr, $msg:expr) => {
        __assert($arg, $msg, file!(), line!(), column!())
    };
}
pub fn __cpu_dbg<T: ExprProxy>(arg: T, file: &'static str, line: u32)
where
    T::Value: Debug,
{
    if !is_cpu_backend() {
        return;
    }
    let f = CpuFn::new(move |x: &mut T::Value| {
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
            op: CArc::new(op),
            _marker: std::marker::PhantomData,
        }
    }
    pub fn call(&self, arg: impl ExprProxy<Value = T>) -> Expr<T> {
        RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            assert!(r.lock);
            assert_eq!(
                r.device
                    .as_ref()
                    .unwrap()
                    .inner
                    .query("device_name")
                    .unwrap(),
                "cpu",
                "CpuFn can only be used in cpu backend"
            );
            let addr = CArc::as_ptr(&self.op) as u64;
            if let Some((_, op)) = r.cpu_custom_ops.get(&addr) {
                assert_eq!(CArc::as_ptr(op), CArc::as_ptr(&self.op));
            } else {
                let i = r.cpu_custom_ops.len();
                r.cpu_custom_ops.insert(addr, (i, self.op.clone()));
            }
        });
        Expr::<T>::from_node(__current_scope(|b| {
            b.call(
                Func::CpuCustomOp(self.op.clone()),
                &[arg.node()],
                T::type_(),
            )
        }))
    }
}

pub(crate) struct Recorder {
    pub(crate) scopes: Vec<IrBuilder>,
    pub(crate) kernel_id: Option<usize>,
    pub(crate) lock: bool,
    pub(crate) captured_buffer: IndexMap<Binding, (usize, NodeRef, Binding, Arc<dyn Any>)>,
    pub(crate) cpu_custom_ops: IndexMap<u64, (usize, CArc<CpuCustomOp>)>,
    pub(crate) callables: IndexMap<u64, CallableModuleRef>,
    pub(crate) shared: Vec<NodeRef>,
    pub(crate) device: Option<Device>,
    pub(crate) block_size: Option<[u32; 3]>,
    pub(crate) building_kernel: bool,
    pub(crate) pools: Option<CArc<ModulePools>>,
    pub(crate) arena: Bump,
}

impl Recorder {
    fn reset(&mut self) {
        self.scopes.clear();
        self.captured_buffer.clear();
        self.cpu_custom_ops.clear();
        self.callables.clear();
        self.lock = false;
        self.device = None;
        self.block_size = None;
        self.arena.reset();
        self.shared.clear();
        self.kernel_id = None;
    }
    pub(crate) fn new() -> Self {
        Recorder {
            scopes: vec![],
            lock: false,
            captured_buffer: IndexMap::new(),
            cpu_custom_ops: IndexMap::new(),
            callables: IndexMap::new(),
            shared: vec![],
            device: None,
            block_size: None,
            pools: None,
            arena: Bump::new(),
            building_kernel: false,
            kernel_id: None,
        }
    }
}
thread_local! {
    pub(crate) static RECORDER: RefCell<Recorder> = RefCell::new(Recorder::new());
}

// Don't call this function directly unless you know what you are doing
pub fn __current_scope<F: FnOnce(&mut IrBuilder) -> R, R>(f: F) -> R {
    RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        assert!(r.lock, "__current_scope must be called within a kernel");
        let s = &mut r.scopes;
        f(s.last_mut().unwrap())
    })
}

pub(crate) fn __invoke_callable(callable: &CallableModuleRef, args: &[NodeRef]) -> NodeRef {
    RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let id = CArc::as_ptr(&callable.0) as u64;
        if let Some(c) = r.callables.get(&id) {
            assert_eq!(CArc::as_ptr(&c.0), CArc::as_ptr(&callable.0));
        } else {
            r.callables.insert(id, callable.clone());
        }
    });
    __current_scope(|b| {
        b.call(
            Func::Callable(callable.clone()),
            args,
            callable.0.ret_type.clone(),
        )
    })
}

pub(crate) fn __check_node_type(a: NodeRef, b: NodeRef) -> bool {
    if !ir::context::is_type_equal(a.type_(), b.type_()) {
        return false;
    }
    match (a.get().instruction.as_ref(), b.get().instruction.as_ref()) {
        (Instruction::Buffer, Instruction::Buffer) => true,
        (Instruction::Texture2D, Instruction::Texture2D) => true,
        (Instruction::Texture3D, Instruction::Texture3D) => true,
        (Instruction::Bindless, Instruction::Bindless) => true,
        (Instruction::Accel, Instruction::Accel) => true,
        (Instruction::Uniform, Instruction::Uniform) => true,
        (Instruction::Local { .. }, Instruction::Local { .. }) => true,
        (Instruction::Argument { by_value: true }, _) => b.get().instruction.has_value(),
        (Instruction::Argument { by_value: false }, _) => b.is_lvalue(),
        _ => false,
    }
}

pub(crate) fn __check_callable(callable: &CallableModuleRef, args: &[NodeRef]) -> bool {
    assert_eq!(callable.0.args.len(), args.len());
    for i in 0..args.len() {
        if !__check_node_type(callable.0.args[i], args[i]) {
            return false;
        }
    }
    true
}

// Don't call this function directly unless you know what you are doing
pub fn __pop_scope() -> Pooled<BasicBlock> {
    RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        let s = &mut r.scopes;
        s.pop().unwrap().finish()
    })
}

pub fn __module_pools() -> &'static CArc<ModulePools> {
    RECORDER.with(|r| {
        let r = r.borrow();
        assert!(r.lock, "__module_pools must be called within a kernel");
        let pool = r.pools.as_ref().unwrap();
        unsafe { std::mem::transmute(pool) }
    })
}
// pub fn __load<T: Value>(node: NodeRef) -> Expr<T> {
//     __current_scope(|b| {
//         let node = b.load(node);
//         Expr::<T>::from_node(node)
//     })
// }
// pub fn __store(var:NodeRef, value:NodeRef) {
//     let inst = &var.get().instruction;
// }

pub fn __extract<T: Value>(node: NodeRef, index: usize) -> NodeRef {
    let inst = &node.get().instruction;
    __current_scope(|b| {
        let i = b.const_(Const::Int32(index as i32));
        let op = match inst.as_ref() {
            Instruction::Local { .. } => Func::GetElementPtr,
            Instruction::Argument { by_value } => {
                if *by_value {
                    Func::ExtractElement
                } else {
                    Func::GetElementPtr
                }
            }
            Instruction::Call(f, args) => match f {
                Func::AtomicRef => {
                    let mut indices = args.to_vec();
                    indices.push(i);
                    return b.call(Func::AtomicRef, &indices, <T as TypeOf>::type_());
                }
                _ => Func::ExtractElement,
            },
            _ => Func::ExtractElement,
        };
        let node = b.call(op, &[node, i], <T as TypeOf>::type_());
        node
    })
}

pub fn __insert<T: Value>(node: NodeRef, index: usize, value: NodeRef) -> NodeRef {
    let inst = &node.get().instruction;
    __current_scope(|b| {
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
            __current_scope(|b| b.call(Func::Struct, nodes, <T as TypeOf>::type_()))
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
            __current_scope(|b| b.call(func, nodes, <T as TypeOf>::type_()))
        }
        Type::Matrix(vt) => {
            let length = vt.dimension;
            let func = match length {
                2 => Func::Mat2,
                3 => Func::Mat3,
                4 => Func::Mat4,
                _ => panic!("Can't compose vector with length {}", length),
            };
            __current_scope(|b| b.call(func, nodes, <T as TypeOf>::type_()))
        }
        _ => todo!(),
    }
}
#[macro_export]
macro_rules! struct_ {
    ($t:ty { $($it:ident : $value:expr), *  $(,)?}) =>{
        {
            type Init = <$t as $crate::lang::StructInitiaizable>::Init;
            let init = Init { $($it : $value), *  };
            type Expr = <$t as $crate::lang::Value>::Expr;
            let e:Expr = init.into();
            e
        }
    }
}
#[macro_export]
macro_rules! var {
    ($t:ty) => {
        local_zeroed::<$t>()
    };
    ($t:ty, 0) => {
        local_zeroed::<$t>()
    };
    ($t:ty, $init:expr) => {
        local::<$t>($init.into())
    };
    ($e:expr) => {
        def($e)
    };
}
pub fn def<E: ExprProxy<Value = T>, T: Value>(init: E) -> Var<T> {
    Var::<T>::from_node(__current_scope(|b| b.local(init.node())))
}
pub fn local<T: Value>(init: Expr<T>) -> Var<T> {
    Var::<T>::from_node(__current_scope(|b| b.local(init.node())))
}

pub fn local_zeroed<T: Value>() -> Var<T> {
    Var::<T>::from_node(__current_scope(|b| {
        b.local_zero_init(<T as TypeOf>::type_())
    }))
}

pub fn thread_id() -> Expr<Uint3> {
    Expr::<Uint3>::from_node(__current_scope(|b| {
        b.call(Func::ThreadId, &[], Uint3::type_())
    }))
}

pub fn block_id() -> Expr<Uint3> {
    Expr::<Uint3>::from_node(__current_scope(|b| {
        b.call(Func::BlockId, &[], Uint3::type_())
    }))
}

pub fn dispatch_id() -> Expr<Uint3> {
    Expr::<Uint3>::from_node(__current_scope(|b| {
        b.call(Func::DispatchId, &[], Uint3::type_())
    }))
}

pub fn dispatch_size() -> Expr<Uint3> {
    Expr::<Uint3>::from_node(__current_scope(|b| {
        b.call(Func::DispatchSize, &[], Uint3::type_())
    }))
}
fn check_block_size_for_cpu() {
    RECORDER.with(|r| {
        let r = r.borrow();
        assert!(
            r.block_size.is_some(),
            "CPU backend only support block operations on block size 1"
        );
        let size = r.block_size.unwrap();
        assert_eq!(
            size,
            [1, 1, 1],
            "CPU backend only support block operations on block size 1"
        );
    });
}
pub fn sync_block() {
    if is_cpu_backend() {
        check_block_size_for_cpu();
        return;
    }
    __current_scope(|b| {
        b.call(Func::SynchronizeBlock, &[], Type::void());
    })
}

pub fn warp_is_first_active_lane() -> Expr<bool> {
    Expr::<bool>::from_node(__current_scope(|b| {
        b.call(Func::WarpIsFirstActiveLane, &[], Bool::type_())
    }))
}
pub fn warp_active_all_equal(v: impl ScalarOrVector) -> Expr<bool> {
    Expr::<bool>::from_node(__current_scope(|b| {
        b.call(
            Func::WarpActiveAllEqual,
            &[v.node()],
            <bool as TypeOf>::type_(),
        )
    }))
}
pub fn warp_active_bit_and<T: ScalarOrVector<Element = E>, E: IntVarTrait>(v: T) -> T {
    T::from_node(__current_scope(|b| {
        b.call(
            Func::WarpActiveBitAnd,
            &[v.node()],
            <bool as TypeOf>::type_(),
        )
    }))
}

pub fn warp_active_bit_or<T: ScalarOrVector<Element = E>, E: IntVarTrait>(v: T) -> T {
    T::from_node(__current_scope(|b| {
        b.call(
            Func::WarpActiveBitOr,
            &[v.node()],
            <bool as TypeOf>::type_(),
        )
    }))
}

pub fn warp_active_bit_xor<T: ScalarOrVector<Element = E>, E: IntVarTrait>(v: T) -> T {
    T::from_node(__current_scope(|b| {
        b.call(
            Func::WarpActiveBitXor,
            &[v.node()],
            <bool as TypeOf>::type_(),
        )
    }))
}

pub fn warp_active_count_bits(v: impl Into<Expr<bool>>) -> Expr<u32> {
    Expr::<u32>::from_node(__current_scope(|b| {
        b.call(
            Func::WarpActiveCountBits,
            &[v.into().node()],
            <u32 as TypeOf>::type_(),
        )
    }))
}
pub fn warp_active_max<T: ScalarOrVector>(v: T) -> T::Element {
    <T::Element>::from_node(__current_scope(|b| {
        b.call(Func::WarpActiveMax, &[v.node()], <T::ElementHost>::type_())
    }))
}
pub fn warp_active_min<T: ScalarOrVector>(v: T) -> T::Element {
    <T::Element>::from_node(__current_scope(|b| {
        b.call(Func::WarpActiveMin, &[v.node()], <T::ElementHost>::type_())
    }))
}
pub fn warp_active_product<T: ScalarOrVector>(v: T) -> T::Element {
    <T::Element>::from_node(__current_scope(|b| {
        b.call(
            Func::WarpActiveProduct,
            &[v.node()],
            <T::ElementHost>::type_(),
        )
    }))
}
pub fn warp_active_sum<T: ScalarOrVector>(v: T) -> T::Element {
    <T::Element>::from_node(__current_scope(|b| {
        b.call(Func::WarpActiveSum, &[v.node()], <T::ElementHost>::type_())
    }))
}
pub fn warp_active_all(v: Expr<bool>) -> Expr<bool> {
    Expr::<bool>::from_node(__current_scope(|b| {
        b.call(Func::WarpActiveAll, &[v.node()], <bool as TypeOf>::type_())
    }))
}
pub fn warp_active_any(v: Expr<bool>) -> Expr<bool> {
    Expr::<bool>::from_node(__current_scope(|b| {
        b.call(Func::WarpActiveAny, &[v.node()], <bool as TypeOf>::type_())
    }))
}
pub fn warp_active_bit_mask() -> Expr<Uint4> {
    Expr::<Uint4>::from_node(__current_scope(|b| {
        b.call(Func::WarpActiveBitMask, &[], <Uint4 as TypeOf>::type_())
    }))
}
pub fn warp_prefix_count_bits(v: Expr<bool>) -> Expr<u32> {
    Expr::<u32>::from_node(__current_scope(|b| {
        b.call(
            Func::WarpPrefixCountBits,
            &[v.node()],
            <u32 as TypeOf>::type_(),
        )
    }))
}
pub fn warp_prefix_sum_exclusive<T: ScalarOrVector>(v: T) -> T {
    T::from_node(__current_scope(|b| {
        b.call(Func::WarpPrefixSum, &[v.node()], v.node().type_().clone())
    }))
}
pub fn warp_prefix_product_exclusive<T: ScalarOrVector>(v: T) -> T {
    T::from_node(__current_scope(|b| {
        b.call(
            Func::WarpPrefixProduct,
            &[v.node()],
            v.node().type_().clone(),
        )
    }))
}
pub fn warp_read_lane_at<T: BuiltinVarTrait>(v: T, index: impl Into<Expr<u32>>) -> T {
    let index = index.into();
    T::from_node(__current_scope(|b| {
        b.call(
            Func::WarpReadLaneAt,
            &[v.node(), index.node()],
            v.node().type_().clone(),
        )
    }))
}
pub fn warp_read_first_active_lane<T: BuiltinVarTrait>(v: T) -> T {
    T::from_node(__current_scope(|b| {
        b.call(
            Func::WarpReadFirstLane,
            &[v.node()],
            v.node().type_().clone(),
        )
    }))
}
pub fn set_block_size(size: [u32; 3]) {
    RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        assert!(
            r.building_kernel,
            "set_block_size cannot be called in callable!"
        );
        assert!(r.block_size.is_none(), "Block size already set");

        r.block_size = Some(size);
    });
}

pub fn block_size() -> Expr<Uint3> {
    RECORDER.with(|r| {
        let r = r.borrow();
        let s = r.block_size.unwrap_or_else(|| panic!("Block size not set"));
        const_::<Uint3>(Uint3::new(s[0], s[1], s[2]))
    })
}

pub type Expr<T> = <T as Value>::Expr;
pub type Var<T> = <T as Value>::Var;

pub fn zeroed<T: Value>() -> T::Expr {
    FromNode::from_node(__current_scope(|b| b.zero_initializer(T::type_())))
}

pub fn const_<T: Value + Copy + 'static>(value: T) -> T::Expr {
    let node = __current_scope(|s| -> NodeRef {
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
    Expr::<To>::from_node(__current_scope(|b| {
        b.call(Func::Bitcast, &[expr.node()], <To as TypeOf>::type_())
    }))
}

pub const fn packed_size<T: Value>() -> usize {
    (std::mem::size_of::<T>() + 3) / 4
}

pub fn pack_to<E, B>(expr: E, buffer: &B, index: impl Into<Expr<u32>>)
where
    E: ExprProxy,
    B: IndexWrite<Element = u32>,
{
    let index = index.into();
    __current_scope(|b| {
        b.call(
            Func::Pack,
            &[expr.node(), buffer.node(), index.node()],
            Type::void(),
        );
    });
}

pub fn unpack_from<T>(
    buffer: &impl IndexWrite<Element = u32>,
    index: impl Into<Expr<u32>>,
) -> Expr<T>
where
    T: Value,
{
    let index = index.into();
    Expr::<T>::from_node(__current_scope(|b| {
        b.call(
            Func::Unpack,
            &[buffer.node(), index.node()],
            <T as TypeOf>::type_(),
        )
    }))
}

impl<T: Value + TypeOf, const N: usize> Value for [T; N] {
    type Expr = ArrayExpr<T, N>;
    type Var = ArrayVar<T, N>;
    fn fields() -> Vec<String> {
        todo!("why this method exists?")
    }
}

#[derive(Clone, Copy)]
pub struct DynExpr {
    node: NodeRef,
}

impl<T: ExprProxy> From<T> for DynExpr {
    fn from(value: T) -> Self {
        Self { node: value.node() }
    }
}

impl<T: VarProxy> From<T> for DynVar {
    fn from(value: T) -> Self {
        Self { node: value.node() }
    }
}

impl DynExpr {
    pub fn downcast<T: Value>(&self) -> Option<Expr<T>> {
        if ir::context::is_type_equal(self.node.type_(), &T::type_()) {
            Some(Expr::<T>::from_node(self.node))
        } else {
            None
        }
    }
    pub fn get<T: Value>(&self) -> Expr<T> {
        self.downcast::<T>().unwrap_or_else(|| {
            panic!(
                "DynExpr::get: type mismatch: expected {}, got {}",
                std::any::type_name::<T>(),
                self.node.type_().to_string()
            )
        })
    }
    pub fn downcast_array<T: Value>(&self, len: usize) -> Option<VLArrayExpr<T>> {
        let array_type = ir::context::register_type(Type::Array(ArrayType {
            element: T::type_(),
            length: len,
        }));
        if ir::context::is_type_equal(self.node.type_(), &array_type) {
            Some(VLArrayExpr::<T>::from_node(self.node))
        } else {
            None
        }
    }
    pub fn get_array<T: Value>(&self, len: usize) -> VLArrayExpr<T> {
        let array_type = ir::context::register_type(Type::Array(ArrayType {
            element: T::type_(),
            length: len,
        }));
        self.downcast_array::<T>(len).unwrap_or_else(|| {
            panic!(
                "DynExpr::get: type mismatch: expected {}, got {}",
                array_type,
                self.node.type_().to_string()
            )
        })
    }
    pub fn new<E: ExprProxy>(expr: E) -> Self {
        Self { node: expr.node() }
    }
}

impl CallableParameter for DynExpr {
    fn def_param(arg: Option<Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self {
        let arg = arg.unwrap_or_else(|| panic!("DynExpr should be used in DynCallable only!"));
        let arg = arg.downcast_ref::<Self>().unwrap();
        let node = builder.arg(arg.node.type_().clone(), true);
        Self { node }
    }
    fn encode(&self, encoder: &mut CallableArgEncoder) {
        encoder.args.push(self.node)
    }
}

impl Aggregate for DynExpr {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        nodes.push(self.node)
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        Self {
            node: iter.next().unwrap(),
        }
    }
}

impl FromNode for DynExpr {
    fn from_node(node: NodeRef) -> Self {
        Self { node }
    }
}

impl ToNode for DynExpr {
    fn node(&self) -> NodeRef {
        self.node
    }
}

unsafe impl CallableRet for DynExpr {
    fn _return(&self) -> CArc<Type> {
        __current_scope(|b| {
            b.return_(self.node);
        });
        self.node.type_().clone()
    }
    fn _from_return(node: NodeRef) -> Self {
        Self::from_node(node)
    }
}

impl Aggregate for DynVar {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        nodes.push(self.node)
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        Self {
            node: iter.next().unwrap(),
        }
    }
}

impl FromNode for DynVar {
    fn from_node(node: NodeRef) -> Self {
        Self { node }
    }
}

impl ToNode for DynVar {
    fn node(&self) -> NodeRef {
        self.node
    }
}

#[derive(Clone, Copy)]
pub struct DynVar {
    node: NodeRef,
}

impl CallableParameter for DynVar {
    fn def_param(arg: Option<Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self {
        let arg = arg.unwrap_or_else(|| panic!("DynVar should be used in DynCallable only!"));
        let arg = arg.downcast_ref::<Self>().unwrap();
        let node = builder.arg(arg.node.type_().clone(), false);
        Self { node }
    }
    fn encode(&self, encoder: &mut CallableArgEncoder) {
        encoder.args.push(self.node)
    }
}

impl DynVar {
    pub fn downcast<T: Value>(&self) -> Option<Var<T>> {
        if ir::context::is_type_equal(self.node.type_(), &T::type_()) {
            Some(Var::<T>::from_node(self.node))
        } else {
            None
        }
    }
    pub fn get<T: Value>(&self) -> Var<T> {
        self.downcast::<T>().unwrap_or_else(|| {
            panic!(
                "DynVar::get: type mismatch: expected {}, got {}",
                std::any::type_name::<T>(),
                self.node.type_().to_string()
            )
        })
    }
    pub fn downcast_array<T: Value>(&self, len: usize) -> Option<VLArrayVar<T>> {
        let array_type = ir::context::register_type(Type::Array(ArrayType {
            element: T::type_(),
            length: len,
        }));
        if ir::context::is_type_equal(self.node.type_(), &array_type) {
            Some(VLArrayVar::<T>::from_node(self.node))
        } else {
            None
        }
    }
    pub fn get_array<T: Value>(&self, len: usize) -> VLArrayVar<T> {
        let array_type = ir::context::register_type(Type::Array(ArrayType {
            element: T::type_(),
            length: len,
        }));
        self.downcast_array::<T>(len).unwrap_or_else(|| {
            panic!(
                "DynExpr::get: type mismatch: expected {}, got {}",
                array_type,
                self.node.type_().to_string()
            )
        })
    }
    pub fn load(&self) -> DynExpr {
        DynExpr {
            node: __current_scope(|b| b.call(Func::Load, &[self.node], self.node.type_().clone())),
        }
    }
    pub fn store(&self, value: &DynExpr) {
        __current_scope(|b| b.update(self.node, value.node));
    }
    pub fn zero<T: Value>() -> Self {
        let v = local_zeroed::<T>();
        Self { node: v.node() }
    }
}
pub struct Shared<T: Value> {
    marker: std::marker::PhantomData<T>,
    node: NodeRef,
}
impl<T: Value> Shared<T> {
    pub fn new(length: usize) -> Self {
        Self {
            marker: std::marker::PhantomData,
            node: __current_scope(|b| {
                let shared = new_node(
                    b.pools(),
                    Node::new(
                        CArc::new(Instruction::Shared),
                        ir::context::register_type(Type::Array(ArrayType {
                            element: T::type_(),
                            length,
                        })),
                    ),
                );
                RECORDER.with(|r| {
                    let mut r = r.borrow_mut();
                    r.shared.push(shared);
                });
                shared
            }),
        }
    }
    pub fn len(&self) -> Expr<u64> {
        match self.node.type_().as_ref() {
            Type::Array(ArrayType { element: _, length }) => const_(*length as u64),
            _ => unreachable!(),
        }
    }
    pub fn static_len(&self) -> usize {
        match self.node.type_().as_ref() {
            Type::Array(ArrayType { element: _, length }) => *length,
            _ => unreachable!(),
        }
    }
    pub fn write<I: IntoIndex, V: Into<Expr<T>>>(&self, i: I, value: V) {
        let i = i.to_u64();
        let value = value.into();

        if need_runtime_check() {
            lc_assert!(i.cmplt(self.len()), "VLArrayVar::read out of bounds");
        }

        __current_scope(|b| {
            let gep = b.call(Func::GetElementPtr, &[self.node, i.node()], T::type_());
            b.update(gep, value.node());
        });
    }
    pub fn load(&self) -> VLArrayExpr<T> {
        VLArrayExpr::from_node(__current_scope(|b| {
            b.call(Func::Load, &[self.node], self.node.type_().clone())
        }))
    }
    pub fn store(&self, value: VLArrayExpr<T>) {
        __current_scope(|b| {
            b.update(self.node, value.node);
        });
    }
}
#[derive(Clone, Copy, Debug)]
pub struct VLArrayExpr<T: Value> {
    marker: std::marker::PhantomData<T>,
    node: NodeRef,
}

impl<T: Value> FromNode for VLArrayExpr<T> {
    fn from_node(node: NodeRef) -> Self {
        Self {
            marker: std::marker::PhantomData,
            node,
        }
    }
}

impl<T: Value> ToNode for VLArrayExpr<T> {
    fn node(&self) -> NodeRef {
        self.node
    }
}

impl<T: Value> Aggregate for VLArrayExpr<T> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        nodes.push(self.node);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        Self::from_node(iter.next().unwrap())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VLArrayVar<T: Value> {
    marker: std::marker::PhantomData<T>,
    node: NodeRef,
}

impl<T: Value> FromNode for VLArrayVar<T> {
    fn from_node(node: NodeRef) -> Self {
        Self {
            marker: std::marker::PhantomData,
            node,
        }
    }
}

impl<T: Value> ToNode for VLArrayVar<T> {
    fn node(&self) -> NodeRef {
        self.node
    }
}

impl<T: Value> Aggregate for VLArrayVar<T> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        nodes.push(self.node);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        Self::from_node(iter.next().unwrap())
    }
}

impl<T: Value> VLArrayVar<T> {
    pub fn read<I: Into<Expr<u32>>>(&self, i: I) -> Expr<T> {
        let i = i.into();
        if need_runtime_check() {
            lc_assert!(i.cmplt(self.len()), "VLArrayVar::read out of bounds");
        }

        Expr::<T>::from_node(__current_scope(|b| {
            let gep = b.call(Func::GetElementPtr, &[self.node, i.node()], T::type_());
            b.call(Func::Load, &[gep], T::type_())
        }))
    }
    pub fn len(&self) -> Expr<u32> {
        match self.node.type_().as_ref() {
            Type::Array(ArrayType { element: _, length }) => const_(*length as u32),
            _ => unreachable!(),
        }
    }
    pub fn static_len(&self) -> usize {
        match self.node.type_().as_ref() {
            Type::Array(ArrayType { element: _, length }) => *length,
            _ => unreachable!(),
        }
    }
    pub fn write<I: Into<Expr<u32>>, V: Into<Expr<T>>>(&self, i: I, value: V) {
        let i = i.into();
        let value = value.into();

        if need_runtime_check() {
            lc_assert!(i.cmplt(self.len()), "VLArrayVar::read out of bounds");
        }

        __current_scope(|b| {
            let gep = b.call(Func::GetElementPtr, &[self.node, i.node()], T::type_());
            b.update(gep, value.node());
        });
    }
    pub fn load(&self) -> VLArrayExpr<T> {
        VLArrayExpr::from_node(__current_scope(|b| {
            b.call(Func::Load, &[self.node], self.node.type_().clone())
        }))
    }
    pub fn store(&self, value: VLArrayExpr<T>) {
        __current_scope(|b| {
            b.update(self.node, value.node);
        });
    }
    pub fn zero(length: usize) -> Self {
        FromNode::from_node(__current_scope(|b| {
            b.local_zero_init(ir::context::register_type(Type::Array(ArrayType {
                element: T::type_(),
                length,
            })))
        }))
    }
}

impl<T: Value> VLArrayExpr<T> {
    pub fn zero(length: usize) -> Self {
        let node = __current_scope(|b| {
            b.call(
                Func::ZeroInitializer,
                &[],
                ir::context::register_type(Type::Array(ArrayType {
                    element: T::type_(),
                    length,
                })),
            )
        });
        Self::from_node(node)
    }
    pub fn static_len(&self) -> usize {
        match self.node.type_().as_ref() {
            Type::Array(ArrayType { element: _, length }) => *length,
            _ => unreachable!(),
        }
    }
    pub fn read<I: IntoIndex>(&self, i: I) -> Expr<T> {
        let i = i.to_u64();
        if need_runtime_check() {
            lc_assert!(i.cmplt(self.len()));
        }

        Expr::<T>::from_node(__current_scope(|b| {
            b.call(Func::ExtractElement, &[self.node, i.node()], T::type_())
        }))
    }
    pub fn len(&self) -> Expr<u64> {
        match self.node.type_().as_ref() {
            Type::Array(ArrayType { element: _, length }) => const_(*length as u64),
            _ => unreachable!(),
        }
    }
}

impl<T: Value, const N: usize> IndexRead for ArrayExpr<T, N> {
    type Element = T;
    fn read<I: IntoIndex>(&self, i: I) -> Expr<T> {
        let i = i.to_u64();

        lc_assert!(i.cmplt(const_(N as u64)));

        Expr::<T>::from_node(__current_scope(|b| {
            b.call(Func::ExtractElement, &[self.node, i.node()], T::type_())
        }))
    }
}

impl<T: Value, const N: usize> IndexRead for ArrayVar<T, N> {
    type Element = T;
    fn read<I: IntoIndex>(&self, i: I) -> Expr<T> {
        let i = i.to_u64();
        if need_runtime_check() {
            lc_assert!(i.cmplt(const_(N as u64)));
        }

        Expr::<T>::from_node(__current_scope(|b| {
            let gep = b.call(Func::GetElementPtr, &[self.node, i.node()], T::type_());
            b.call(Func::Load, &[gep], T::type_())
        }))
    }
}

impl<T: Value, const N: usize> IndexWrite for ArrayVar<T, N> {
    fn write<I: IntoIndex, V: Into<Expr<T>>>(&self, i: I, value: V) {
        let i = i.to_u64();
        let value = value.into();

        if need_runtime_check() {
            lc_assert!(i.cmplt(const_(N as u64)));
        }

        __current_scope(|b| {
            let gep = b.call(Func::GetElementPtr, &[self.node, i.node()], T::type_());
            b.update(gep, value.node());
        });
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
}

impl<T: Value, const N: usize> ToNode for ArrayExpr<T, N> {
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
}

impl<T: Value, const N: usize> ToNode for ArrayVar<T, N> {
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

impl<T: Value, const N: usize> ExprProxy for ArrayExpr<T, N> {
    type Value = [T; N];
}

impl<T: Value, const N: usize> VarProxy for ArrayVar<T, N> {
    type Value = [T; N];
}

impl<T: Value, const N: usize> ArrayVar<T, N> {
    pub fn len(&self) -> Expr<u32> {
        const_(N as u32)
    }
}

impl<T: Value, const N: usize> ArrayExpr<T, N> {
    pub fn zero() -> Self {
        let node = __current_scope(|b| b.call(Func::ZeroInitializer, &[], <[T; N]>::type_()));
        Self::from_node(node)
    }
    pub fn len(&self) -> Expr<u32> {
        const_(N as u32)
    }
}

// Not recommended to use this directly
pub struct KernelBuilder {
    device: Option<crate::runtime::Device>,
    args: Vec<NodeRef>,
}

pub trait CallableParameter: Sized + Clone + 'static {
    fn def_param(arg: Option<Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self;
    fn encode(&self, encoder: &mut CallableArgEncoder);
}
macro_rules! impl_callable_parameter_for_tuple {
    ()=>{
        impl CallableParameter for () {
            fn def_param(_: Option<Rc<dyn Any>>, _: &mut KernelBuilder) {}
            fn encode(&self, _: &mut CallableArgEncoder) { }
        }
    };
    ($first:ident  $($rest:ident) *) => {
        impl<$first:CallableParameter, $($rest: CallableParameter),*> CallableParameter for ($first, $($rest,)*) {
            #[allow(non_snake_case)]
            fn def_param(arg: Option<Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self {
                if let Some(arg) = arg {
                    let ($first, $($rest,)*) = arg.downcast_ref::<($first, $($rest,)*)>().cloned().unwrap();
                    let $first = $first::def_param(Some(std::rc::Rc::new($first)), builder);
                    let ($($rest,)*) = ($($rest::def_param(Some(std::rc::Rc::new($rest)), builder),)*);
                    ($first, $($rest,)*)
                }else {
                    let $first = $first::def_param(None, builder);
                    let ($($rest,)*) = ($($rest::def_param(None, builder),)*);
                    ($first, $($rest,)*)
                }
            }
            #[allow(non_snake_case)]
            fn encode(&self, encoder: &mut CallableArgEncoder) {
                let ($first, $($rest,)*) = self;
                $first.encode(encoder);
                $($rest.encode(encoder);)*
            }
        }
        impl_callable_parameter_for_tuple!($($rest)*);
    };

}
impl_callable_parameter_for_tuple!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);

impl<T: Value + 'static> CallableParameter for BufferVar<T> {
    fn def_param(_: Option<Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self {
        builder.buffer()
    }
    fn encode(&self, encoder: &mut CallableArgEncoder) {
        encoder.buffer(self)
    }
}
impl CallableParameter for ByteBufferVar {
    fn def_param(_: Option<Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self {
        builder.byte_buffer()
    }
    fn encode(&self, encoder: &mut CallableArgEncoder) {
        encoder.byte_buffer(self)
    }
}
impl<T: IoTexel + 'static> CallableParameter for Tex2dVar<T> {
    fn def_param(_: Option<Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self {
        builder.tex2d()
    }
    fn encode(&self, encoder: &mut CallableArgEncoder) {
        encoder.tex2d(self)
    }
}

impl<T: IoTexel + 'static> CallableParameter for Tex3dVar<T> {
    fn def_param(_: Option<Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self {
        builder.tex3d()
    }
    fn encode(&self, encoder: &mut CallableArgEncoder) {
        encoder.tex3d(self)
    }
}

impl CallableParameter for BindlessArrayVar {
    fn def_param(_: Option<Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self {
        builder.bindless_array()
    }
    fn encode(&self, encoder: &mut CallableArgEncoder) {
        encoder.bindless_array(self)
    }
}

impl CallableParameter for rtx::AccelVar {
    fn def_param(_: Option<Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self {
        builder.accel()
    }
    fn encode(&self, encoder: &mut CallableArgEncoder) {
        encoder.accel(self)
    }
}

pub trait KernelParameter {
    fn def_param(builder: &mut KernelBuilder) -> Self;
}

impl<T: Value, U> KernelParameter for U
where
    U: ExprProxy<Value = T>,
    T: Value<Expr = U>,
{
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.uniform::<T>()
    }
}
impl KernelParameter for ByteBufferVar {
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.byte_buffer()
    }
}
impl<T: Value> KernelParameter for BufferVar<T> {
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.buffer()
    }
}

impl<T: IoTexel> KernelParameter for Tex2dVar<T> {
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.tex2d()
    }
}

impl<T: IoTexel> KernelParameter for Tex3dVar<T> {
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.tex3d()
    }
}

impl KernelParameter for BindlessArrayVar {
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.bindless_array()
    }
}

impl KernelParameter for rtx::AccelVar {
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.accel()
    }
}
macro_rules! impl_kernel_param_for_tuple {
    ($first:ident  $($rest:ident)*) => {
        impl<$first:KernelParameter, $($rest: KernelParameter),*> KernelParameter for ($first, $($rest,)*) {
            #[allow(non_snake_case)]
            fn def_param(builder: &mut KernelBuilder) -> Self {
                ($first::def_param(builder), $($rest::def_param(builder)),*)
            }
        }
        impl_kernel_param_for_tuple!($($rest)*);
    };
    ()=>{
        impl KernelParameter for () {
            fn def_param(_: &mut KernelBuilder) -> Self {
            }
        }
    }
}
impl_kernel_param_for_tuple!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);
impl KernelBuilder {
    pub fn new(device: Option<crate::runtime::Device>, is_kernel: bool) -> Self {
        RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            assert!(!r.lock, "Cannot record multiple kernels at the same time");
            assert!(
                r.scopes.is_empty(),
                "Cannot record multiple kernels at the same time"
            );
            r.lock = true;
            r.device = device.clone();
            r.pools = Some(CArc::new(ModulePools::new()));
            r.scopes.clear();
            r.building_kernel = is_kernel;
            let pools = r.pools.clone().unwrap();
            r.scopes.push(IrBuilder::new(pools));
        });
        Self {
            device,
            args: vec![],
        }
    }
    pub(crate) fn arg(&mut self, ty: CArc<Type>, by_value: bool) -> NodeRef {
        let node = new_node(
            __module_pools(),
            Node::new(CArc::new(Instruction::Argument { by_value }), ty),
        );
        self.args.push(node);
        node
    }
    pub fn value<T: Value>(&mut self) -> Expr<T> {
        let node = self.arg(T::type_(), true);
        FromNode::from_node(node)
    }
    pub fn var<T: Value>(&mut self) -> Var<T> {
        let node = self.arg(T::type_(), false);
        FromNode::from_node(node)
    }
    pub fn uniform<T: Value>(&mut self) -> Expr<T> {
        let node = new_node(
            __module_pools(),
            Node::new(CArc::new(Instruction::Uniform), T::type_()),
        );
        self.args.push(node);
        FromNode::from_node(node)
    }
    pub fn byte_buffer(&mut self) -> ByteBufferVar {
        let node = new_node(
            __module_pools(),
            Node::new(CArc::new(Instruction::Buffer), Type::void()),
        );
        self.args.push(node);
        ByteBufferVar {
            node,
            handle: None,
        }
    }
    pub fn buffer<T: Value>(&mut self) -> BufferVar<T> {
        let node = new_node(
            __module_pools(),
            Node::new(CArc::new(Instruction::Buffer), T::type_()),
        );
        self.args.push(node);
        BufferVar {
            node,
            marker: std::marker::PhantomData,
            handle: None,
        }
    }
    pub fn tex2d<T: IoTexel>(&mut self) -> Tex2dVar<T> {
        let node = new_node(
            __module_pools(),
            Node::new(CArc::new(Instruction::Texture2D), T::type_()),
        );
        self.args.push(node);
        Tex2dVar {
            node,
            marker: std::marker::PhantomData,
            handle: None,
            level: None,
        }
    }
    pub fn tex3d<T: IoTexel>(&mut self) -> Tex3dVar<T> {
        let node = new_node(
            __module_pools(),
            Node::new(CArc::new(Instruction::Texture3D), T::type_()),
        );
        self.args.push(node);
        Tex3dVar {
            node,
            marker: std::marker::PhantomData,
            handle: None,
            level: None,
        }
    }
    pub fn bindless_array(&mut self) -> BindlessArrayVar {
        let node = new_node(
            __module_pools(),
            Node::new(CArc::new(Instruction::Bindless), Type::void()),
        );
        self.args.push(node);
        BindlessArrayVar { node, handle: None }
    }
    pub fn accel(&mut self) -> rtx::AccelVar {
        let node = new_node(
            __module_pools(),
            Node::new(CArc::new(Instruction::Accel), Type::void()),
        );
        self.args.push(node);
        rtx::AccelVar { node, handle: None }
    }
    fn collect_module_info(&self) -> (ResourceTracker, Vec<CArc<CpuCustomOp>>, Vec<Capture>) {
        RECORDER.with(|r| {
            let mut resource_tracker = ResourceTracker::new();
            let r = r.borrow_mut();
            let mut captured: Vec<Capture> = Vec::new();
            let mut captured_buffers: Vec<_> = r.captured_buffer.values().cloned().collect();
            captured_buffers.sort_by_key(|(i, _, _, _)| *i);
            for (j, (i, node, binding, handle)) in captured_buffers.into_iter().enumerate() {
                assert_eq!(j, i);
                captured.push(Capture { node, binding });
                resource_tracker.add_any(handle);
            }
            let mut cpu_custom_ops: Vec<_> = r.cpu_custom_ops.values().cloned().collect();
            cpu_custom_ops.sort_by_key(|(i, _)| *i);
            let mut cpu_custom_ops: Vec<CArc<CpuCustomOp>> = cpu_custom_ops
                .iter()
                .enumerate()
                .map(|(j, (i, op))| {
                    assert_eq!(j, *i);
                    (*op).clone()
                })
                .collect::<Vec<_>>();
            let callables: Vec<CallableModuleRef> = r.callables.values().cloned().collect();
            let mut captured_set = HashSet::<Capture>::new();
            let mut cpu_custom_ops_set = HashSet::<u64>::new();
            let mut callable_set = HashSet::<u64>::new();
            for capture in captured.iter() {
                captured_set.insert(*capture);
            }
            for op in &cpu_custom_ops {
                cpu_custom_ops_set.insert(CArc::as_ptr(op) as u64);
            }
            for c in &callables {
                callable_set.insert(CArc::as_ptr(&c.0) as u64);
                for capture in c.0.captures.as_ref() {
                    if !captured_set.contains(capture) {
                        captured_set.insert(*capture);
                        captured.push(*capture);
                    }
                }
                for op in c.0.cpu_custom_ops.as_ref() {
                    let id = CArc::as_ptr(op) as u64;
                    if !cpu_custom_ops_set.contains(&id) {
                        cpu_custom_ops_set.insert(id);
                        cpu_custom_ops.push(op.clone());
                    }
                }
            }
            (resource_tracker, cpu_custom_ops, captured)
        })
    }
    fn build_callable<R: CallableRet>(&mut self, body: impl FnOnce(&mut Self) -> R) -> RawCallable {
        let ret = body(self);
        let ret_type = ret._return();
        let (rt, cpu_custom_ops, captures) = self.collect_module_info();
        RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            assert!(r.lock);
            r.lock = false;
            assert_eq!(r.scopes.len(), 1);
            let scope = r.scopes.pop().unwrap();
            let entry = scope.finish();
            let ir_module = Module {
                entry,
                kind: ModuleKind::Kernel,
                pools: r.pools.clone().unwrap(),
            };
            let ir_module = {
                // perform IR passes
                let ad_transform = transform::autodiff::Autodiff;
                ad_transform.transform(ir_module)
            };
            let module = CallableModule {
                module: ir_module,
                ret_type,
                cpu_custom_ops: CBoxedSlice::new(cpu_custom_ops),
                captures: CBoxedSlice::new(captures),
                args: CBoxedSlice::new(self.args.clone()),
                pools: r.pools.clone().unwrap(),
            };
            let module = CallableModuleRef(CArc::new(module));
            r.reset();
            RawCallable {
                module,
                resource_tracker: rt,
            }
        })
    }
    fn build_kernel(
        &mut self,
        options: KernelBuildOptions,
        body: impl FnOnce(&mut Self),
    ) -> crate::runtime::RawKernel {
        body(self);
        let (rt, cpu_custom_ops, captures) = self.collect_module_info();
        RECORDER.with(|r| -> crate::runtime::RawKernel {
            let mut r = r.borrow_mut();
            assert!(r.lock);
            r.lock = false;
            assert_eq!(r.scopes.len(), 1);
            let scope = r.scopes.pop().unwrap();
            let entry = scope.finish();

            let ir_module = Module {
                entry,
                kind: ModuleKind::Kernel,
                pools: r.pools.clone().unwrap(),
            };
            let ir_module = {
                // perform IR passes
                let ad_transform = transform::autodiff::Autodiff;
                ad_transform.transform(ir_module)
            };
            let module = KernelModule {
                module: ir_module,
                cpu_custom_ops: CBoxedSlice::new(cpu_custom_ops),
                captures: CBoxedSlice::new(captures),
                shared: CBoxedSlice::new(r.shared.clone()),
                args: CBoxedSlice::new(self.args.clone()),
                block_size: r.block_size.unwrap_or([64, 1, 1]),
                pools: r.pools.clone().unwrap(),
            };

            let module = CArc::new(module);
            let name = options.name.unwrap_or("".to_string());
            let name = Arc::new(CString::new(name).unwrap());
            let shader_options = api::ShaderOption {
                enable_cache: options.enable_cache,
                enable_fast_math: options.enable_fast_math,
                enable_debug_info: options.enable_debug_info,
                compile_only: false,
                name: name.as_ptr(),
            };
            let artifact = if options.async_compile {
                ShaderArtifact::Async(AsyncShaderArtifact::new(
                    self.device.clone().unwrap(),
                    module.clone(),
                    shader_options,
                    name,
                ))
            } else {
                ShaderArtifact::Sync(
                    self.device
                        .as_ref()
                        .unwrap()
                        .inner
                        .create_shader(&module, &shader_options),
                )
            };
            //
            r.reset();
            RawKernel {
                artifact,
                device: self.device.clone().unwrap(),
                resource_tracker: rt,
                module,
            }
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelBuildOptions {
    pub enable_debug_info: bool,
    pub enable_optimization: bool,
    pub async_compile: bool,
    pub enable_cache: bool,
    pub enable_fast_math: bool,
    pub name: Option<String>,
}

impl Default for KernelBuildOptions {
    fn default() -> Self {
        Self {
            enable_debug_info: false,
            enable_optimization: true,
            async_compile: false,
            enable_cache: true,
            enable_fast_math: true,
            name: None,
        }
    }
}

pub trait KernelBuildFn {
    fn build_kernel(
        &self,
        builder: &mut KernelBuilder,
        options: KernelBuildOptions,
    ) -> crate::runtime::RawKernel;
}

pub trait CallableBuildFn {
    fn build_callable(&self, args: Option<Rc<dyn Any>>, builder: &mut KernelBuilder)
        -> RawCallable;
}

pub trait StaticCallableBuildFn: CallableBuildFn {}

// @FIXME: this looks redundant
pub unsafe trait CallableRet {
    fn _return(&self) -> CArc<Type>;
    fn _from_return(node: NodeRef) -> Self;
}

unsafe impl CallableRet for () {
    fn _return(&self) -> CArc<Type> {
        Type::void()
    }
    fn _from_return(_: NodeRef) -> Self {}
}

unsafe impl<T: ExprProxy> CallableRet for T {
    fn _return(&self) -> CArc<Type> {
        __current_scope(|b| {
            b.return_(self.node());
        });
        T::Value::type_()
    }
    fn _from_return(node: NodeRef) -> Self {
        Self::from_node(node)
    }
}

pub trait CallableSignature<'a, R: CallableRet> {
    type Callable;
    type DynCallable;
    type Fn: CallableBuildFn;
    type StaticFn: StaticCallableBuildFn;
    type DynFn: CallableBuildFn + 'static;
    fn wrap_raw_callable(callable: RawCallable) -> Self::Callable;
    fn create_dyn_callable(device: Device, init_once: bool, f: Self::DynFn) -> Self::DynCallable;
}

pub trait KernelSignature<'a> {
    type Fn: KernelBuildFn;
    type Kernel;

    fn wrap_raw_kernel(kernel: crate::runtime::RawKernel) -> Self::Kernel;
}
macro_rules! impl_callable_signature {
    ()=>{
        impl<'a, R: CallableRet +'static> CallableSignature<'a, R> for () {
            type Fn = &'a dyn Fn() ->R;
            type DynFn = Box<dyn Fn() ->R>;
            type StaticFn = fn() -> R;
            type Callable = Callable<(), R>;
            type DynCallable = DynCallable<(), R>;
            fn wrap_raw_callable(callable: RawCallable) -> Self::Callable{
                Callable {
                    inner: callable,
                    _marker:std::marker::PhantomData,
                }
            }
            fn create_dyn_callable(device:Device, init_once:bool, f: Self::DynFn) -> Self::DynCallable {
                DynCallable::new(device, init_once, Box::new(move |arg, builder| {
                    let raw_callable = CallableBuildFn::build_callable(&f, Some(arg), builder);
                    Self::wrap_raw_callable(raw_callable)
                }))
            }
        }
    };
    ($first:ident  $($rest:ident)*) => {
        impl<'a, R:CallableRet +'static, $first:CallableParameter +'static, $($rest: CallableParameter +'static),*> CallableSignature<'a, R> for ($first, $($rest,)*) {
            type Fn = &'a dyn Fn($first, $($rest),*)->R;
            type DynFn = Box<dyn Fn($first, $($rest),*)->R>;
            type Callable = Callable<($first, $($rest,)*), R>;
            type StaticFn = fn($first, $($rest,)*)->R;
            type DynCallable = DynCallable<($first, $($rest,)*), R>;
            fn wrap_raw_callable(callable: RawCallable) -> Self::Callable{
                Callable {
                    inner: callable,
                    _marker:std::marker::PhantomData,
                }
            }
            fn create_dyn_callable(device:Device, init_once:bool, f: Self::DynFn) -> Self::DynCallable {
                DynCallable::new(device, init_once, Box::new(move |arg, builder| {
                    let raw_callable = CallableBuildFn::build_callable(&f, Some(arg), builder);
                    Self::wrap_raw_callable(raw_callable)
                }))
            }
        }
        impl_callable_signature!($($rest)*);
    };
}
impl_callable_signature!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);
macro_rules! impl_kernel_signature {
    ()=>{
        impl<'a> KernelSignature<'a> for () {
            type Fn = &'a dyn Fn();
            type Kernel = Kernel<()>;
            fn wrap_raw_kernel(kernel: crate::runtime::RawKernel) -> Self::Kernel {
                Self::Kernel{
                    inner:kernel,
                    _marker:std::marker::PhantomData,
                }
            }
        }
    };
    ($first:ident  $($rest:ident)*) => {
        impl<'a, $first:KernelArg +'static, $($rest: KernelArg +'static),*> KernelSignature<'a> for ($first, $($rest,)*) {
            type Fn = &'a dyn Fn($first::Parameter, $($rest::Parameter),*);
            type Kernel = Kernel<($first, $($rest,)*)>;
            fn wrap_raw_kernel(kernel: crate::runtime::RawKernel) -> Self::Kernel {
                Self::Kernel{
                    inner:kernel,
                    _marker:std::marker::PhantomData,
                }
            }
        }
        impl_kernel_signature!($($rest)*);
    };
}
impl_kernel_signature!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);

macro_rules! impl_callable_build_for_fn {
    ()=>{
        impl<R:CallableRet +'static> CallableBuildFn for &dyn Fn()->R {
            fn build_callable(&self, _args: Option<Rc<dyn Any>>, builder: &mut KernelBuilder)->RawCallable {
                builder.build_callable( |_| {
                    self()
                })
            }
        }
        impl<R:CallableRet +'static> CallableBuildFn for fn()->R {
            fn build_callable(&self, _args: Option<Rc<dyn Any>>, builder: &mut KernelBuilder)->RawCallable {
                builder.build_callable( |_| {
                    self()
                })
            }
        }
        impl<R:CallableRet +'static> CallableBuildFn for Box<dyn Fn()->R> {
            fn build_callable(&self, _args: Option<Rc<dyn Any>>, builder: &mut KernelBuilder)->RawCallable {
                builder.build_callable( |_| {
                    self()
                })
            }
        }
        impl <R:CallableRet +'static> StaticCallableBuildFn  for fn()->R {}
    };
    ($first:ident  $($rest:ident)*) => {
        impl<R:CallableRet +'static, $first:CallableParameter, $($rest: CallableParameter),*> CallableBuildFn for &dyn Fn($first, $($rest,)*)->R {
            #[allow(non_snake_case)]
            fn build_callable(&self, args: Option<Rc<dyn Any>>, builder: &mut KernelBuilder)->RawCallable {
                builder.build_callable( |builder| {
                    if let Some(args) = args {
                        let ($first, $($rest,)*) = args.downcast_ref::<($first, $($rest,)*)>().cloned().unwrap();
                        let $first = $first::def_param(Some(Rc::new($first)), builder);
                        $(let $rest = $rest::def_param(Some(Rc::new($rest)), builder);)*
                        self($first, $($rest,)*)
                    } else {
                        let $first = $first::def_param(None, builder);
                        $(let $rest = $rest::def_param(None, builder);)*
                        self($first, $($rest,)*)
                    }
                })
            }
        }
        impl<R:CallableRet +'static, $first:CallableParameter, $($rest: CallableParameter),*> CallableBuildFn for Box<dyn Fn($first, $($rest,)*)->R> {
            #[allow(non_snake_case)]
            fn build_callable(&self, args: Option<Rc<dyn Any>>, builder: &mut KernelBuilder)->RawCallable {
                builder.build_callable( |builder| {
                    if let Some(args) = args {
                        let ($first, $($rest,)*) = args.downcast_ref::<($first, $($rest,)*)>().cloned().unwrap();
                        let $first = $first::def_param(Some(Rc::new($first)), builder);
                        $(let $rest = $rest::def_param(Some(Rc::new($rest)), builder);)*
                        self($first, $($rest,)*)
                    } else {
                        let $first = $first::def_param(None, builder);
                        $(let $rest = $rest::def_param(None, builder);)*
                        self($first, $($rest,)*)
                    }
                })
            }
        }
        impl<R:CallableRet +'static, $first:CallableParameter, $($rest: CallableParameter),*> CallableBuildFn for fn($first, $($rest,)*)->R {
            #[allow(non_snake_case)]
            fn build_callable(&self, args: Option<Rc<dyn Any>>, builder: &mut KernelBuilder)->RawCallable {
                builder.build_callable( |builder| {
                    if let Some(args) = args {
                        let ($first, $($rest,)*) = args.downcast_ref::<($first, $($rest,)*)>().cloned().unwrap();
                        let $first = $first::def_param(Some(Rc::new($first)), builder);
                        $(let $rest = $rest::def_param(Some(Rc::new($rest)), builder);)*
                        self($first, $($rest,)*)
                    } else {
                        let $first = $first::def_param(None, builder);
                        $(let $rest = $rest::def_param(None, builder);)*
                        self($first, $($rest,)*)
                    }
                })
            }
        }
        impl<R:CallableRet +'static, $first:CallableParameter, $($rest: CallableParameter),*> StaticCallableBuildFn for fn($first, $($rest,)*)->R {}
        impl_callable_build_for_fn!($($rest)*);
    };
}
impl_callable_build_for_fn!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);
macro_rules! impl_kernel_build_for_fn {
    ()=>{
        impl KernelBuildFn for &dyn Fn() {
            fn build_kernel(&self, builder: &mut KernelBuilder, options:KernelBuildOptions) -> crate::runtime::RawKernel {
                builder.build_kernel(options, |_| {
                    self()
                })
            }
        }
    };
    ($first:ident  $($rest:ident)*) => {
        impl<$first:KernelParameter, $($rest: KernelParameter),*> KernelBuildFn for &dyn Fn($first, $($rest,)*) {
            #[allow(non_snake_case)]
            fn build_kernel(&self, builder: &mut KernelBuilder, options:KernelBuildOptions) -> crate::runtime::RawKernel {
                builder.build_kernel(options, |builder| {
                    let $first = $first::def_param(builder);
                    $(let $rest = $rest::def_param(builder);)*
                    self($first, $($rest,)*)
                })
            }
        }
        impl_kernel_build_for_fn!($($rest)*);
    };
}
impl_kernel_build_for_fn!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);

pub fn if_then_else<R: Aggregate>(
    cond: impl Mask,
    then: impl Fn() -> R,
    else_: impl Fn() -> R,
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

pub fn generic_loop(cond: impl Fn() -> Bool, body: impl Fn(), update: impl Fn()) {
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

#[macro_export]
/**
 * If you want rustfmt to format your code, use if_!(cond, { .. }, { .. }) or if_!(cond, { .. }, else, {...})
 * instead of if_!(cond, { .. }, else {...}).
 *
 */
macro_rules! if_ {
    ($cond:expr, $then:block, else $else_:block) => {
        if_then_else($cond, || $then, || $else_)
    };
    ($cond:expr, $then:block, else, $else_:block) => {
        if_then_else($cond, || $then, || $else_)
    };
    ($cond:expr, $then:block, $else_:block) => {
        if_then_else($cond, || $then, || $else_)
    };
    ($cond:expr, $then:block) => {
        if_then_else($cond, || $then, || {})
    };
}
#[macro_export]
macro_rules! while_ {
    ($cond:expr,$body:block) => {
        generic_loop(|| $cond, || $body, || {})
    };
}
#[macro_export]
macro_rules! loop_ {
    ($body:block) => {
        while_!(const_(true), $body)
    };
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
                const_(*self.start()).node()
            }
            fn end(&self) -> NodeRef {
                const_(*self.end()).node()
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
                const_(self.start).node()
            }
            fn end(&self) -> NodeRef {
                const_(self.end).node()
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

#[inline]
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
                Bool::from_node(b.call(
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

// pub fn return_v<T: FromNode>(v: T) {
//     __current_scope(|b| {
//         b.return_(Some(v.node()));
//     });
// }
pub fn return_() {
    __current_scope(|b| {
        b.return_(INVALID_REF);
    });
}

struct AdContext {
    started: bool,
    backward_called: bool,
    // forward: Option<Pooled<BasicBlock>>,
}

impl AdContext {
    fn new() -> Self {
        Self {
            started: false,
            backward_called: false,
            // forward: None,
        }
    }
    fn reset(&mut self) {
        *self = Self::new();
    }
}
thread_local! {
    static AD_CONTEXT:RefCell<AdContext> = RefCell::new(AdContext::new());
}
pub fn requires_grad(var: impl ExprProxy) {
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

pub fn gradient<T: ExprProxy>(var: T) -> T {
    T::from_node(__current_scope(|b| {
        b.call(Func::Gradient, &[var.node()], var.node().type_().clone())
    }))
}

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
pub fn detach<T: FromNode>(v: T) -> T {
    let v = v.node();
    let node = __current_scope(|b| b.call(Func::Detach, &[v], v.type_().clone()));
    T::from_node(node)
}

pub fn autodiff(body: impl Fn()) {
    AD_CONTEXT.with(|c| {
        let mut c = c.borrow_mut();
        assert!(!c.started, "autodiff section is already started");
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
        let node = Node::new(CArc::new(Instruction::AdScope { body }), Type::void());
        b.append(new_node(b.pools(), node))
    });
}

pub fn is_cpu_backend() -> bool {
    RECORDER.with(|r| {
        let r = r.borrow();
        if r.device.is_none() {
            return false;
        }
        r.device
            .as_ref()
            .unwrap()
            .inner
            .query("device_name")
            .map(|s| s == "cpu")
            .unwrap_or(false)
    })
}

pub fn __env_need_backtrace() -> bool {
    match std::env::var("LUISA_BACKTRACE") {
        Ok(s) => s == "1" || s == "ON",
        Err(_) => false,
    }
}

pub fn __unreachable(file: &str, line: u32, col: u32) {
    let path = std::path::Path::new(file);
    let pretty_filename: String;
    if path.exists() {
        pretty_filename = std::fs::canonicalize(path)
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
    } else {
        pretty_filename = file.to_string();
    }
    let msg = if is_cpu_backend() && __env_need_backtrace() {
        let backtrace = get_backtrace();
        format!(
            "unreachable code at {}:{}:{} \nbacktrace: {}",
            pretty_filename, line, col, backtrace
        )
    } else {
        format!(
            "unreachable code at {}:{}:{} \n",
            pretty_filename, line, col
        )
    };
    __current_scope(|b| {
        b.call(
            Func::Unreachable(CBoxedSlice::new(
                CString::new(msg).unwrap().into_bytes_with_nul(),
            )),
            &[],
            Type::void(),
        );
    });
}

#[inline]
pub fn __assert(cond: impl Into<Expr<bool>>, msg: &str, file: &str, line: u32, col: u32) {
    let cond = cond.into();
    let path = std::path::Path::new(file);
    let pretty_filename: String;
    if path.exists() {
        pretty_filename = std::fs::canonicalize(path)
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
    } else {
        pretty_filename = file.to_string();
    }
    let msg = if is_cpu_backend() && __env_need_backtrace() {
        let backtrace = get_backtrace();
        format!(
            "assertion failed: {} at {}:{}:{} \nbacktrace: {}",
            msg, pretty_filename, line, col, backtrace
        )
    } else {
        format!(
            "assertion failed: {} at {}:{}:{} \n",
            msg, pretty_filename, line, col
        )
    };
    __current_scope(|b| {
        b.call(
            Func::Assert(CBoxedSlice::new(
                CString::new(msg).unwrap().into_bytes_with_nul(),
            )),
            &[cond.node()],
            Type::void(),
        );
    });
}

pub(crate) fn need_runtime_check() -> bool {
    cfg!(debug_assertions)
        || match env::var("LUISA_DEBUG") {
            Ok(s) => s == "full" || s == "1",
            Err(_) => false,
        }
        || __env_need_backtrace()
}
