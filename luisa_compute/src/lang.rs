use std::any::Any;
use std::cell::{Cell, RefCell};
use std::fmt::Debug;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::{env, unreachable};

use crate::internal_prelude::*;

use bumpalo::Bump;
use indexmap::IndexMap;

use crate::runtime::WeakDevice;

pub mod ir {
    pub use luisa_compute_ir::context::register_type;
    pub use luisa_compute_ir::ir::*;
    pub use luisa_compute_ir::*;
}

pub use ir::NodeRef;
use ir::{
    new_user_node, BasicBlock, Binding, CArc, CallableModuleRef, Const, CpuCustomOp, Func,
    Instruction, IrBuilder, ModulePools, Pooled, Type, TypeOf, UserNodeData,
};

use self::index::IntoIndex;

pub mod control_flow;
pub mod debug;
pub mod autodiff;
pub mod functions;
pub mod index;
pub mod ops;
pub mod poly;
pub mod soa;
pub mod types;
pub mod external;

pub(crate) trait CallFuncTrait {
    fn call<T: Value, S: Value>(self, x: Expr<T>) -> Expr<S>;
    fn call2<T: Value, S: Value, U: Value>(self, x: Expr<T>, y: Expr<S>) -> Expr<U>;
    fn call3<T: Value, S: Value, U: Value, V: Value>(
        self,
        x: Expr<T>,
        y: Expr<S>,
        z: Expr<U>,
    ) -> Expr<V>;
    fn call_void<T: Value>(self, x: Expr<T>);
    fn call2_void<T: Value, S: Value>(self, x: Expr<T>, y: Expr<S>);
    fn call3_void<T: Value, S: Value, U: Value>(self, x: Expr<T>, y: Expr<S>, z: Expr<U>);
}
impl CallFuncTrait for Func {
    fn call<T: Value, S: Value>(self, x: Expr<T>) -> Expr<S> {
        let x = x.node();
        Expr::<S>::from_node(__current_scope(|b| {
            b.call(self, &[x], <S as TypeOf>::type_())
        }))
    }
    fn call2<T: Value, S: Value, U: Value>(self, x: Expr<T>, y: Expr<S>) -> Expr<U> {
        let x = x.node();
        let y = y.node();
        Expr::<U>::from_node(__current_scope(|b| {
            b.call(self, &[x, y], <U as TypeOf>::type_())
        }))
    }
    fn call3<T: Value, S: Value, U: Value, V: Value>(
        self,
        x: Expr<T>,
        y: Expr<S>,
        z: Expr<U>,
    ) -> Expr<V> {
        let x = x.node();
        let y = y.node();
        let z = z.node();
        Expr::<V>::from_node(__current_scope(|b| {
            b.call(self, &[x, y, z], <V as TypeOf>::type_())
        }))
    }
    fn call_void<T: Value>(self, x: Expr<T>) {
        let x = x.node();
        __current_scope(|b| {
            b.call(self, &[x], Type::void());
        });
    }
    fn call2_void<T: Value, S: Value>(self, x: Expr<T>, y: Expr<S>) {
        let x = x.node();
        let y = y.node();
        __current_scope(|b| {
            b.call(self, &[x, y], Type::void());
        });
    }
    fn call3_void<T: Value, S: Value, U: Value>(self, x: Expr<T>, y: Expr<S>, z: Expr<U>) {
        let x = x.node();
        let y = y.node();
        let z = z.node();
        __current_scope(|b| {
            b.call(self, &[x, y, z], Type::void());
        });
    }
}

#[allow(dead_code)]
pub(crate) static KERNEL_ID: AtomicUsize = AtomicUsize::new(0);
// prevent node being shared across kernels
// TODO: replace NodeRef with SafeNodeRef
#[derive(Clone, Copy, Debug)]
pub(crate) struct SafeNodeRef {
    #[allow(dead_code)]
    pub(crate) node: NodeRef,
    #[allow(dead_code)]
    pub(crate) kernel_id: usize,
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

impl<T: Aggregate> Aggregate for Vec<T> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        let len_node = new_user_node(__module_pools(), nodes.len());
        nodes.push(len_node);
        for item in self {
            item.to_nodes(nodes);
        }
    }

    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        let len_node = iter.next().unwrap();
        let len = len_node.unwrap_user_data::<usize>();
        let mut ret = Vec::with_capacity(*len);
        for _ in 0..*len {
            ret.push(T::from_nodes(iter));
        }
        ret
    }
}

impl<T: Aggregate> Aggregate for RefCell<T> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        self.borrow().to_nodes(nodes);
    }

    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        RefCell::new(T::from_nodes(iter))
    }
}
impl<T: Aggregate + Copy> Aggregate for Cell<T> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        self.get().to_nodes(nodes);
    }

    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        Cell::new(T::from_nodes(iter))
    }
}
impl<T: Aggregate> Aggregate for Option<T> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        match self {
            Some(x) => {
                let node = new_user_node(__module_pools(), 1);
                nodes.push(node);
                x.to_nodes(nodes);
            }
            None => {
                let node = new_user_node(__module_pools(), 0);
                nodes.push(node);
            }
        }
    }

    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        let node = iter.next().unwrap();
        let tag = node.unwrap_user_data::<usize>();
        match *tag {
            0 => None,
            1 => Some(T::from_nodes(iter)),
            _ => unreachable!(),
        }
    }
}

pub trait ToNode {
    fn node(&self) -> NodeRef;
}

pub trait NodeLike: FromNode + ToNode {}
impl<T> NodeLike for T where T: FromNode + ToNode {}

pub trait FromNode {
    fn from_node(node: NodeRef) -> Self;
}

// impl<T: Default> FromNode for T {
//     fn from_node(_: NodeRef) -> Self {
//         Default::default()
//     }
// }

fn _store<T1: Aggregate, T2: Aggregate>(var: &T1, value: &T2) {
    let value_nodes = value.to_vec_nodes();
    let self_nodes = var.to_vec_nodes();
    assert_eq!(value_nodes.len(), self_nodes.len());
    __current_scope(|b| {
        for (value_node, self_node) in value_nodes.into_iter().zip(self_nodes.into_iter()) {
            b.update(self_node, value_node);
        }
    })
}

#[inline(always)]
pub fn __new_user_node<T: UserNodeData>(data: T) -> NodeRef {
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

pub(crate) struct Recorder {
    pub(crate) scopes: Vec<IrBuilder>,
    pub(crate) kernel_id: Option<usize>,
    pub(crate) lock: bool,
    pub(crate) captured_resources: IndexMap<Binding, (usize, NodeRef, Binding, Arc<dyn Any>)>,
    pub(crate) cpu_custom_ops: IndexMap<u64, (usize, CArc<CpuCustomOp>)>,
    pub(crate) callables: IndexMap<u64, CallableModuleRef>,
    pub(crate) shared: Vec<NodeRef>,
    pub(crate) device: Option<WeakDevice>,
    pub(crate) block_size: Option<[u32; 3]>,
    pub(crate) building_kernel: bool,
    pub(crate) pools: Option<CArc<ModulePools>>,
    pub(crate) arena: Bump,
    pub(crate) callable_ret_type: Option<CArc<Type>>,
}

impl Recorder {
    pub(crate) fn reset(&mut self) {
        self.scopes.clear();
        self.captured_resources.clear();
        self.cpu_custom_ops.clear();
        self.callables.clear();
        self.lock = false;
        self.device = None;
        self.block_size = None;
        self.arena.reset();
        self.shared.clear();
        self.kernel_id = None;
        self.callable_ret_type = None;
    }

    pub(crate) fn check_on_same_device(&mut self, other: &Device) -> Option<(String, String)> {
        if let Some(device) = &self.device {
            let device = device.upgrade().unwrap();
            if !Arc::ptr_eq(&device.inner, &other.inner) {
                return Some((
                    format!("{} at {:?}", device.name(), Arc::as_ptr(&device.inner)),
                    format!("{} at {:?}", other.name(), Arc::as_ptr(&other.inner)),
                ));
            }
        } else {
            self.device = Some(WeakDevice::new(other));
        }
        None
    }
    pub(crate) fn capture_or_get<T: Any>(
        &mut self,
        binding: ir::Binding,
        handle: &Arc<T>,
        create_node: impl FnOnce() -> Node,
    ) -> NodeRef {
        if let Some((_, node, _, _)) = self.captured_resources.get(&binding) {
            *node
        } else {
            let node = new_node(self.pools.as_ref().unwrap(), create_node());
            let i = self.captured_resources.len();
            self.captured_resources
                .insert(binding, (i, node, binding, handle.clone()));
            node
        }
    }
    pub(crate) fn new() -> Self {
        Recorder {
            scopes: vec![],
            lock: false,
            captured_resources: IndexMap::new(),
            cpu_custom_ops: IndexMap::new(),
            callables: IndexMap::new(),
            shared: vec![],
            device: None,
            block_size: None,
            pools: None,
            arena: Bump::new(),
            building_kernel: false,
            kernel_id: None,
            callable_ret_type: None,
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

/// Don't call this function directly unless you know what you are doing
/** This function is soley for constructing proxies
 *  Given a node, __extract selects the correct Func based on the node's
 * type  It then inserts the extract(node, i) call *at where the node is
 * defined*  *Note*, after insertion, the IrBuilder in the correct/parent
 * scope might not be up to date  Thus, for IrBuilder of each scope, it
 * updates the insertion point to the end of the current basic block
 */
pub fn __extract<T: Value>(node: NodeRef, index: usize) -> NodeRef {
    let inst = &node.get().instruction;
    RECORDER.with(|r| {
        let mut r = r.borrow_mut();

        let pools = {
            let cur_builder = r.scopes.last_mut().unwrap();
            cur_builder.pools()
        };
        let mut b = IrBuilder::new_without_bb(pools.clone());

        if !node.is_argument() && !node.is_uniform() && !node.is_atomic_ref() {
            // These nodes are not attached to any BB
            // however, we need to generate the index node
            // We generate them at the top of current module
            b.set_insert_point(node);
        } else {
            let first_scope = &r.scopes[0];
            let first_scope_bb = first_scope.bb();
            b.set_insert_point(first_scope_bb.first());
        }

        let i = b.const_(Const::Int32(index as i32));
        // Since we have inserted something, the insertion point in cur_builder might
        // not be up to date So we need to set it to the end of the current
        // basic block
        macro_rules! update_builders {
            () => {
                for scope in &mut r.scopes {
                    scope.set_insert_point_to_end();
                }
            };
        }
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
                    let n = b.call_no_append(Func::AtomicRef, &indices, <T as TypeOf>::type_());
                    update_builders!();
                    return n;
                }
                Func::GetElementPtr => {
                    let mut indices = args.to_vec();
                    indices.push(i);
                    let n = b.call(Func::GetElementPtr, &indices, <T as TypeOf>::type_());
                    update_builders!();
                    return n;
                }
                _ => Func::ExtractElement,
            },
            _ => Func::ExtractElement,
        };
        let node = b.call(op, &[node, i], <T as TypeOf>::type_());

        update_builders!();
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

pub const fn packed_size<T: Value>() -> usize {
    (std::mem::size_of::<T>() + 3) / 4
}

pub fn pack_to<V: Value, B>(expr: Expr<V>, buffer: &B, index: impl AsExpr<Value = u32>)
where
    B: IndexWrite<Element = u32> + ToNode,
{
    let index = index.as_expr();
    __current_scope(|b| {
        b.call(
            Func::Pack,
            &[expr.node(), buffer.node(), index.node()],
            Type::void(),
        );
    });
}

pub fn unpack_from<T>(
    buffer: &(impl IndexWrite<Element = u32> + ToNode),
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

pub(crate) fn need_runtime_check() -> bool {
    cfg!(debug_assertions)
        || match env::var("LUISA_DEBUG") {
            Ok(s) => s == "full" || s == "1",
            Err(_) => false,
        }
        || debug::__env_need_backtrace()
}
fn try_eval_const_index(index: NodeRef) -> Option<usize> {
    let inst = &index.get().instruction;
    match inst.as_ref() {
        Instruction::Const(c) => match c {
            Const::Int8(i) => Some(*i as usize),
            Const::Int16(i) => Some(*i as usize),
            Const::Int32(i) => Some(*i as usize),
            Const::Int64(i) => Some(*i as usize),
            Const::Uint8(i) => Some(*i as usize),
            Const::Uint16(i) => Some(*i as usize),
            Const::Uint32(i) => Some(*i as usize),
            Const::Uint64(i) => Some(*i as usize),
            _ => None,
        },
        Instruction::Call(f, args) => match f {
            Func::Cast => try_eval_const_index(args[0]),
            Func::Add => {
                let a = try_eval_const_index(args[0]);
                let b = try_eval_const_index(args[1]);
                match (a, b) {
                    (Some(a), Some(b)) => Some(a + b),
                    _ => None,
                }
            }
            _ => None,
        },
        _ => None,
    }
}
pub(crate) fn check_index_lt_usize(index: impl IntoIndex, size: usize) {
    let index = index.to_u64();
    let i: Option<usize> = try_eval_const_index(index.node());
    if let Some(i) = i {
        assert!(i < size, "Index out of bound, index: {}, size: {}", i, size);
    } else {
        lc_assert!(index.lt(size as u64));
    }
}
