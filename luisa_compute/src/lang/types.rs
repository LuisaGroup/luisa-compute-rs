use std::cell::UnsafeCell;
use std::ops::Deref;

use crate::internal_prelude::*;

pub mod alignment;
pub mod array;
pub mod core;
pub mod dynamic;
pub mod shared;
pub mod vector;

// TODO: Check up on comments.

/// A value that can be used in a [`Kernel`](crate::runtime::Kernel) or
/// [`Callable`](crate::runtime::Callable). Call [`expr`](Value::expr) or
/// [`var`](Value::var) to convert into a kernel-trackable type.
pub trait Value: Copy + TypeOf + 'static {
    /// A proxy for additional impls on [`Expr<Self>`].
    type Expr: ExprProxy<Value = Self>;
    /// A proxy for additional impls on [`Var<Self>`].
    type Var: VarProxy<Value = Self>;

    type AtomicRef: AtomicRefProxy<Value = Self>;

    fn expr(self) -> Expr<Self> {
        let node = __current_scope(|s| -> NodeRef {
            let mut buf = vec![0u8; std::mem::size_of::<Self>()];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &self as *const Self as *const u8,
                    buf.as_mut_ptr(),
                    buf.len(),
                );
            }
            s.const_(Const::Generic(CBoxedSlice::new(buf), Self::type_()))
        });
        Expr::<Self>::from_node(node)
    }
    fn var(self) -> Var<Self> {
        self.expr().var()
    }

    fn var_zeroed() -> Var<Self> {
        Var::<Self>::from_node(__current_scope(|b| b.local_zero_init(Self::type_())))
    }
}

/// A trait for implementing remote impls on top of an [`Expr`] using [`Deref`].
///
/// For example, `Expr<[f32; 4]>` dereferences to `ArrayExpr<f32, 4>`, which
/// exposes an [`Index`](std::ops::Index) impl.
pub trait ExprProxy: Copy + 'static {
    type Value: Value<Expr = Self>;

    fn from_expr(expr: Expr<Self::Value>) -> Self;
    fn as_expr_from_proxy(&self) -> &Expr<Self::Value>;
}

/// A trait for implementing remote impls on top of an [`Var`] using [`Deref`].
///
/// For example, `Var<[f32; 4]>` dereferences to `ArrayVar<f32, 4>`, which
/// exposes [`Index`](std::ops::Index) and [`IndexMut`](std::ops::IndexMut)
/// impls.
pub trait VarProxy: Copy + 'static {
    type Value: Value<Var = Self>;
    fn as_var_from_proxy(&self) -> &Var<Self::Value>;

    fn from_var(expr: Var<Self::Value>) -> Self;
}

pub unsafe trait AtomicRefProxy: Copy + 'static {
    type Value: Value<AtomicRef = Self>;
    fn as_atomic_ref_from_proxy(&self) -> &AtomicRef<Self::Value>;

    fn from_atomic_ref(expr: AtomicRef<Self::Value>) -> Self;
}

pub(crate) struct ExprProxyData<T: Value> {
    pub(crate) data: UnsafeCell<Option<T::Expr>>,
}
impl<T: Value> ExprProxyData<T> {
    pub(crate) fn new() -> Self {
        Self {
            data: UnsafeCell::new(None),
        }
    }
    pub(crate) fn deref_(&self, e: Expr<T>) -> &'static T::Expr {
        unsafe {
            let data = self.data.get().as_mut().unwrap();
            if let Some(data) = data {
                return std::mem::transmute(data);
            }
            let v = T::Expr::from_expr(e);
            data.replace(v);
            std::mem::transmute(data.as_ref().unwrap())
        }
    }
}
pub(crate) struct VarProxyData<T: Value> {
    pub(crate) data: UnsafeCell<Option<T::Var>>,
}
impl<T: Value> VarProxyData<T> {
    pub(crate) fn new() -> Self {
        Self {
            data: UnsafeCell::new(None),
        }
    }
    pub(crate) fn deref_(&self, e: Var<T>) -> &'static T::Var {
        unsafe {
            let data = self.data.get().as_mut().unwrap();
            if let Some(data) = data {
                return std::mem::transmute(data);
            }
            let v = T::Var::from_var(e);
            data.replace(v);
            std::mem::transmute(data.as_ref().unwrap())
        }
    }
}
pub(crate) struct AtomciRefProxyDataProxyData<T: Value> {
    pub(crate) data: UnsafeCell<Option<T::AtomicRef>>,
}
impl<T: Value> AtomciRefProxyDataProxyData<T> {
    pub(crate) fn new() -> Self {
        Self {
            data: UnsafeCell::new(None),
        }
    }
    pub(crate) fn deref_(&self, e: AtomicRef<T>) -> &'static T::AtomicRef {
        unsafe {
            let data = self.data.get().as_mut().unwrap();
            if let Some(data) = data {
                return std::mem::transmute(data);
            }
            let v = T::AtomicRef::from_atomic_ref(e);
            data.replace(v);
            std::mem::transmute(data.as_ref().unwrap())
        }
    }
}
/// An expression within a [`Kernel`](crate::runtime::Kernel) or
/// [`Callable`](crate::runtime::Callable). Created from a raw value
/// using [`Value::expr`].
///
/// Note that this does not store the value, and in order to get the result of a
/// function returning an `Expr`, you must call
/// [`Kernel::dispatch`](crate::runtime::Kernel::dispatch).
#[derive(Copy, Clone)]
#[repr(C)]
pub struct Expr<T: Value> {
    pub(crate) node: NodeRef,
    _marker: PhantomData<T>,
    proxy: *mut ExprProxyData<T>,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct TypeTag<T: Value>(PhantomData<T>);

/// A variable within a [`Kernel`](crate::runtime::Kernel) or
/// [`Callable`](crate::runtime::Callable). Created using [`Expr::var`]
/// and [`Value::var`].
///
/// Note that setting a `Var` using direct assignment will not work. Instead,
/// either use the [`store`](Var::store) method or the `track!` macro and `*var
/// = expr` syntax.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct Var<T: Value> {
    pub(crate) node: NodeRef,
    _marker: PhantomData<T>,
    proxy: *mut VarProxyData<T>,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct AtomicRef<T: Value> {
    pub(crate) node: NodeRef,
    _marker: PhantomData<T>,
    proxy: *mut AtomciRefProxyDataProxyData<T>,
}

impl<T: Value> ToNode for AtomicRef<T> {
    fn node(&self) -> NodeRef {
        self.node
    }
}
impl<T: Value> FromNode for AtomicRef<T> {
    fn from_node(node: NodeRef) -> Self {
        let proxy = RECORDER.with(|r| {
            let r = r.borrow();
            let proxy = r.arena.alloc(AtomciRefProxyDataProxyData::<T>::new());
            proxy as *mut _
        });
        Self {
            node,
            _marker: PhantomData,
            proxy,
        }
    }
}
impl<T: Value> Aggregate for Expr<T> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        nodes.push(self.node);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        let node = iter.next().unwrap();
        Self::from_node(node)
    }
}
impl<T: Value> FromNode for Expr<T> {
    fn from_node(node: NodeRef) -> Self {
        let proxy = RECORDER.with(|r| {
            let r = r.borrow();
            let proxy = r.arena.alloc(ExprProxyData::<T>::new());
            proxy as *mut _
        });
        Self {
            node,
            _marker: PhantomData,
            proxy,
        }
    }
}
impl<T: Value> ToNode for Expr<T> {
    fn node(&self) -> NodeRef {
        self.node
    }
}

impl<T: Value> Aggregate for Var<T> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        nodes.push(self.node);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        let node = iter.next().unwrap();
        Self::from_node(node)
    }
}
impl<T: Value> FromNode for Var<T> {
    fn from_node(node: NodeRef) -> Self {
        let proxy = RECORDER.with(|r| {
            let r = r.borrow();
            let proxy = r.arena.alloc(VarProxyData::<T>::new());
            proxy as *mut _
        });
        Self {
            node,
            _marker: PhantomData,
            proxy,
        }
    }
}
impl<T: Value> ToNode for Var<T> {
    fn node(&self) -> NodeRef {
        self.node
    }
}

impl<T: Value> Deref for Expr<T> {
    type Target = T::Expr;
    fn deref(&self) -> &Self::Target {
        unsafe { self.proxy.as_mut().unwrap().deref_(*self) }
    }
}
impl<T: Value> Deref for Var<T> {
    type Target = T::Var;
    fn deref(&self) -> &Self::Target {
        unsafe { self.proxy.as_mut().unwrap().deref_(*self) }
    }
}
impl<T: Value> Deref for AtomicRef<T> {
    type Target = T::AtomicRef;
    fn deref(&self) -> &Self::Target {
        unsafe { self.proxy.as_mut().unwrap().deref_(*self) }
    }
}
impl<T: Value> Expr<T> {
    pub fn var(self) -> Var<T> {
        Var::<T>::from_node(__current_scope(|b| b.local(self.node())))
    }
    pub fn zeroed() -> Self {
        FromNode::from_node(__current_scope(|b| b.zero_initializer(T::type_())))
    }
    pub fn _ref<'a>(self) -> &'a Self {
        RECORDER.with(|r| {
            let r = r.borrow();
            let v: &Expr<T> = r.arena.alloc(self);
            unsafe {
                let v: &'a Expr<T> = std::mem::transmute(v);
                v
            }
        })
    }
    pub unsafe fn bitcast<S: Value>(self) -> Expr<S> {
        assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<S>());
        let ty = S::type_();
        let node = __current_scope(|s| s.bitcast(self.node(), ty));
        Expr::<S>::from_node(node)
    }
    pub fn _type_tag(&self) -> TypeTag<T> {
        TypeTag(PhantomData)
    }
}

impl<T: Value> Var<T> {
    pub fn zeroed() -> Self {
        Self::from_node(__current_scope(|b| {
            b.local_zero_init(<T as TypeOf>::type_())
        }))
    }
    pub fn _ref<'a>(self) -> &'a Self {
        RECORDER.with(|r| {
            let r = r.borrow();
            let v: &Var<T> = r.arena.alloc(self);
            unsafe {
                let v: &'a Var<T> = std::mem::transmute(v);
                v
            }
        })
    }
    pub fn load(&self) -> Expr<T> {
        Expr::<T>::from_nodes(
            &mut __current_scope(|b| {
                let nodes = self.to_vec_nodes();
                let mut ret = vec![];
                for node in nodes {
                    ret.push(b.call(Func::Load, &[node], node.type_().clone()));
                }
                ret
            })
            .into_iter(),
        )
    }
    pub fn store(&self, value: impl AsExpr<Value = T>) {
        crate::lang::_store(self, &value.as_expr());
    }
}
impl<T: Value> AtomicRef<T> {
    pub fn _ref<'a>(self) -> &'a Self {
        RECORDER.with(|r| {
            let r = r.borrow();
            let v: &AtomicRef<T> = r.arena.alloc(self);
            unsafe {
                let v: &'a AtomicRef<T> = std::mem::transmute(v);
                v
            }
        })
    }
}

pub fn _deref_proxy<P: VarProxy>(proxy: &P) -> &Expr<P::Value> {
    proxy.as_var_from_proxy().load()._ref()
}

#[macro_export]
macro_rules! impl_simple_expr_proxy {
    ($([ $($bounds:tt)* ])? $name: ident $([ $($qualifiers:tt)* ])? for $t: ty $(where $($where_bounds:tt)+)?) => {
        #[derive(Clone, Copy)]
        #[repr(transparent)]
        pub struct $name $(< $($bounds)* >)? ($crate::lang::types::Expr<$t>) $(where $($where_bounds)+)?;
        impl $(< $($bounds)* >)? $crate::lang::types::ExprProxy for $name $(< $($qualifiers)* >)? $(where $($where_bounds)+)? {
            type Value = $t;
            fn from_expr(expr: $crate::lang::types::Expr<$t>) -> Self {
                Self(expr)
            }
            fn as_expr_from_proxy(&self) -> &$crate::lang::types::Expr<$t> {
                &self.0
            }
        }
    }
}

#[macro_export]
macro_rules! impl_simple_var_proxy {
    ($([ $($bounds:tt)* ])? $name: ident $([ $($qualifiers:tt)* ])? for $t: ty $(where $($where_bounds:tt)+)?) => {
        #[derive(Clone, Copy)]
        #[repr(transparent)]
        pub struct $name $(< $($bounds)* >)? ($crate::lang::types::Var<$t>) $(where $($where_bounds)+)?;
        impl $(< $($bounds)* >)? $crate::lang::types::VarProxy for $name $(< $($qualifiers)* >)? $(where $($where_bounds)+)? {
            type Value = $t;
            fn from_var(var: $crate::lang::types::Var<$t>) -> Self {
                Self(var)
            }
            fn as_var_from_proxy(&self) -> &$crate::lang::types::Var<$t> {
                &self.0
            }
        }
        impl $(< $($bounds)* >)? std::ops::Deref for $name $(< $($qualifiers)* >)? $(where $($where_bounds)+)? {
            type Target = $crate::lang::types::Expr<$t>;
            fn deref(&self) -> &Self::Target {
                $crate::lang::types::_deref_proxy(self)
            }
        }
    }
}

#[macro_export]
macro_rules! impl_simple_atomic_ref_proxy {
    ($([ $($bounds:tt)* ])? $name: ident $([ $($qualifiers:tt)* ])? for $t: ty $(where $($where_bounds:tt)+)?) => {
        #[derive(Clone, Copy)]
        #[repr(transparent)]
        pub struct $name $(< $($bounds)* >)? ($crate::lang::types::AtomicRef<$t>) $(where $($where_bounds)+)?;
        unsafe impl $(< $($bounds)* >)? $crate::lang::types::AtomicRefProxy for $name $(< $($qualifiers)* >)? $(where $($where_bounds)+)? {
            type Value = $t;
            fn from_atomic_ref(var: $crate::lang::types::AtomicRef<$t>) -> Self {
                Self(var)
            }
            fn as_atomic_ref_from_proxy(&self) -> &$crate::lang::types::AtomicRef<$t> {
                &self.0
            }
        }
    }
}

#[macro_export]
macro_rules! impl_marker_trait {
    ($T:ident for $($t:ty),*) => {
        $(impl $T for $t {})*
    };
}

mod private {
    use super::*;

    pub trait Sealed {}
    impl<T: Value> Sealed for T {}
    impl<T: Value> Sealed for Expr<T> {}
    impl<T: Value> Sealed for &Expr<T> {}
    impl<T: Value> Sealed for Var<T> {}
    impl<T: Value> Sealed for &Var<T> {}

    impl Sealed for ValueType {}
    impl Sealed for ExprType {}
    impl Sealed for VarType {}
}

pub trait Tracked: private::Sealed {
    type Type: TrackingType;
    type Value: Value;
}

pub trait TrackingType: private::Sealed {}
pub struct ValueType;
impl TrackingType for ValueType {}
pub struct ExprType;
impl TrackingType for ExprType {}
pub struct VarType;
impl TrackingType for VarType {}

impl<T: Value> Tracked for T {
    type Type = ValueType;
    type Value = T;
}
impl<T: Value> Tracked for Expr<T> {
    type Type = ExprType;
    type Value = T;
}
impl<T: Value> Tracked for &Expr<T> {
    type Type = ExprType;
    type Value = T;
}
impl<T: Value> Tracked for Var<T> {
    type Type = VarType;
    type Value = T;
}
impl<T: Value> Tracked for &Var<T> {
    type Type = VarType;
    type Value = T;
}

pub trait AsExpr: Tracked {
    fn as_expr(&self) -> Expr<Self::Value>;
}

impl<T: Value> AsExpr for T {
    fn as_expr(&self) -> Expr<T> {
        self.expr()
    }
}
impl<T: Value> AsExpr for Expr<T> {
    fn as_expr(&self) -> Expr<T> {
        self.clone()
    }
}
impl<T: Value> AsExpr for &Expr<T> {
    fn as_expr(&self) -> Expr<T> {
        (*self).clone()
    }
}
impl<T: Value> AsExpr for Var<T> {
    fn as_expr(&self) -> Expr<T> {
        self.load()
    }
}
impl<T: Value> AsExpr for &Var<T> {
    fn as_expr(&self) -> Expr<T> {
        self.load()
    }
}
