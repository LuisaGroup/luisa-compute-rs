use std::any::Any;
use std::cell::Cell;
use std::ops::{Deref, DerefMut};

use crate::internal_prelude::*;

pub mod array;
pub mod core;
pub mod dynamic;
pub mod shared;
pub mod vector;

pub type Expr<T> = <T as Value>::Expr;
pub type Var<T> = <T as Value>::Var;

pub trait Value: Copy + ir::TypeOf + 'static {
    type Expr: ExprProxy<Value = Self>;
    type Var: VarProxy<Value = Self>;
    fn fields() -> Vec<String>;
    fn expr(self) -> Self::Expr {
        const_(self)
    }
    fn var(self) -> Self::Var {
        local::<Self>(self.expr())
    }
}

pub trait ExprProxy: Copy + Aggregate + NodeLike {
    type Value: Value<Expr = Self>;

    fn var(self) -> Var<Self::Value> {
        def(self)
    }

    fn zeroed() -> Self {
        zeroed::<Self::Value>()
    }
}

pub trait VarProxy: Copy + Aggregate + NodeLike {
    type Value: Value<Var = Self>;
    fn store<U: Into<Expr<Self::Value>>>(&self, value: U) {
        let value = value.into();
        super::_store(self, &value);
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
            _phantom: PhantomData,
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
    fn zeroed() -> Self {
        local_zeroed::<Self::Value>()
    }
}

pub struct VarDerefProxy<P, T: Value>
where
    P: VarProxy<Value = T>,
{
    pub(crate) var: P,
    pub(crate) dirty: Cell<bool>,
    pub(crate) assigned: Expr<T>,
    pub(crate) _phantom: PhantomData<T>,
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

fn def<E: ExprProxy<Value = T>, T: Value>(init: E) -> Var<T> {
    Var::<T>::from_node(__current_scope(|b| b.local(init.node())))
}
fn local<T: Value>(init: Expr<T>) -> Var<T> {
    Var::<T>::from_node(__current_scope(|b| b.local(init.node())))
}

fn local_zeroed<T: Value>() -> Var<T> {
    Var::<T>::from_node(__current_scope(|b| {
        b.local_zero_init(<T as TypeOf>::type_())
    }))
}

fn zeroed<T: Value>() -> T::Expr {
    FromNode::from_node(__current_scope(|b| b.zero_initializer(T::type_())))
}

fn const_<T: Value + Copy + 'static>(value: T) -> T::Expr {
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
