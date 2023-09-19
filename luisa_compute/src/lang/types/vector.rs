use super::alignment::*;
use super::core::*;
use super::*;
use ir::{MatrixType, VectorElementType, VectorType};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::ops::Mul;

#[cfg(feature = "glam")]
mod glam;
#[cfg(feature = "nalgebra")]
mod nalgebra;

pub mod coords;
mod element;

trait VectorElement<const N: usize>: Primitive {
    type A: Alignment;
}

impl<T: Debug + VectorElement<N>, const N: usize> Debug for Vector<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.elements.fmt(f)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Hash, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Vector<T: VectorElement<N>, const N: usize> {
    _align: T::A,
    elements: [T; N],
}

impl<T: VectorElement<N>, const N: usize> Vector<T, N> {
    pub fn new(elements: [T; N]) -> Self {
        Self {
            _align: T::A::default(),
            elements,
        }
    }
    pub fn splat(element: T) -> Self {
        Self {
            _align: T::A::default(),
            elements: [element; N],
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct VectorExprData<T: VectorElement<N>, const N: usize>([Expr<T>; N]);
impl<const N: usize, T: VectorElement<N>> FromNode for VectorExprData<T, N> {
    fn from_node(node: NodeRef) -> Self {
        Self(std::array::from_fn(|i| {
            FromNode::from_node(__extract::<T>(node, i))
        }))
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct VectorVarData<T: VectorElement<N>, const N: usize>([Var<T>; N]);
impl<const N: usize, T: VectorElement<N>> FromNode for VectorVarData<T, N> {
    fn from_node(node: NodeRef) -> Self {
        Self(std::array::from_fn(|i| {
            FromNode::from_node(__extract::<T>(node, i))
        }))
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct DoubledProxyData<X: FromNode + Copy>(X, X);
impl<X: FromNode + Copy> FromNode for DoubledProxyData<X> {
    fn from_node(node: NodeRef) -> Self {
        Self(X::from_node(node), X::from_node(node))
    }
}

macro_rules! vector_proxies {
    ($N:literal [ $($c:ident),* ]: $ExprName:ident, $VarName:ident) => {
        #[repr(C)]
        #[derive(Debug, Copy, Clone)]
        pub struct $ExprName<T: VectorElement<$N>> {
            _node: NodeRef,
            $(pub $c: Expr<T>),*
        }
        #[repr(C)]
        #[derive(Debug, Copy, Clone)]
        pub struct $VarName<T: VectorElement<$N>> {
            _node: NodeRef,
            $(pub $c: Var<T>),*
        }

        unsafe impl<T: VectorElement<$N>> HasExprLayout<Vector<T, $N>::ExprData> for $ExprName<T> {}
        unsafe impl<T: VectorElement<$N>> HasVarLayout<Vector<T, $N>::VarData> for $VarName<T> {}

        impl<T: VectorElement<$N>> ExprProxy for $ExprName<T> {
            type Value = Vector<T, $N>;
        }
        impl<T: VectorElement<$N>> VarProxy for $VarName<T> {
            type Value = Vector<T, $N>;
        }
        impl<T: VectorElement<$N>> Deref for $VarName<T> {
            type Target = $ExprName<T>;
            fn deref(&self) -> &Self::Target {
                _deref_proxy(self)
            }
        }
    }
}

vector_proxies!(2 [x, y]: VectorExprProxy2, VectorVarProxy2);
vector_proxies!(3 [x, y, z, r, g, b]: VectorExprProxy3, VectorVarProxy3);
vector_proxies!(4 [x, y, z, w, r, g, b, a]: VectorExprProxy4, VectorVarProxy4);

impl<T: VectorElement<N>, const N: usize> TypeOf for Vector<T, N> {
    fn type_() -> CArc<Type> {
        let type_ = Type::Vector(VectorType {
            element: VectorElementType::Scalar(T::type_()),
            length: N as u32,
        });
        register_type(type_)
    }
}

impl<T: VectorElement<2>> Value for Vector<T, 2> {
    type Expr = VectorExprProxy2<T>;
    type Var = VectorVarProxy2<T>;
    type ExprData = VectorExprData<T, 2>;
    type VarData = VectorVarData<T, 2>;
}
impl<T: VectorElement<3>> Value for Vector<T, 3> {
    type Expr = VectorExprProxy3<T>;
    type Var = VectorVarProxy3<T>;
    type ExprData = DoubledProxyData<VectorExprData<T, 3>>;
    type VarData = DoubledProxyData<VectorVarData<T, 3>>;
}
impl<T: VectorElement<4>> Value for Vector<T, 4> {
    type Expr = VectorExprProxy4<T>;
    type Var = VectorVarProxy4<T>;
    type ExprData = DoubledProxyData<VectorExprData<T, 4>>;
    type VarData = DoubledProxyData<VectorVarData<T, 4>>;
}
