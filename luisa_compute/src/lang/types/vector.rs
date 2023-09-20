use super::alignment::*;
use super::core::*;
use super::*;
use ir::{VectorElementType, VectorType};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[cfg(feature = "glam")]
mod glam;
#[cfg(feature = "nalgebra")]
mod nalgebra;

pub mod coords;
mod element;
pub mod swizzle;

use swizzle::*;

pub trait VectorElement: VectorAlign<2> + VectorAlign<3> + VectorAlign<4> {}
impl<T: VectorAlign<2> + VectorAlign<3> + VectorAlign<4>> VectorElement for T {}

pub trait VectorAlign<const N: usize>: Primitive {
    type A: Alignment;
    type VectorExpr: ExprProxy<Value = Vector<Self, N>>;
    type VectorVar: VarProxy<Value = Vector<Self, N>>;
    type VectorExprData: Clone + FromNode + 'static;
    type VectorVarData: Clone + FromNode + 'static;
}

impl<T: Debug + VectorAlign<N>, const N: usize> Debug for Vector<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.elements.fmt(f)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Hash, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Vector<T: VectorAlign<N>, const N: usize> {
    #[serde(skip)]
    _align: T::A,
    elements: [T; N],
}

impl<T: VectorAlign<N>, const N: usize> Vector<T, N> {
    pub fn from_elements(elements: [T; N]) -> Self {
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
    pub fn splat_expr(element: impl AsExpr<Value = T>) -> Expr<Self> {
        Func::Vec.call(element.as_expr())
    }
    fn _permute2(&self, x: u32, y: u32) -> Vector<T, 2>
    where
        T: VectorAlign<2>,
    {
        Vector::from_elements([self.elements[x as usize], self.elements[y as usize]])
    }
    fn _permute3(&self, x: u32, y: u32, z: u32) -> Vector<T, 3>
    where
        T: VectorAlign<3>,
    {
        Vector::from_elements([
            self.elements[x as usize],
            self.elements[y as usize],
            self.elements[z as usize],
        ])
    }
    fn _permute4(&self, x: u32, y: u32, z: u32, w: u32) -> Vector<T, 4>
    where
        T: VectorAlign<4>,
    {
        Vector::from_elements([
            self.elements[x as usize],
            self.elements[y as usize],
            self.elements[z as usize],
            self.elements[w as usize],
        ])
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct VectorExprData<T: VectorAlign<N>, const N: usize>([Expr<T>; N]);
impl<T: VectorAlign<N>, const N: usize> FromNode for VectorExprData<T, N> {
    fn from_node(node: NodeRef) -> Self {
        Self(std::array::from_fn(|i| {
            FromNode::from_node(__extract::<T>(node, i))
        }))
    }
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct VectorVarData<T: VectorAlign<N>, const N: usize>([Var<T>; N]);
impl<T: VectorAlign<N>, const N: usize> FromNode for VectorVarData<T, N> {
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

pub trait VectorExprProxy {
    type T: Primitive;
}

macro_rules! vector_proxies {
    ($N:literal [ $($c:ident),* ]: $ExprName:ident, $VarName:ident) => {
        #[repr(C)]
        #[derive(Debug, Copy, Clone)]
        pub struct $ExprName<T: VectorAlign<$N>> {
            _node: NodeRef,
            $(pub $c: Expr<T>),*
        }
        #[repr(C)]
        #[derive(Debug, Copy, Clone)]
        pub struct $VarName<T: VectorAlign<$N>> {
            _node: NodeRef,
            $(pub $c: Var<T>),*
        }

        unsafe impl<T: VectorAlign<$N>> HasExprLayout<<Vector<T, $N> as Value>::ExprData> for $ExprName<T> {}
        unsafe impl<T: VectorAlign<$N>> HasVarLayout<<Vector<T, $N> as Value>::VarData> for $VarName<T> {}

        impl<T: VectorAlign<$N>> ExprProxy for $ExprName<T> {
            type Value = Vector<T, $N>;
        }
        impl<T: VectorAlign<$N>> VectorExprProxy for $ExprName<T> {
            type T = T;
        }
        impl<T: VectorAlign<$N>> VarProxy for $VarName<T> {
            type Value = Vector<T, $N>;
        }
        impl<T: VectorAlign<$N>> Deref for $VarName<T> {
            type Target = Expr<Vector<T, $N>>;
            fn deref(&self) -> &Self::Target {
                _deref_proxy(self)
            }
        }
    }
}

vector_proxies!(2 [x, y]: VectorExprProxy2, VectorVarProxy2);
vector_proxies!(3 [x, y, z, r, g, b]: VectorExprProxy3, VectorVarProxy3);
vector_proxies!(4 [x, y, z, w, r, g, b, a]: VectorExprProxy4, VectorVarProxy4);

impl<T: VectorAlign<N>, const N: usize> TypeOf for Vector<T, N> {
    fn type_() -> CArc<Type> {
        let type_ = Type::Vector(VectorType {
            element: VectorElementType::Scalar(T::type_()),
            length: N as u32,
        });
        register_type(type_)
    }
}

impl<T: VectorAlign<N>, const N: usize> Value for Vector<T, N> {
    type Expr = T::VectorExpr;
    type Var = T::VectorVar;
    type ExprData = T::VectorExprData;
    type VarData = T::VectorVarData;
}

impl<T: VectorElement> Vec2Swizzle for Vector<T, 2> {
    type Vec2 = Self;
    type Vec3 = Vector<T, 3>;
    type Vec4 = Vector<T, 4>;
    fn permute2(&self, x: i32, y: i32) -> Self::Vec2 {
        self._permute2(x, y)
    }
    fn permute3(&self, x: i32, y: i32, z: i32) -> Self::Vec3 {
        self._permute3(x, y, z)
    }
    fn permute4(&self, x: i32, y: i32, z: i32, w: i32) -> Self::Vec4 {
        self._permute4(x, y, z, w)
    }
}
impl<T: VectorElement> Vec3Swizzle for Vector<T, 3> {
    type Vec2 = Vector<T, 2>;
    type Vec3 = Self;
    type Vec4 = Vector<T, 4>;
    fn permute2(&self, x: i32, y: i32) -> Self::Vec2 {
        self._permute2(x, y)
    }
    fn permute3(&self, x: i32, y: i32, z: i32) -> Self::Vec3 {
        self._permute3(x, y, z)
    }
    fn permute4(&self, x: i32, y: i32, z: i32, w: i32) -> Self::Vec4 {
        self._permute4(x, y, z, w)
    }
}
impl<T: VectorElement> Vec4Swizzle for Vector<T, 4> {
    type Vec2 = Vector<T, 2>;
    type Vec3 = Vector<T, 3>;
    type Vec4 = Self;
    fn permute2(&self, x: i32, y: i32) -> Self::Vec2 {
        self._permute2(x, y)
    }
    fn permute3(&self, x: i32, y: i32, z: i32) -> Self::Vec3 {
        self._permute3(x, y, z)
    }
    fn permute4(&self, x: i32, y: i32, z: i32, w: i32) -> Self::Vec4 {
        self._permute4(x, y, z, w)
    }
}

impl<T: VectorAlign<N>, const N: usize> VectorExprData<T, N> {
    fn _permute2(&self, x: u32, y: u32) -> Expr<Vector<T, 2>>
    where
        T: VectorAlign<2>,
    {
        assert!(x < N as u32);
        assert!(y < N as u32);
        let x = x.expr();
        let y = y.expr();
        Expr::<Vector<T, 2>>::from_node(__current_scope(|s| {
            s.call(
                Func::Permute,
                &[self.node, ToNode::node(&x), ToNode::node(&y)],
                Vector::<T, 2>::type_(),
            )
        }))
    }
    fn _permute3(&self, x: u32, y: u32, z: u32) -> Expr<Vector<T, 3>>
    where
        T: VectorAlign<3>,
    {
        assert!(x < N as u32);
        assert!(y < N as u32);
        assert!(z < N as u32);
        let x = x.expr();
        let y = y.expr();
        let z = z.expr();
        Expr::<Vector<T, 3>>::from_node(__current_scope(|s| {
            s.call(
                Func::Permute,
                &[
                    self.node,
                    ToNode::node(&x),
                    ToNode::node(&y),
                    ToNode::node(&z),
                ],
                Vector::<T, 3>::type_(),
            )
        }))
    }
    fn _permute4(&self, x: u32, y: u32, z: u32, w: u32) -> Expr<Vector<T, 4>>
    where
        T: VectorAlign<4>,
    {
        assert!(x < N as u32);
        assert!(y < N as u32);
        assert!(z < N as u32);
        assert!(w < N as u32);
        let x = x.expr();
        let y = y.expr();
        let z = z.expr();
        let w = w.expr();
        Expr::<Vector<T, 4>>::from_node(__current_scope(|s| {
            s.call(
                Func::Permute,
                &[
                    self.node,
                    ToNode::node(&x),
                    ToNode::node(&y),
                    ToNode::node(&z),
                    ToNode::node(&w),
                ],
                Vector::<T, 4>::type_(),
            )
        }))
    }
}

impl<T: VectorElement> Vec2Swizzle for VectorExprProxy2<T> {
    type Vec2 = Self;
    type Vec3 = Expr<Vector<T, 3>>;
    type Vec4 = Expr<Vector<T, 4>>;
    fn permute2(&self, x: u32, y: u32) -> Self::Vec2 {
        self._permute2(x, y)
    }
    fn permute3(&self, x: u32, y: u32, z: u32) -> Self::Vec3 {
        self._permute3(x, y, z)
    }
    fn permute4(&self, x: u32, y: u32, z: u32, w: u32) -> Self::Vec4 {
        self._permute4(x, y, z, w)
    }
}

impl<T: VectorElement> Vec3Swizzle for VectorExprProxy3<T> {
    type Vec2 = Expr<Vector<T, 2>>;
    type Vec3 = Self;
    type Vec4 = Expr<Vector<T, 4>>;
    fn permute2(&self, x: u32, y: u32) -> Self::Vec2 {
        self._permute2(x, y)
    }
    fn permute3(&self, x: u32, y: u32, z: u32) -> Self::Vec3 {
        self._permute3(x, y, z)
    }
    fn permute4(&self, x: u32, y: u32, z: u32, w: u32) -> Self::Vec4 {
        self._permute4(x, y, z, w)
    }
}
impl<T: VectorElement> Vec4Swizzle for VectorExprProxy4<T> {
    type Vec2 = Expr<Vector<T, 2>>;
    type Vec3 = Expr<Vector<T, 3>>;
    type Vec4 = Self;
    fn permute2(&self, x: u32, y: u32) -> Self::Vec2 {
        self._permute2(x, y)
    }
    fn permute3(&self, x: u32, y: u32, z: u32) -> Self::Vec3 {
        self._permute3(x, y, z)
    }
    fn permute4(&self, x: u32, y: u32, z: u32, w: u32) -> Self::Vec4 {
        self._permute4(x, y, z, w)
    }
}
