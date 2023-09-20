use super::alignment::*;
use super::core::*;
use super::*;
use ir::{VectorElementType, VectorType};
use std::fmt::Debug;

#[cfg(feature = "glam")]
mod glam;
#[cfg(feature = "nalgebra")]
mod nalgebra;

pub mod coords;
mod element;
mod impls;
pub mod legacy;
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
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Vector<T: VectorAlign<N>, const N: usize> {
    _align: T::A,
    pub elements: [T; N],
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

        impl<T: VectorAlign<$N, VectorExpr = $ExprName<T>>> ExprProxy for $ExprName<T> {
            type Value = Vector<T, $N>;
        }
        impl<T: VectorAlign<$N>> VectorExprProxy for $ExprName<T> {
            type T = T;
        }
        impl<T: VectorAlign<$N, VectorVar = $VarName<T>>> VarProxy for $VarName<T> {
            type Value = Vector<T, $N>;
        }
        impl<T: VectorAlign<$N, VectorVar = $VarName<T>>> Deref for $VarName<T> {
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
            element: VectorElementType::Scalar(T::primitive()),
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

impl<T: VectorAlign<N>, const N: usize> Vector<T, N> {
    fn _permute2(&self, x: u32, y: u32) -> Vec2<T>
    where
        T: VectorAlign<2>,
    {
        Vector::from_elements([self.elements[x as usize], self.elements[y as usize]])
    }
    fn _permute3(&self, x: u32, y: u32, z: u32) -> Vec3<T>
    where
        T: VectorAlign<3>,
    {
        Vector::from_elements([
            self.elements[x as usize],
            self.elements[y as usize],
            self.elements[z as usize],
        ])
    }
    fn _permute4(&self, x: u32, y: u32, z: u32, w: u32) -> Vec4<T>
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

impl<T: VectorElement> Vec2Swizzle for Vec2<T> {
    type Vec2 = Self;
    type Vec3 = Vec3<T>;
    type Vec4 = Vec4<T>;
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
impl<T: VectorElement> Vec3Swizzle for Vec3<T> {
    type Vec2 = Vec2<T>;
    type Vec3 = Self;
    type Vec4 = Vec4<T>;
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
impl<T: VectorElement> Vec4Swizzle for Vec4<T> {
    type Vec2 = Vec2<T>;
    type Vec3 = Vec3<T>;
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

impl<T: VectorAlign<N>, const N: usize> VectorExprData<T, N> {
    fn _permute2(&self, x: u32, y: u32) -> Expr<Vec2<T>>
    where
        T: VectorAlign<2>,
    {
        assert!(x < N as u32);
        assert!(y < N as u32);
        let x = x.expr();
        let y = y.expr();
        Expr::<Vec2<T>>::from_node(__current_scope(|s| {
            s.call(
                Func::Permute,
                &[self.node, ToNode::node(&x), ToNode::node(&y)],
                Vector::<T, 2>::type_(),
            )
        }))
    }
    fn _permute3(&self, x: u32, y: u32, z: u32) -> Expr<Vec3<T>>
    where
        T: VectorAlign<3>,
    {
        assert!(x < N as u32);
        assert!(y < N as u32);
        assert!(z < N as u32);
        let x = x.expr();
        let y = y.expr();
        let z = z.expr();
        Expr::<Vec3<T>>::from_node(__current_scope(|s| {
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
    fn _permute4(&self, x: u32, y: u32, z: u32, w: u32) -> Expr<Vec4<T>>
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
        Expr::<Vec4<T>>::from_node(__current_scope(|s| {
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
    type Vec3 = Expr<Vec3<T>>;
    type Vec4 = Expr<Vec4<T>>;
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
    type Vec2 = Expr<Vec2<T>>;
    type Vec3 = Self;
    type Vec4 = Expr<Vec4<T>>;
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
    type Vec2 = Expr<Vec2<T>>;
    type Vec3 = Expr<Vec3<T>>;
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

pub type Vec2<T: VectorAlign<2>> = Vector<T, 2>;
pub type Vec3<T: VectorAlign<3>> = Vector<T, 3>;
pub type Vec4<T: VectorAlign<4>> = Vector<T, 4>;

// Matrix

impl<const N: usize> Debug for SquareMatrix<N>
where
    f32: VectorAlign<N>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.elements.fmt(f)
    }
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct SquareMatrix<const N: usize>
where
    f32: VectorAlign<N>,
{
    pub elements: [Vector<f32, N>; N],
}

impl<const N: usize> TypeOf for SquareMatrix<N>
where
    f32: VectorAlign<N>,
{
    fn type_() -> CArc<Type> {
        let type_ = Type::Matrix(ir::MatrixType {
            element: VectorElementType::Scalar(Primitive::Float32),
            dimension: N,
        });
        register_type(type_)
    }
}

impl_simple_expr_proxy!(SquareMatrixExpr2 for SquareMatrix<2>);
impl_simple_var_proxy!(SquareMatrixVar2 for SquareMatrix<2>);

impl_simple_expr_proxy!(SquareMatrixExpr3 for SquareMatrix<3>);
impl_simple_var_proxy!(SquareMatrixVar3 for SquareMatrix<3>);

impl_simple_expr_proxy!(SquareMatrixExpr4 for SquareMatrix<4>);
impl_simple_var_proxy!(SquareMatrixVar4 for SquareMatrix<4>);

impl Value for SquareMatrix<2> {
    type Expr = SquareMatrixExpr2;
    type Var = SquareMatrixVar2;
    type ExprData = ();
    type VarData = ();
}
impl Value for SquareMatrix<3> {
    type Expr = SquareMatrixExpr3;
    type Var = SquareMatrixVar3;
    type ExprData = ();
    type VarData = ();
}
impl Value for SquareMatrix<4> {
    type Expr = SquareMatrixExpr4;
    type Var = SquareMatrixVar4;
    type ExprData = ();
    type VarData = ();
}

pub type Mat2 = SquareMatrix<2>;
pub type Mat3 = SquareMatrix<3>;
pub type Mat4 = SquareMatrix<4>;
