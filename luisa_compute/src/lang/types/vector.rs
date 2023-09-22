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
pub mod swizzle;

pub use impls::*;
pub use swizzle::*;

pub trait VectorElement: VectorAlign<2> + VectorAlign<3> + VectorAlign<4> {}
impl<T: VectorAlign<2> + VectorAlign<3> + VectorAlign<4>> VectorElement for T {}

pub trait VectorAlign<const N: usize>: Primitive {
    type A: Alignment;
    type VectorExpr: ExprProxy<Value = Vector<Self, N>>;
    type VectorVar: VarProxy<Value = Vector<Self, N>>;
    type VectorAtomicRef: AtomicRefProxy<Value = Vector<Self, N>>;
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
#[derive(Copy, Clone)]
pub struct VectorExprData<T: VectorAlign<N>, const N: usize>([Expr<T>; N]);
impl<T: VectorAlign<N>, const N: usize> FromNode for VectorExprData<T, N> {
    fn from_node(node: NodeRef) -> Self {
        Self(std::array::from_fn(|i| {
            FromNode::from_node(__extract::<T>(node, i))
        }))
    }
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct VectorVarData<T: VectorAlign<N>, const N: usize>([Var<T>; N]);
impl<T: VectorAlign<N>, const N: usize> FromNode for VectorVarData<T, N> {
    fn from_node(node: NodeRef) -> Self {
        Self(std::array::from_fn(|i| {
            FromNode::from_node(__extract::<T>(node, i))
        }))
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VectorAtomicRefData<T: VectorAlign<N>, const N: usize>([AtomicRef<T>; N]);
impl<T: VectorAlign<N>, const N: usize> FromNode for VectorAtomicRefData<T, N> {
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
    ($N:literal [ $($c:ident),* ]: $ExprName:ident, $VarName:ident, $AtomicName:ident) => {
        #[repr(C)]
        #[derive(Copy, Clone)]
        pub struct $ExprName<T: VectorAlign<$N>> {
            self_: Expr<Vector<T, $N>>,
            $(pub $c: Expr<T>),*
        }
        #[repr(C)]
        #[derive(Copy, Clone)]
        pub struct $VarName<T: VectorAlign<$N>> {
            self_: Var<Vector<T, $N>>,
            $(pub $c: Var<T>),*
        }
        #[repr(C)]
        #[derive(Copy, Clone)]
        pub struct $AtomicName<T: VectorAlign<$N>> {
            self_: AtomicRef<Vector<T, $N>>,
            $(pub $c: AtomicRef<T>),*
        }
        impl<T: VectorAlign<$N, VectorExpr = $ExprName<T>>> ExprProxy for $ExprName<T> {
            type Value = Vector<T, $N>;
            #[allow(unused_assignments)]
            fn from_expr(e:Expr<Self::Value>) -> Self {
                let data = VectorExprData::<T, $N>::from_node(e.node());
                let mut i = 0;
                $(
                    let $c = data.0[i].clone();
                    i += 1;
                    if i >= $N { i = 0; }
                )*
                Self{
                    self_: e,
                    $($c),*
                }
            }
            fn as_expr_from_proxy(&self)->&Expr<Self::Value> {
                &self.self_
            }
        }
        impl<T: VectorAlign<$N>> VectorExprProxy for $ExprName<T> {
            const N: usize = $N;
            type T = T;
            fn node(&self) -> NodeRef {
                self.self_.node()
            }
        }
        impl<T: VectorAlign<$N, VectorVar = $VarName<T>>> VarProxy for $VarName<T> {
            type Value = Vector<T, $N>;
            #[allow(unused_assignments)]
            fn from_var(e:Var<Self::Value>) -> Self {
                let data = VectorVarData::<T, $N>::from_node(e.node());
                let mut i = 0;
                $(
                    let $c = data.0[i].clone();
                    i += 1;
                    if i >= $N { i = 0; }
                )*
                Self{
                    self_: e,
                    $($c),*
                }
            }
            fn as_var_from_proxy(&self)->&Var<Self::Value> {
                &self.self_
            }
        }
        impl<T: VectorAlign<$N, VectorVar = $VarName<T>>> Deref for $VarName<T> {
            type Target = Expr<Vector<T, $N>>;
            fn deref(&self) -> &Self::Target {
                _deref_proxy(self)
            }
        }
        unsafe impl<T: VectorAlign<$N, VectorAtomicRef = $AtomicName<T>>> AtomicRefProxy for $AtomicName<T> {
            type Value = Vector<T, $N>;
            #[allow(unused_assignments)]
            fn from_atomic_ref(e:AtomicRef<Self::Value>) -> Self {
                let data = VectorAtomicRefData::<T, $N>::from_node(e.node());
                let mut i = 0;
                $(
                    let $c = data.0[i].clone();
                    i += 1;
                    if i >= $N { i = 0; }
                )*
                Self{
                    self_: e,
                    $($c),*
                }
            }
            fn as_atomic_ref_from_proxy(&self)->&AtomicRef<Self::Value> {
                &self.self_
            }
        }
    }
}

vector_proxies!(2 [x, y]: VectorExprProxy2, VectorVarProxy2, VectorAtomicRefProxy2);
vector_proxies!(3 [x, y, z, r, g, b]: VectorExprProxy3, VectorVarProxy3, VectorAtomicRefProxy3);
vector_proxies!(4 [x, y, z, w, r, g, b, a]: VectorExprProxy4, VectorVarProxy4, VectorAtomicRefProxy4);

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
    type AtomicRef = T::VectorAtomicRef;
}

impl<T: VectorAlign<N>, const N: usize> Vector<T, N> {
    fn _permute2(&self, x: u32, y: u32) -> Vec2<T>
    where
        T: VectorAlign<2>,
    {
        Vector::from([self.elements[x as usize], self.elements[y as usize]])
    }
    fn _permute3(&self, x: u32, y: u32, z: u32) -> Vec3<T>
    where
        T: VectorAlign<3>,
    {
        Vector::from([
            self.elements[x as usize],
            self.elements[y as usize],
            self.elements[z as usize],
        ])
    }
    fn _permute4(&self, x: u32, y: u32, z: u32, w: u32) -> Vec4<T>
    where
        T: VectorAlign<4>,
    {
        Vector::from([
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

impl<T: VectorElement> Vec2Swizzle for VectorExprProxy2<T> {
    type Vec2 = Expr<Vec2<T>>;
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
impl<T: VectorElement> Vec4Swizzle for VectorExprProxy4<T> {
    type Vec2 = Expr<Vec2<T>>;
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

pub type Vec2<T> = Vector<T, 2>;
pub type Vec3<T> = Vector<T, 3>;
pub type Vec4<T> = Vector<T, 4>;

pub mod alias {
    use super::*;
    pub type Half2 = Vec2<f16>;
    pub type Half3 = Vec3<f16>;
    pub type Half4 = Vec4<f16>;
    pub type Float2 = Vec2<f32>;
    pub type Float3 = Vec3<f32>;
    pub type Float4 = Vec4<f32>;
    pub type Double2 = Vec2<f64>;
    pub type Double3 = Vec3<f64>;
    pub type Double4 = Vec4<f64>;
    pub type Byte2 = Vec2<i8>;
    pub type Byte3 = Vec3<i8>;
    pub type Byte4 = Vec4<i8>;
    pub type Short2 = Vec2<i16>;
    pub type Short3 = Vec3<i16>;
    pub type Short4 = Vec4<i16>;
    pub type Int2 = Vec2<i32>;
    pub type Int3 = Vec3<i32>;
    pub type Int4 = Vec4<i32>;
    pub type Long2 = Vec2<i64>;
    pub type Long3 = Vec3<i64>;
    pub type Long4 = Vec4<i64>;
    pub type Ulong2 = Vec2<u64>;
    pub type Ulong3 = Vec3<u64>;
    pub type Ulong4 = Vec4<u64>;
    pub type Ubyte2 = Vec2<u8>;
    pub type Ubyte3 = Vec3<u8>;
    pub type Ubyte4 = Vec4<u8>;
    pub type Ushort2 = Vec2<u16>;
    pub type Ushort3 = Vec3<u16>;
    pub type Ushort4 = Vec4<u16>;
    pub type Uint2 = Vec2<u32>;
    pub type Uint3 = Vec3<u32>;
    pub type Uint4 = Vec4<u32>;
    pub type Bool2 = Vec2<bool>;
    pub type Bool3 = Vec3<bool>;
    pub type Bool4 = Vec4<bool>;
}

// Matrix

impl<const N: usize> Debug for SquareMatrix<N>
where
    f32: VectorAlign<N>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cols.fmt(f)
    }
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq)]
pub struct SquareMatrix<const N: usize>
where
    f32: VectorAlign<N>,
{
    pub cols: [Vector<f32, N>; N],
}

impl<const N: usize> TypeOf for SquareMatrix<N>
where
    f32: VectorAlign<N>,
{
    fn type_() -> CArc<Type> {
        let type_ = Type::Matrix(ir::MatrixType {
            element: VectorElementType::Scalar(f32::primitive()),
            dimension: N as u32,
        });
        register_type(type_)
    }
}

impl_simple_expr_proxy!(SquareMatrixExpr2 for SquareMatrix<2>);
impl_simple_var_proxy!(SquareMatrixVar2 for SquareMatrix<2>);
impl_simple_atomic_ref_proxy!(SquareMatrixAtomicRef2 for SquareMatrix<2>);

impl_simple_expr_proxy!(SquareMatrixExpr3 for SquareMatrix<3>);
impl_simple_var_proxy!(SquareMatrixVar3 for SquareMatrix<3>);
impl_simple_atomic_ref_proxy!(SquareMatrixAtomicRef3 for SquareMatrix<3>);

impl_simple_expr_proxy!(SquareMatrixExpr4 for SquareMatrix<4>);
impl_simple_var_proxy!(SquareMatrixVar4 for SquareMatrix<4>);
impl_simple_atomic_ref_proxy!(SquareMatrixAtomicRef4 for SquareMatrix<4>);

impl Value for SquareMatrix<2> {
    type Expr = SquareMatrixExpr2;
    type Var = SquareMatrixVar2;
    type AtomicRef = SquareMatrixAtomicRef2;
}
impl Value for SquareMatrix<3> {
    type Expr = SquareMatrixExpr3;
    type Var = SquareMatrixVar3;
    type AtomicRef = SquareMatrixAtomicRef3;
}
impl Value for SquareMatrix<4> {
    type Expr = SquareMatrixExpr4;
    type Var = SquareMatrixVar4;
    type AtomicRef = SquareMatrixAtomicRef4;
}

impl SquareMatrix<4> {
    #[inline]
    pub fn into_affine3x4(self) -> [f32; 12] {
        [
            self.cols[0].x,
            self.cols[1].x,
            self.cols[2].x,
            self.cols[3].x,
            self.cols[0].y,
            self.cols[1].y,
            self.cols[2].y,
            self.cols[3].y,
            self.cols[0].z,
            self.cols[1].z,
            self.cols[2].z,
            self.cols[3].z,
        ]
    }
}
pub type Mat2 = SquareMatrix<2>;
pub type Mat3 = SquareMatrix<3>;
pub type Mat4 = SquareMatrix<4>;
impl Mat2 {
    pub fn identity() -> Self {
        Self {
            cols: [Vector::from([1.0, 0.0]), Vector::from([0.0, 1.0])],
        }
    }
}
impl Mat3 {
    pub fn identity() -> Self {
        Self {
            cols: [
                Vector::from([1.0, 0.0, 0.0]),
                Vector::from([0.0, 1.0, 0.0]),
                Vector::from([0.0, 0.0, 1.0]),
            ],
        }
    }
}

impl Mat4 {
    pub fn identity() -> Self {
        Self {
            cols: [
                Vector::from([1.0, 0.0, 0.0, 0.0]),
                Vector::from([0.0, 1.0, 0.0, 0.0]),
                Vector::from([0.0, 0.0, 1.0, 0.0]),
                Vector::from([0.0, 0.0, 0.0, 1.0]),
            ],
        }
    }
}
