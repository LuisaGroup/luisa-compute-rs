use super::*;
use crate::lang::index::IntoIndex;
use std::ops::{Index, Neg};

impl<T: VectorAlign<N>, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(elements: [T; N]) -> Self {
        Self {
            _align: T::A::default(),
            elements,
        }
    }
}
impl<T: VectorAlign<N>, const N: usize> From<Vector<T, N>> for [T; N] {
    fn from(value: Vector<T, N>) -> Self {
        value.elements
    }
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
    pub fn map(&self, f: impl Fn(T) -> T) -> Self {
        Self {
            _align: T::A::default(),
            elements: self.elements.map(f),
        }
    }
    pub fn from_elems_expr(elements: [Expr<T>; N]) -> Expr<Self> {
        let elements = elements.map(|x| x.node().get());
        Expr::<Self>::from_node(__compose::<Vector<T, N>>(&elements).into())
    }
}

impl<const N: usize> SquareMatrix<N>
where
    f32: VectorAlign<N>,
{
    pub fn to_column_array(&self) -> [[f32; N]; N] {
        self.cols.map(|x| x.elements)
    }
    pub fn from_column_array(array: &[[f32; N]; N]) -> Self {
        Self {
            cols: array.map(|x| Vector::<f32, N>::from_elements(x)),
        }
    }
}

macro_rules! impl_sized {
    ($Vn:ident($N: literal), $Vexpr:ident, $Vvar:ident : $($xs:ident),+) => {
        impl<T: VectorAlign<$N>> $Vn<T> {
            pub fn new($($xs: T),+) -> Self {
                Self {
                    _align: T::A::default(),
                    elements: [$($xs),+],
                }
            }
            pub fn expr($($xs: impl AsExpr<Value = T>),+) -> Expr<Self> {
                Self::from_elems_expr([$($xs.as_expr()),+])
            }
        }
        impl<T: VectorAlign<$N>, X: IntoIndex> Index<X> for $Vexpr<T> {
            type Output = Expr<T>;
            fn index(&self, i: X) -> &Self::Output {
                let i = i.to_u64();

                if need_runtime_check() {
                    check_index_lt_usize(i, $N);
                }
                let self_node = self.self_.node().get();
                let i = i.node().get();
                Expr::<T>::from_node(__current_scope(|s| {
                    s.call(
                        Func::ExtractElement,
                        &[self_node, i],
                        T::type_(),
                    )
                }).into())._ref()
            }
        }
        impl<T: VectorAlign<$N>, X: IntoIndex> Index<X> for $Vvar<T> {
            type Output = Var<T>;
            fn index(&self, i: X) -> &Self::Output {
                let i = i.to_u64();

                if need_runtime_check() {
                    check_index_lt_usize(i, $N);
                }
                let self_node = self.self_.node().get();
                let i = i.node().get();
                Var::<T>::from_node(__current_scope(|s| {
                    s.call(
                        Func::GetElementPtr,
                        &[self_node, i],
                        T::type_(),
                    )
                }).into())._ref()
            }
        }
    }
}
impl_sized!(Vec2(2), VectorExprProxy2, VectorVarProxy2: x, y);
impl_sized!(Vec3(3), VectorExprProxy3, VectorVarProxy3: x, y, z);
impl_sized!(Vec4(4), VectorExprProxy4, VectorVarProxy4: x, y, z, w);

pub trait ZeroOne {
    fn zero() -> Self;
    fn one() -> Self;
}
impl ZeroOne for bool {
    fn zero() -> Self {
        false
    }
    fn one() -> Self {
        true
    }
}
macro_rules! zero_one {
    ($($t:ty),*) => {
        $(
            impl ZeroOne for $t {
                fn zero() -> Self {
                    0
                }
                fn one() -> Self {
                    1
                }
            }
        )*
    }
}
zero_one!(u8, u16, u32, u64, i8, i16, i32, i64);
impl ZeroOne for f16 {
    fn zero() -> Self {
        f16::ZERO
    }
    fn one() -> Self {
        f16::ONE
    }
}
impl ZeroOne for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}
impl ZeroOne for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

impl<T: ZeroOne + VectorAlign<2>> Vector<T, 2> {
    pub fn x() -> Self {
        Self::new(T::one(), T::zero())
    }
    pub fn y() -> Self {
        Self::new(T::zero(), T::one())
    }
}
impl<T: ZeroOne + VectorAlign<3>> Vector<T, 3> {
    pub fn x() -> Self {
        Self::new(T::one(), T::zero(), T::zero())
    }
    pub fn y() -> Self {
        Self::new(T::zero(), T::one(), T::zero())
    }
    pub fn z() -> Self {
        Self::new(T::zero(), T::zero(), T::one())
    }
}
impl<T: ZeroOne + VectorAlign<4>> Vector<T, 4> {
    pub fn x() -> Self {
        Self::new(T::one(), T::zero(), T::zero(), T::zero())
    }
    pub fn y() -> Self {
        Self::new(T::zero(), T::one(), T::zero(), T::zero())
    }
    pub fn z() -> Self {
        Self::new(T::zero(), T::zero(), T::one(), T::zero())
    }
    pub fn w() -> Self {
        Self::new(T::zero(), T::zero(), T::zero(), T::one())
    }
}
impl<T: Neg<Output = T> + VectorAlign<N>, const N: usize> Neg for Vector<T, N> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::from(self.elements.map(|x| -x))
    }
}

pub trait VectorExprProxy {
    const N: usize;
    type T: Primitive;
    fn node(&self) -> SafeNodeRef;
    fn _permute2(&self, x: u32, y: u32) -> Expr<Vec2<Self::T>>
    where
        Self::T: VectorAlign<2>,
    {
        assert!(x < Self::N as u32);
        assert!(y < Self::N as u32);
        let x = x.expr().node().get();
        let y = y.expr().node().get();
        let self_node = self.node().get();
        Expr::<Vec2<Self::T>>::from_node(
            __current_scope(|s| {
                s.call(Func::Permute, &[self_node, x, y], Vec2::<Self::T>::type_())
            })
            .into(),
        )
    }
    fn _permute3(&self, x: u32, y: u32, z: u32) -> Expr<Vec3<Self::T>>
    where
        Self::T: VectorAlign<3>,
    {
        assert!(x < Self::N as u32);
        assert!(y < Self::N as u32);
        assert!(z < Self::N as u32);
        let x = x.expr().node().get();
        let y = y.expr().node().get();
        let z = z.expr().node().get();
        let self_node = self.node().get();
        Expr::<Vec3<Self::T>>::from_node(
            __current_scope(|s| {
                s.call(
                    Func::Permute,
                    &[self_node, x, y, z],
                    Vec3::<Self::T>::type_(),
                )
            })
            .into(),
        )
    }
    fn _permute4(&self, x: u32, y: u32, z: u32, w: u32) -> Expr<Vec4<Self::T>>
    where
        Self::T: VectorAlign<4>,
    {
        assert!(x < Self::N as u32);
        assert!(y < Self::N as u32);
        assert!(z < Self::N as u32);
        assert!(w < Self::N as u32);
        let x = x.expr().node().get();
        let y = y.expr().node().get();
        let z = z.expr().node().get();
        let w = w.expr().node().get();
        let self_node = self.node().get();
        Expr::<Vec4<Self::T>>::from_node(
            __current_scope(|s| {
                s.call(
                    Func::Permute,
                    &[self_node, x, y, z, w],
                    Vec4::<Self::T>::type_(),
                )
            })
            .into(),
        )
    }
}
impl<T: VectorAlign<2>> VectorExprProxy2<T> {
    #[tracked]
    pub fn extend(self, z: impl AsExpr<Value = T>) -> Expr<Vec3<T>>
    where
        T: VectorAlign<3>,
    {
        Vec3::expr(self.x, self.y, z)
    }
}
impl<T: VectorAlign<3>> VectorExprProxy3<T> {
    #[tracked]
    pub fn extend(self, w: impl AsExpr<Value = T>) -> Expr<Vec4<T>>
    where
        T: VectorAlign<4>,
    {
        Vec4::expr(self.x, self.y, self.z, w)
    }
}
impl<const N: usize> SquareMatrix<N>
where
    f32: VectorAlign<N>,
    Self: Value,
{
    pub fn from_elems_expr(elements: [Expr<Vector<f32, N>>; N]) -> Expr<Self> {
        let elements = elements.map(|x| x.node().get());
        Expr::<Self>::from_node(__compose::<Self>(&elements).into())
    }
}
impl SquareMatrix<2> {
    #[tracked]
    pub fn diag_expr(diag: impl AsExpr<Value = Float2>) -> Expr<Self> {
        let diag = diag.as_expr();
        Self::expr(Float2::expr(diag.x, 0.0), Float2::expr(0.0, diag.y))
    }
}
impl SquareMatrix<3> {
    #[tracked]
    pub fn diag_expr(diag: impl AsExpr<Value = Float3>) -> Expr<Self> {
        let diag = diag.as_expr();
        Self::expr(
            Float3::expr(diag.x, 0.0, 0.0),
            Float3::expr(0.0, diag.y, 0.0),
            Float3::expr(0.0, 0.0, diag.z),
        )
    }
}
impl SquareMatrix<4> {
    #[tracked]
    pub fn diag_expr(diag: impl AsExpr<Value = Float4>) -> Expr<Self> {
        let diag = diag.as_expr();
        Self::expr(
            Float4::expr(diag.x, 0.0, 0.0, 0.0),
            Float4::expr(0.0, diag.y, 0.0, 0.0),
            Float4::expr(0.0, 0.0, diag.z, 0.0),
            Float4::expr(0.0, 0.0, 0.0, diag.w),
        )
    }
}
macro_rules! impl_mat_proxy {
    ($M:ident, $V:ty, $N:literal: $($xs:ident),+) => {
        impl $M {
            pub fn expr($($xs: impl AsExpr<Value = $V>),+) -> Expr<Self> {
                Self::from_elems_expr([$($xs.as_expr()),+])
            }
            pub fn full_expr(scalar: impl AsExpr<Value = f32>) -> Expr<Self> {
                let scalar = scalar.as_expr().node().get();
                Expr::<Self>::from_node(__current_scope(|b|{
                    b.call(Func::Mat, &[scalar], Self::type_())
                }).into())
            }
        }
        impl AddExpr<Expr<$M>> for Expr<$M> {
            type Output = Self;
            fn add(self, rhs: Self) -> Self::Output {
                Func::Add.call2(self, rhs)
            }
        }
        impl SubExpr<Expr<$M>> for Expr<$M> {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self::Output {
                Func::Sub.call2(self, rhs)
            }
        }
        impl MulExpr<Expr<$M>> for Expr<$M> {
            type Output = Self;
            fn mul(self, rhs: Self) -> Self::Output {
                Func::Mul.call2(self, rhs)
            }
        }

        impl MulExpr<Expr<f32>> for Expr<$M> {
            type Output = Self;
            fn mul(self, rhs: Expr<f32>) -> Self::Output {
                Func::MatCompMul.call2(self, $M::full_expr(rhs))
            }
        }
        impl MulExpr<f32> for Expr<$M> {
            type Output = Self;
            fn mul(self, rhs: f32) -> Self::Output {
                Func::MatCompMul.call2(self, $M::full_expr(rhs))
            }
        }
        impl MulExpr<Expr<$M>> for Expr<f32> {
            type Output = Expr<$M>;
            fn mul(self, rhs: Expr<$M>) -> Self::Output {
                Func::MatCompMul.call2($M::full_expr(self), rhs)
            }
        }
        impl MulExpr<Expr<$M>> for f32 {
            type Output = Expr<$M>;
            fn mul(self, rhs: Expr<$M>) -> Self::Output {
                Func::MatCompMul.call2($M::full_expr(self), rhs)
            }
        }

        impl DivExpr<Expr<f32>> for Expr<$M> {
            type Output = Self;
            #[tracked]
            fn div(self, rhs: Expr<f32>) -> Self::Output {
                self * rhs.recip()
            }
        }
        impl DivExpr<f32> for Expr<$M> {
            type Output = Self;
            #[tracked]
            fn div(self, rhs: f32) -> Self::Output {
                self * rhs.recip()
            }
        }
        impl DivExpr<Expr<$M>> for Expr<f32> {
            type Output = Expr<$M>;
            #[tracked]
            fn div(self, rhs: Expr<$M>) -> Self::Output {
                self.recip() * rhs
            }
        }
        impl DivExpr<Expr<$M>> for f32 {
            type Output = Expr<$M>;
            #[tracked]
            fn div(self, rhs: Expr<$M>) -> Self::Output {
                self.recip() * rhs
            }
        }


        impl MulExpr<Expr<$V>> for Expr<$M> {
            type Output = Expr<$V>;
            fn mul(self, rhs: Expr<$V>) -> Self::Output {
                Func::Mul.call2(self, rhs)
            }
        }

        impl MatExpr for Expr<$M> {
            type Scalar = Expr<f32>;
            type Value = $M;
            fn comp_mul(&self, rhs: impl AsExpr<Value=$M>) -> Self {
                Func::MatCompMul.call2(*self, rhs.as_expr())
            }
            fn transpose(&self) -> Self {
                Func::Transpose.call(*self)
            }
            fn determinant(&self) -> Self::Scalar {
                Func::Determinant.call(*self)
            }
            fn inverse(&self) -> Self {
                Func::Inverse.call(*self)
            }
        }
        impl Expr<$M> {
            pub fn col<I: IntoIndex>(&self, i: I) -> Expr<$V> {
                self.read(i)
            }
            pub fn row<I: IntoIndex>(&self, i: I) -> Expr<$V> {
                self.transpose().read(i)
            }
        }
        impl<X:IntoIndex> Index<X> for Expr<$M> {
            type Output = Expr<$V>;
            fn index(&self, i: X) -> &Self::Output {
                let i = i.to_u64();
                if need_runtime_check() {
                    check_index_lt_usize(i, $N);
                }
                let i = i.node().get();
                let self_node  = self.node().get();
                Expr::<$V>::from_node(__current_scope(|b| {
                    b.call(Func::ExtractElement, &[self_node, i], <$V>::type_())
                }).into())
                ._ref()
            }
        }
        impl IndexRead for Expr<$M> {
            type Element = $V;
            fn read<I: IntoIndex>(&self, i: I) -> Expr<Self::Element> {
                let i = i.to_u64();
                if need_runtime_check() {
                    lc_assert!(i.lt($N as u64));
                }
                let i = i.node().get();
                let self_node  = self.node().get();
                Expr::<$V>::from_node(__current_scope(|b| {
                    b.call(Func::ExtractElement, &[self_node, i], <$V>::type_())
                }).into())
            }
        }
        impl IndexRead for Var<$M> {
            type Element = $V;
            fn read<I: IntoIndex>(&self, i: I) -> Expr<Self::Element> {
                let i = i.to_u64();
                if need_runtime_check() {
                    lc_assert!(i.lt($N as u64));
                }
                let i = i.node().get();
                let self_node  = self.node().get();
                Expr::<$V>::from_node(__current_scope(|b| {
                    let gep = b.call(Func::GetElementPtr, &[self_node, i], <$V>::type_());
                    b.load(gep)
                }).into())
            }
        }
        impl IndexWrite for Var<$M> {
            fn write<I: IntoIndex, V: AsExpr<Value = Self::Element>>(&self, i: I, value: V) {
                let i = i.to_u64();
                let value = value.as_expr();
                if need_runtime_check() {
                    lc_assert!(i.lt($N as u64));
                }
                let i = i.node().get();
                let self_node  = self.node().get();
                let value  = value.node().get();
                __current_scope(|b| {
                    let gep = b.call(Func::GetElementPtr, &[self_node, i], <$V>::type_());
                    b.update(gep, value);
                });
            }
        }
    }
}
impl_mat_proxy!(Mat2, Vec2<f32>,2: x, y);
impl_mat_proxy!(Mat3, Vec3<f32>,3: x, y, z);
impl_mat_proxy!(Mat4, Vec4<f32>,4: x, y, z, w);
