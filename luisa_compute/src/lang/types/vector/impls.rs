use super::*;
use crate::lang::index::IntoIndex;
use std::ops::Index;

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
        Expr::<Self>::from_node(__compose::<Vector<T, N>>(&elements.map(|x| x.node())))
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

                Expr::<T>::from_node(__current_scope(|s| {
                    s.call(
                        Func::ExtractElement,
                        &[self.node(), i.node()],
                        T::type_(),
                    )
                }))._ref()
            }
        }
        impl<T: VectorAlign<$N>, X: IntoIndex> Index<X> for $Vvar<T> {
            type Output = Var<T>;
            fn index(&self, i: X) -> &Self::Output {
                let i = i.to_u64();

                if need_runtime_check() {
                    check_index_lt_usize(i, $N);
                }

                Var::<T>::from_node(__current_scope(|s| {
                    s.call(
                        Func::GetElementPtr,
                        &[self.self_.node(), i.node()],
                        T::type_(),
                    )
                }))._ref()
            }
        }
    }
}
impl_sized!(Vec2(2), VectorExprProxy2, VectorVarProxy2: x, y);
impl_sized!(Vec3(3), VectorExprProxy3, VectorVarProxy3: x, y, z);
impl_sized!(Vec4(4), VectorExprProxy4, VectorVarProxy4: x, y, z, w);

pub trait VectorExprProxy {
    const N: usize;
    type T: Primitive;
    fn node(&self) -> NodeRef;
    fn _permute2(&self, x: u32, y: u32) -> Expr<Vec2<Self::T>>
    where
        Self::T: VectorAlign<2>,
    {
        assert!(x < Self::N as u32);
        assert!(y < Self::N as u32);
        let x = x.expr();
        let y = y.expr();
        Expr::<Vec2<Self::T>>::from_node(__current_scope(|s| {
            s.call(
                Func::Permute,
                &[self.node(), x.node(), y.node()],
                Vec2::<Self::T>::type_(),
            )
        }))
    }
    fn _permute3(&self, x: u32, y: u32, z: u32) -> Expr<Vec3<Self::T>>
    where
        Self::T: VectorAlign<3>,
    {
        assert!(x < Self::N as u32);
        assert!(y < Self::N as u32);
        assert!(z < Self::N as u32);
        let x = x.expr();
        let y = y.expr();
        let z = z.expr();
        Expr::<Vec3<Self::T>>::from_node(__current_scope(|s| {
            s.call(
                Func::Permute,
                &[self.node(), x.node(), y.node(), z.node()],
                Vec3::<Self::T>::type_(),
            )
        }))
    }
    fn _permute4(&self, x: u32, y: u32, z: u32, w: u32) -> Expr<Vec4<Self::T>>
    where
        Self::T: VectorAlign<4>,
    {
        assert!(x < Self::N as u32);
        assert!(y < Self::N as u32);
        assert!(z < Self::N as u32);
        assert!(w < Self::N as u32);
        let x = x.expr();
        let y = y.expr();
        let z = z.expr();
        let w = w.expr();
        Expr::<Vec4<Self::T>>::from_node(__current_scope(|s| {
            s.call(
                Func::Permute,
                &[self.node(), x.node(), y.node(), z.node(), w.node()],
                Vec4::<Self::T>::type_(),
            )
        }))
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
        Expr::<Self>::from_node(__compose::<Self>(&elements.map(|x| x.node())))
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
                let scalar = scalar.as_expr();
                Expr::<Self>::from_node(__current_scope(|b|{
                    b.call(Func::Mat, &[scalar.node()], Self::type_())
                }))
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
                Expr::<$V>::from_node(__current_scope(|b| {
                    b.call(Func::ExtractElement, &[self.0.node, i.node()], <$V>::type_())
                }))
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
                Expr::<$V>::from_node(__current_scope(|b| {
                    b.call(Func::ExtractElement, &[self.node(), i.node()], <$V>::type_())
                }))
            }
        }
        impl IndexRead for Var<$M> {
            type Element = $V;
            fn read<I: IntoIndex>(&self, i: I) -> Expr<Self::Element> {
                let i = i.to_u64();
                if need_runtime_check() {
                    lc_assert!(i.lt($N as u64));
                }
                Expr::<$V>::from_node(__current_scope(|b| {
                    let gep = b.call(Func::GetElementPtr, &[self.node(), i.node()], <$V>::type_());
                    b.load(gep)
                }))
            }
        }
        impl IndexWrite for Var<$M> {
            fn write<I: IntoIndex, V: AsExpr<Value = Self::Element>>(&self, i: I, value: V) {
                let i = i.to_u64();
                let value = value.as_expr();
                if need_runtime_check() {
                    lc_assert!(i.lt($N as u64));
                }
                __current_scope(|b| {
                    let gep = b.call(Func::GetElementPtr, &[self.node(), i.node()], <$V>::type_());
                    b.update(gep, value.node());
                });
            }
        }
    }
}
impl_mat_proxy!(Mat2, Vec2<f32>,2: x, y);
impl_mat_proxy!(Mat3, Vec3<f32>,3: x, y, z);
impl_mat_proxy!(Mat4, Vec4<f32>,4: x, y, z, w);
