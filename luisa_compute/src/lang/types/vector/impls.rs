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
    pub fn expr_from_elements(elements: [Expr<T>; N]) -> Expr<Self> {
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
                Self::expr_from_elements([$($xs.as_expr()),+])
            }
        }
        impl<T: VectorAlign<$N>> $Vexpr<T> {
            pub fn dot(&self, other: impl AsExpr<Value = $Vn<T>>) -> Expr<T> {
                Expr::<T>::from_node(__current_scope(|s| {
                    s.call(
                        Func::Dot,
                        &[self.node(), other.as_expr().node()],
                        T::type_(),
                    )
                }))
            }
        }
        impl<T: VectorAlign<$N>, X: IntoIndex> Index<X> for $Vexpr<T> {
            type Output = Expr<T>;
            fn index(&self, i: X) -> &Self::Output {
                let i = i.to_u64();

                if need_runtime_check() {
                    lc_assert!(i.lt(($N as u64).expr()));
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
                    lc_assert!(i.lt(($N as u64).expr()));
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
    fn length(&self) -> Expr<Self::T> {
        Expr::<Self::T>::from_node(__current_scope(|s| {
            s.call(Func::Length, &[self.node()], Self::T::type_())
        }))
    }
    fn length_squared(&self) -> Expr<Self::T> {
        Expr::<Self::T>::from_node(__current_scope(|s| {
            s.call(Func::LengthSquared, &[self.node()], Self::T::type_())
        }))
    }
}
