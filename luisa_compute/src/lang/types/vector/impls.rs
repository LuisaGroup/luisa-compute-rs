use super::*;

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
    ($Vn:ident($N: literal): $($xs:ident),+) => {
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
    }
}
impl_sized!(Vec2(2): x, y);
impl_sized!(Vec3(3): x, y, z);
impl_sized!(Vec4(4): x, y, z, w);
