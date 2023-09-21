use super::*;
use ::nalgebra as na;

impl<const N: usize, T: VectorAlign<N> + PartialEq + Debug> From<na::SVector<T, N>>
    for Vector<T, N>
{
    fn from(value: na::SVector<T, N>) -> Self {
        Self::from_elements(value.into())
    }
}

impl<const N: usize, T: VectorAlign<N> + PartialEq + Debug> From<Vector<T, N>>
    for na::SVector<T, N>
{
    fn from(value: Vector<T, N>) -> Self {
        Self::from(value.elements)
    }
}

impl<const N: usize> From<na::SMatrix<f32, N, N>> for SquareMatrix<N>
where
    f32: VectorAlign<N>,
{
    fn from(value: na::SMatrix<f32, N, N>) -> Self {
        Self::from_column_array(&value.into())
    }
}

impl<const N: usize> From<SquareMatrix<N>> for na::SMatrix<f32, N, N>
where
    f32: VectorAlign<N>,
{
    fn from(value: SquareMatrix<N>) -> Self {
        Self::from(value.to_column_array())
    }
}
