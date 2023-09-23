use crate::internal_prelude::*;
use std::ops::*;

use super::types::core::{Floating, Integral, Numeric, Primitive, Signed};
use super::types::vector::{VectorAlign, VectorElement};

mod cast_impls;
mod impls;
mod spread;
mod traits;

pub use spread::*;
pub use traits::*;

pub unsafe trait CastFrom<T: Primitive>: Primitive {}
unsafe impl<T: Numeric, S: Numeric> CastFrom<S> for T {}
unsafe impl<T: Integral> CastFrom<bool> for T {}

pub trait Linear: Value {
    // Note that without #![feature(generic_const_exprs)], I can't use this within
    // the WithScalar restriction. As such, we can't support higher dimensional
    // vector operations. If that ever becomes necessary, check commit
    // 9e6eacf6b0c2b59a2646f45a727e4d82e84a46cd.
    const N: usize;
    type Scalar: VectorElement;
    type WithScalar<S: VectorElement>: Linear<Scalar = S>;
}
impl<T: VectorElement> Linear for T {
    const N: usize = 1;
    type Scalar = T;
    type WithScalar<S: VectorElement> = S;
}

impl<T: VectorElement> Linear for Vector<T, 2> {
    const N: usize = 2;
    type Scalar = T;
    type WithScalar<S: VectorElement> = Vector<S, 2>;
}
impl<T: VectorElement> Linear for Vector<T, 3> {
    const N: usize = 3;
    type Scalar = T;
    type WithScalar<S: VectorElement> = Vector<S, 3>;
}
impl<T: VectorElement> Linear for Vector<T, 4> {
    const N: usize = 4;
    type Scalar = T;
    type WithScalar<S: VectorElement> = Vector<S, 4>;
}
