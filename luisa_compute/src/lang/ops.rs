use crate::internal_prelude::*;
use std::ops::*;

use super::types::core::{Floating, Integral, Numeric, Primitive, Signed};
use super::types::vector::{VectorAlign, VectorElement};

pub mod impls;
pub mod spread;
pub mod traits;

trait CastFrom<T: Primitive>: Primitive {}
impl<T: Numeric, S: Numeric> CastFrom<S> for T {}
impl<T: Integral> CastFrom<bool> for T {}

pub trait Linear: Value {
    // Note that without #![feature(generic_const_exprs)], I can't use this within
    // the WithScalar restriction. As such, we can't support higher dimensional
    // vector operations. If that ever becomes necessary, check commit
    // 9e6eacf6b0c2b59a2646f45a727e4d82e84a46cd.
    const N: usize;
    type Scalar: VectorElement;
    type WithScalar<S: VectorElement>: Linear<Scalar = S>;
    // We don't actually know that the vector has equivalent vectors of every
    // primitive type.
}
impl<T: Primitive> Linear for T {
    const N: usize = 1;
    type Scalar = T;
    type WithScalar<S> = S;
}
macro_rules! impl_linear_vectors {
    ($t:ty) => {
        impl_linear_vectors!($t: 2, 3, 4);
    };
    ($t:ty, $($ts:ty),+) => {
        impl_linear_vectors!($t);
        impl_linear_vectors!($($ts),+);
    };
    ($t:ty : $($n:literal),+) => {
        $(
            impl Linear for Vector<$t, $n> {
                const N: usize = $n;
                type Scalar = $t;
                type WithScalar<S> = Vector<S, $n>;
            }
        )+
    }
}
impl_linear_vectors!(bool, f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);
