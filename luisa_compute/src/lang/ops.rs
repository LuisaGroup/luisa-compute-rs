use crate::internal_prelude::*;
use std::ops::*;

use super::types::core::{Floating, Integral, Numeric, Primitive, Signed};
use super::types::vector::VectorElement;

pub mod impls;
pub mod spread;
pub mod traits;

trait CastFrom<T: Primitive>: Primitive {}
impl<T: Numeric, S: Numeric> CastFrom<S> for T {}
impl<T: Integral> CastFrom<bool> for T {}

// Hack because using an associated constant is not allowed within a trait bound
// without #![feature(generic_const_exprs)].
pub trait Linear<const N: usize>: Value {
    type Scalar: VectorElement<N>;
    type WithScalar<S: VectorElement<N>>: Linear<N, Scalar = S>;
    // We don't actually know that the vector has equivalent vectors of every
    // primitive type.
    type WithBool: Linear<N, Scalar = bool>;
}
impl<T: Primitive> Linear<1> for T {
    type Scalar = T;
    type WithScalar<S> = S;
    type WithBool = bool;
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
            impl Linear<$n> for Vector<$t, $n> {
                type Scalar = $t;
                type WithScalar<S> = Vector<S, $n>;
                type WithBool = Vector<bool, $n>;
            }
        )+
    }
}
impl_linear_vectors!(bool, f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);
