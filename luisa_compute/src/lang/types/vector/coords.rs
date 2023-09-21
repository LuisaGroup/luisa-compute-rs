use super::*;
use std::ops::{Deref, DerefMut};

macro_rules! impl_coords {
    ($T:ident [ $($c:ident), * ]) => {
        #[repr(C)]
        #[derive(Copy, Clone, PartialEq, Eq, Hash)]
        pub struct $T<T: Primitive> {
            $(pub $c: T),*
        }
    }
}
macro_rules! impl_deref {
    ($T:ident; $N:literal) => {
        impl<T: VectorAlign<$N>> Deref for Vector<T, $N> {
            type Target = $T<T>;

            #[inline]
            fn deref(&self) -> &$T<T> {
                unsafe { &*(self as *const Self as *const $T<T>) }
            }
        }

        impl<T: VectorAlign<$N>> DerefMut for Vector<T, $N> {
            #[inline]
            fn deref_mut(&mut self) -> &mut $T<T> {
                unsafe { &mut *(self as *mut Self as *mut $T<T>) }
            }
        }
    };
}

impl_coords!(XY[x, y]);
impl_coords!(XYZ[x, y, z]);
impl_coords!(XYZW[x, y, z, w]);
impl_coords!(RGB[r, g, b]);
impl_coords!(RGBA[r, g, b, a]);

impl_deref![XY; 2];
impl_deref![XYZ; 3];
impl_deref![XYZW; 4];

impl<T: Primitive> Deref for XYZ<T> {
    type Target = RGB<T>;

    #[inline]
    fn deref(&self) -> &RGB<T> {
        unsafe { &*(self as *const Self as *const RGB<T>) }
    }
}
impl<T: Primitive> DerefMut for XYZ<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut RGB<T> {
        unsafe { &mut *(self as *mut Self as *mut RGB<T>) }
    }
}
impl<T: Primitive> Deref for XYZW<T> {
    type Target = RGBA<T>;

    #[inline]
    fn deref(&self) -> &RGBA<T> {
        unsafe { &*(self as *const Self as *const RGBA<T>) }
    }
}
impl<T: Primitive> DerefMut for XYZW<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut RGBA<T> {
        unsafe { &mut *(self as *mut Self as *mut RGBA<T>) }
    }
}
