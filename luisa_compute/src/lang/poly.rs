use std::{any::TypeId, collections::HashMap};

use crate::resource::Buffer;

use super::{switch, traits::CommonVarOp, Aggregate, Uint, Value};

pub struct PolyArray<T: ?Sized + 'static> {
    tag: i32,
    get: Box<dyn Fn(&Self, Uint) -> Box<T>>,
    _marker: std::marker::PhantomData<T>,
}
impl<T: ?Sized + 'static> PolyArray<T> {
    pub fn new(tag: i32, get: Box<dyn Fn(&Self, Uint) -> Box<T>>) -> Self {
        Self {
            tag,
            get,
            _marker: std::marker::PhantomData,
        }
    }
    pub fn tag(&self) -> i32 {
        self.tag
    }
}
pub trait PolymorphicImpl<T: ?Sized + 'static>: Value {
    fn new_poly_array(buffer: &Buffer<Self>, tag: i32) -> PolyArray<T>;
}
#[macro_export]
macro_rules! impl_polymorphic {
    ($trait_:ident, $ty:ty) => {
        impl PolymorphicImpl<dyn $trait_> for $ty {
            fn new_poly_array(buffer: &Buffer<Self>, tag: i32) -> PolyArray<dyn $trait_> {
                let buffer = unsafe { buffer.shallow_clone() };
                PolyArray::new(tag, Box::new(move |_, index| Box::new(buffer.var().read(index))))
            }
        }
    };
}
pub struct Polymorphic<T: ?Sized + 'static> {
    _marker: std::marker::PhantomData<T>,
    arrays: Vec<PolyArray<T>>,
}

impl<T: ?Sized + 'static> Polymorphic<T> {
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
            arrays: vec![],
        }
    }
    #[inline]
    pub fn register<U>(&mut self, array: &Buffer<U>) -> u32
    where
        U: PolymorphicImpl<T>,
    {
        let tag = self.arrays.len() as u32;
        let array = U::new_poly_array(array, tag as i32);
        self.arrays.push(array);
        tag
    }

    #[inline]
    pub fn dispatch<R: Aggregate>(&self, tag: Uint, index: Uint, f: impl Fn(&T) -> R) -> R {
        let mut sw = switch::<R>(tag.int());
        for array in &self.arrays {
            sw = sw.case(array.tag, || {
                let obj = (array.get)(array, index);
                f(&obj)
            });
        }
        sw.finish()
    }
}
