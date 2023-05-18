use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fmt::Debug,
    hash::Hash,
};

use crate::*;
use crate::{resource::Buffer, Device};
use luisa_compute_derive::__Value;

use super::{switch, traits::CommonVarOp, Aggregate, Uint, Value};

pub struct PolyArray<K, T: ?Sized + 'static> {
    tag: i32,
    key: K,
    get: Box<dyn Fn(&Self, Uint) -> Box<T>>,
    _marker: std::marker::PhantomData<T>,
}

impl<K, T: ?Sized + 'static> PolyArray<K, T> {
    pub fn new(tag: i32, key: K, get: Box<dyn Fn(&Self, Uint) -> Box<T>>) -> Self {
        Self {
            tag,
            get,
            key,
            _marker: std::marker::PhantomData,
        }
    }
    pub fn tag(&self) -> i32 {
        self.tag
    }
}

pub trait PolymorphicImpl<T: ?Sized + 'static>: Value {
    fn new_poly_array<K>(buffer: &Buffer<Self>, tag: i32, key: K) -> PolyArray<K, T>;
}
#[macro_export]
macro_rules! impl_new_poly_array {
    ($buffer:expr, $tag:expr, $key:expr)=>{
        {
            let buffer = unsafe { $buffer.shallow_clone() };
            luisa_compute::PolyArray::new(
                $tag,
                $key,
                Box::new(move |_, index| Box::new(buffer.var().read(index))),
            )
        }
    }
}
#[macro_export]
macro_rules! impl_polymorphic {
    ($trait_:ident, $ty:ty) => {
        impl PolymorphicImpl<dyn $trait_> for $ty {
            fn new_poly_array<K>(
                buffer: &luisa_compute::Buffer<Self>,
                tag: i32,
                key: K,
            ) -> luisa_compute::PolyArray<K, dyn $trait_> {
                let buffer = unsafe { buffer.shallow_clone() };
                luisa_compute::PolyArray::new(
                    tag,
                    key,
                    Box::new(move |_, index| Box::new(buffer.var().read(index))),
                )
            }
        }
    };
}
struct PolyVec<K, T: ?Sized + 'static> {
    device: Device,
    push: Box<dyn Fn(&mut dyn Any, *const u8) -> u32>,
    get: Box<dyn Fn(&dyn Any, usize) -> &dyn Any>,
    get_mut: Box<dyn Fn(&mut dyn Any, usize) -> &mut dyn Any>,
    build: Box<dyn Fn(&dyn Any, Device) -> PolyArray<K, T>>,
    array: Box<dyn Any>,
}

impl<K: 'static, T: ?Sized + 'static> PolyVec<K, T> {
    fn build(&self) -> PolyArray<K, T> {
        (self.build)(&*self.array.as_ref(), self.device.clone())
    }
}

#[derive(Clone, Copy, Debug, __Value, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(C)]
pub struct TagIndex {
    pub tag: u32,
    pub index: u32,
}
impl TagIndex {
    pub const INVALID: Self = TagIndex {
        tag: u32::MAX,
        index: u32::MAX,
    };
}
impl TagIndexExpr {
    pub fn valid(&self) -> Expr<bool> {
        self.tag().cmpne(u32::MAX)
    }
}

/**
* A  de-virtualized polymorphic array builder
* K: the key type for de-virtualization
       Objects with the same key will shared the same tag
       Due to the multi-stage nature of the library, the keys can be different
       for the same types.
       If K is (), the the array is devirtualized by type only.
* T: The trait to be de-virtualized

* This class works by maintaining two HashMaps, one containing (Key, TypeId) -> Tag
   and the other containing Key -> Tag.
   Thus a tag can be retrieved by either (Key, TypeId) or Key alone if the key is unique for all types.
 */
pub struct PolymorphicBuilder<DevirtualizationKey: Hash + Eq + Clone, Trait: ?Sized + 'static> {
    device: Device,
    key_to_tag: HashMap<(DevirtualizationKey, TypeId), u32>,
    arrays: Vec<PolyVec<DevirtualizationKey, Trait>>,
}

impl<K: Hash + Eq + Clone + 'static + Debug, T: ?Sized + 'static> PolymorphicBuilder<K, T> {
    pub fn new(device: Device) -> Self {
        Self {
            key_to_tag: HashMap::new(),
            arrays: vec![],
            device,
        }
    }
    pub fn get(&self, tag_index: TagIndex) -> &dyn Any {
        let array = &self.arrays[tag_index.tag as usize];
        (array.get)(&*array.array, tag_index.index as usize)
    }
    pub fn get_mut(&mut self, tag_index: TagIndex) -> &mut dyn Any {
        let array = &mut self.arrays[tag_index.tag as usize];
        (array.get_mut)(&mut *array.array, tag_index.index as usize)
    }
    fn tag_from_key<U: PolymorphicImpl<T> + 'static>(&mut self, key: K) -> u32 {
        let pair = (key.clone(), TypeId::of::<U>());
        if let Some(t) = self.key_to_tag.get(&pair) {
            *t
        } else {
            assert_eq!(self.arrays.len(), self.key_to_tag.len());
            let tag = self.arrays.len() as u32;
            self.key_to_tag.insert(pair, tag);
            let key = key.clone();
            self.arrays.push(PolyVec {
                device: self.device.clone(),
                push: Box::new(|array: &mut dyn Any, value: *const u8| -> u32 {
                    let value: &U = unsafe { &*(value as *const U) };
                    let array = array.downcast_mut::<Vec<U>>().unwrap();
                    let index = array.len() as u32;
                    array.push(*value);
                    index
                }),
                get: Box::new(|array: &dyn Any, index: usize| -> &dyn Any {
                    let array = array.downcast_ref::<Vec<U>>().unwrap();
                    let r = &array[index];
                    r as &dyn Any
                }),
                get_mut: Box::new(|array: &mut dyn Any, index: usize| -> &mut dyn Any {
                    let array = array.downcast_mut::<Vec<U>>().unwrap();
                    let r = &mut array[index];
                    r as &mut dyn Any
                }),
                build: Box::new(
                    move |array: &dyn Any, device: Device| -> PolyArray<K, T> {
                        let array = array.downcast_ref::<Vec<U>>().unwrap();
                        let buffer = device.create_buffer_from_slice(&array);
                        PolymorphicImpl::<T>::new_poly_array(
                            &buffer,
                            tag as i32,
                            key.clone(),
                        )
                    },
                ),
                array: Box::new(Vec::<U>::new()),
            });
            tag
        }
    }

    pub fn push<U: PolymorphicImpl<T> + 'static>(&mut self, key: K, value: U) -> TagIndex {
        let tag = self.tag_from_key::<U>(key);
        let array = &mut self.arrays[tag as usize];
        let index = (array.push)(&mut *array.array, &value as *const U as *const u8);
        TagIndex { tag, index }
    }
    pub fn build(self) -> Polymorphic<K, T> {
        let mut poly = Polymorphic::new();
        poly.key_typeid_to_tag = self.key_to_tag;
        for ((key, _), tag) in &poly.key_typeid_to_tag {
            if poly.key_to_tag.contains_key(key) {
                *poly.key_to_tag.get_mut(key).unwrap() = None;
            } else {
                poly.key_to_tag.insert(key.clone(), Some(*tag));
            }
        }
        for a in &self.arrays {
            poly.arrays.push(a.build());
        }
        poly
    }
}

/**
* A  de-virtualized polymorphic array
* K: the key type for de-virtualization
       Objects with the same key will shared the same tag
       Due to the multi-stage nature of the library, the keys can be different
       for the same types.
       If K is (), the the array is devirtualized by type only.
* T: The trait to be de-virtualized

* This class works by maintaining two HashMaps, one containing (Key, TypeId) -> Tag
   and the other containing Key -> Tag.
   Thus a tag can be retrieved by either (Key, TypeId) or Key alone if the key is unique for all types.
 */
pub struct Polymorphic<DevirtualizationKey, Trait: ?Sized + 'static> {
    _marker: std::marker::PhantomData<Trait>,
    arrays: Vec<PolyArray<DevirtualizationKey, Trait>>,
    key_typeid_to_tag: HashMap<(DevirtualizationKey, TypeId), u32>,
    key_to_tag: HashMap<DevirtualizationKey, Option<u32>>,
}

pub struct PolymorphicRef<'a, DevirtualizationKey, Trait: ?Sized + 'static> {
    parent: &'a Polymorphic<DevirtualizationKey, Trait>,
    pub tag_index: Expr<TagIndex>,
}

impl<'a, K: Eq + Clone + Hash, T: ?Sized + 'static> PolymorphicRef<'a, K, T> {
    #[inline]
    pub fn dispatch<R: Aggregate>(&self, f: impl Fn(u32, &K, &T) -> R) -> R {
        let mut sw = switch::<R>(self.tag_index.tag().int());
        for array in &self.parent.arrays {
            sw = sw.case(array.tag, || {
                let obj = (array.get)(array, self.tag_index.index());
                f(array.tag as u32, &array.key, &obj)
            });
        }
        sw.finish()
    }
    #[inline]
    pub fn unwrap<R: Aggregate>(&self, tag: u32, f: impl Fn(&K, &T) -> R) -> R {
        let mut sw = switch::<R>(self.tag_index.tag().int());
        let mut found = false;
        for array in &self.parent.arrays {
            if array.tag as u32 == tag {
                assert!(
                    !found,
                    "Multiple arrays with the same tag {} found in the Polymorphic",
                    tag
                );
                sw = sw.case(array.tag, || {
                    let obj = (array.get)(array, self.tag_index.index());
                    f(&array.key, &obj)
                });
                found = true;
            }
        }
        sw.finish()
    }

    /* Retrieve the tag from the (key, TypeId::of::<U>()) pair
     */
    #[inline]
    pub fn tag_from_key_value<U: PolymorphicImpl<T> + 'static>(&self, key: &K) -> Option<u32> {
        self.parent.tag_from_key_value::<U>(key)
    }

    /* Retrieve the tag from the key only. If multiple arrays have the same key (but different value type),
        this function will return None.
    */
    #[inline]
    pub fn tag_from_key(&self, key: &K) -> Option<u32> {
        self.parent.tag_from_key(key)
    }
}

impl<K: Hash + Eq + Clone, T: ?Sized + 'static> Polymorphic<K, T> {
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
            arrays: vec![],
            key_to_tag: HashMap::new(),
            key_typeid_to_tag: HashMap::new(),
        }
    }
    #[inline]
    pub fn register<U>(&mut self, key: K, array: &Buffer<U>) -> u32
    where
        U: PolymorphicImpl<T> + 'static,
    {
        let pair = (key, TypeId::of::<U>());
        assert!(!self.key_typeid_to_tag.contains_key(&pair));
        self.key_typeid_to_tag
            .insert(pair.clone(), self.arrays.len() as u32);
        let key = pair.0;
        let tag = self.arrays.len() as u32;
        if self.key_to_tag.contains_key(&key) {
            let tag = self.key_to_tag.get_mut(&key).unwrap();
            *tag = None;
        } else {
            self.key_to_tag.insert(key.clone(), Some(tag));
        }
        let array = U::new_poly_array(array, tag as i32, key);
        self.arrays.push(array);
        tag
    }
    #[inline]
    pub fn get<'a>(&'a self, tag_index: Expr<TagIndex>) -> PolymorphicRef<'a, K, T> {
        PolymorphicRef {
            parent: self,
            tag_index,
        }
    }

    /* Retrieve the tag from the (key, TypeId::of::<U>()) pair
     */
    #[inline]
    pub fn tag_from_key_value<U: PolymorphicImpl<T> + 'static>(&self, key: &K) -> Option<u32> {
        let pair = (key.clone(), TypeId::of::<U>());
        self.key_typeid_to_tag.get(&pair).copied()
    }

    /* Retrieve the tag from the key only. If multiple arrays have the same key (but different value type),
        this function will return None.
    */
    #[inline]
    pub fn tag_from_key(&self, key: &K) -> Option<u32> {
        self.key_to_tag.get(key).copied().flatten()
    }
}
