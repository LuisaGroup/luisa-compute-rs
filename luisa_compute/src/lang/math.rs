#![allow(non_camel_case_types)]
use crate::prelude::*;

macro_rules! def_vec {
    ($t:ident, $el:ty, $align:literal, $($comps:ident), *) => {
        #[repr(C)]
        #[repr(align($align))]
        #[derive(Clone, Copy, Default, Debug)]
        pub struct $t {
            $(pub $comps: $el), *
        }
    };
}

def_vec!(BVec2, bool, 2, x, y);
def_vec!(BVec3, bool, 4, x, y, z);
def_vec!(BVec4, bool, 4, x, y, z, w);

def_vec!(Vec2, f32, 8, x, y);
def_vec!(Vec3, f32, 16, x, y, z);
def_vec!(Vec4, f32, 16, x, y, z, w);

def_vec!(IVec2, i32, 8, x, y);
def_vec!(IVec3, i32, 16, x, y, z);
def_vec!(IVec4, i32, 16, x, y, z, w);

def_vec!(UVec2, u32, 8, x, y);
def_vec!(UVec3, u32, 16, x, y, z);
def_vec!(UVec4, u32, 16, x, y, z, w);

pub type bool2 = BVec2;
pub type bool3 = BVec3;
pub type bool4 = BVec4;

pub type float2 = Vec2;
pub type float3 = Vec3;
pub type float4 = Vec4;

pub type int2 = IVec2;
pub type int3 = IVec3;
pub type int4 = IVec4;

pub type uint2 = UVec2;
pub type uint3 = UVec3;
pub type uint4 = UVec4;
