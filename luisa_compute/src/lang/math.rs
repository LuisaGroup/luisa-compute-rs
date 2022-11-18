#![allow(non_camel_case_types)]
use luisa_compute_derive::{__Aggregate, __Value, function};

use crate::struct_;

use super::{Expr, Value};
macro_rules! def_vec {
    ($t:ident, $el:ty, $align:literal, $($comps:ident), *) => {
        #[repr(C)]
        #[repr(align($align))]
        #[derive(Clone, Copy, Default, Debug)]
        #[derive(__Value)]
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

def_vec!(DVec2, f64, 8, x, y);
def_vec!(DVec3, f64, 16, x, y, z);
def_vec!(DVec4, f64, 16, x, y, z, w);

def_vec!(IVec2, i32, 8, x, y);
def_vec!(IVec3, i32, 16, x, y, z);
def_vec!(IVec4, i32, 16, x, y, z, w);

def_vec!(UVec2, u32, 8, x, y);
def_vec!(UVec3, u32, 16, x, y, z);
def_vec!(UVec4, u32, 16, x, y, z, w);

def_vec!(LVec2, i64, 16, x, y);
def_vec!(LVec3, i64, 32, x, y, z);
def_vec!(LVec4, i64, 32, x, y, z, w);

def_vec!(ULVec2, u64, 16, x, y);
def_vec!(ULVec3, u64, 32, x, y, z);
def_vec!(ULVec4, u64, 32, x, y, z, w);

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

pub type long2 = LVec2;
pub type long3 = LVec3;
pub type long4 = LVec4;

pub type ulong2 = ULVec2;
pub type ulong3 = ULVec3;
pub type ulong4 = ULVec4;

pub type double2 = DVec2;
pub type double3 = DVec3;
pub type double4 = DVec4;

macro_rules! def_make_vec {
    ($name:ident, $t:ident, $el:ty, $($comps:ident), *) => {
        pub fn $name($($comps: Expr<$el>), *) -> Expr<$t> {
            struct_!($t { $($comps), * })
        }
    };
}
def_make_vec!(make_float2, float2, f32, x, y);
def_make_vec!(make_float3, float3, f32, x, y, z);
def_make_vec!(make_float4, float4, f32, x, y, z, w);

def_make_vec!(make_int2, int2, i32, x, y);
def_make_vec!(make_int3, int3, i32, x, y, z);
def_make_vec!(make_int4, int4, i32, x, y, z, w);

def_make_vec!(make_uint2, uint2, u32, x, y);
def_make_vec!(make_uint3, uint3, u32, x, y, z);
def_make_vec!(make_uint4, uint4, u32, x, y, z, w);

def_make_vec!(make_bool2, bool2, bool, x, y);
def_make_vec!(make_bool3, bool3, bool, x, y, z);
def_make_vec!(make_bool4, bool4, bool, x, y, z, w);

def_make_vec!(make_long2, long2, i64, x, y);
def_make_vec!(make_long3, long3, i64, x, y, z);
def_make_vec!(make_long4, long4, i64, x, y, z, w);

def_make_vec!(make_ulong2, ulong2, u64, x, y);
def_make_vec!(make_ulong3, ulong3, u64, x, y, z);
def_make_vec!(make_ulong4, ulong4, u64, x, y, z, w);

def_make_vec!(make_double2, double2, f64, x, y);
def_make_vec!(make_double3, double3, f64, x, y, z);
def_make_vec!(make_double4, double4, f64, x, y, z, w);
