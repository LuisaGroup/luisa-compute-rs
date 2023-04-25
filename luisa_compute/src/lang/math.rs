pub use super::swizzle::*;
use super::{Aggregate, ExprProxy, Value, VarProxy, __extract, traits::*, Float};
use crate::*;
use half::f16;
use luisa_compute_ir::{
    context::register_type,
    ir::{Func, MatrixType, NodeRef, Primitive, Type, VectorElementType, VectorType},
    TypeOf,
};
use std::ops::Mul;
use std::ops::*;

macro_rules! def_vec {
    ($name:ident, $glam_type:ident, $scalar:ty, $align:literal, $($comp:ident), *) => {
        #[repr(C, align($align))]
        #[derive(Copy, Clone, Debug, Default)]
        pub struct $name {
            $(pub $comp: $scalar), *
        }
        impl $name {
            #[inline]
            pub const fn new($($comp: $scalar), *) -> Self {
                Self { $($comp), * }
            }
            #[inline]
            pub const fn splat(scalar: $scalar) -> Self {
                Self { $($comp: scalar), * }
            }
        }
        impl From<$name> for glam::$glam_type {
            #[inline]
            fn from(v: $name) -> Self {
                Self::new($(v.$comp), *)
            }
        }
        impl From<glam::$glam_type> for $name {
            #[inline]
            fn from(v: glam::$glam_type) -> Self {
                Self::new($(v.$comp), *)
            }
        }
    };
}
macro_rules! def_packed_vec {
    ($name:ident, $vec_type:ident, $glam_type:ident, $scalar:ty, $($comp:ident), *) => {
        #[repr(C)]
        #[derive(Copy, Clone, Debug, Default, __Value)]
        pub struct $name {
            $(pub $comp: $scalar), *
        }
        impl $name {
            #[inline]
            pub const fn new($($comp: $scalar), *) -> Self {
                Self { $($comp), * }
            }
            #[inline]
            pub const fn splat(scalar: $scalar) -> Self {
                Self { $($comp: scalar), * }
            }
        }
        impl From<$name> for glam::$glam_type {
            #[inline]
            fn from(v: $name) -> Self {
                Self::new($(v.$comp), *)
            }
        }
        impl From<glam::$glam_type> for $name {
            #[inline]
            fn from(v: glam::$glam_type) -> Self {
                Self::new($(v.$comp), *)
            }
        }
        impl From<$name> for $vec_type {
            #[inline]
            fn from(v: $name) -> Self {
                Self::new($(v.$comp), *)
            }
        }
        impl From<$vec_type> for $name {
            #[inline]
            fn from(v: $vec_type) -> Self {
                Self::new($(v.$comp), *)
            }
        }
    };
}
macro_rules! def_packed_vec_no_glam {
    ($name:ident, $vec_type:ident, $scalar:ty, $($comp:ident), *) => {
        #[repr(C)]
        #[derive(Copy, Clone, Debug, Default, __Value)]
        pub struct $name {
            $(pub $comp: $scalar), *
        }
        impl $name {
            #[inline]
            pub const fn new($($comp: $scalar), *) -> Self {
                Self { $($comp), * }
            }
            #[inline]
            pub const fn splat(scalar: $scalar) -> Self {
                Self { $($comp: scalar), * }
            }
        }
        impl From<$name> for $vec_type {
            #[inline]
            fn from(v: $name) -> Self {
                Self::new($(v.$comp), *)
            }
        }
        impl From<$vec_type> for $name {
            #[inline]
            fn from(v: $vec_type) -> Self {
                Self::new($(v.$comp), *)
            }
        }
    };
}
macro_rules! def_vec_no_glam {
    ($name:ident, $scalar:ty, $align:literal, $($comp:ident), *) => {
        #[repr(C, align($align))]
        #[derive(Copy, Clone, Debug, Default)]
        pub struct $name {
            $(pub $comp: $scalar), *
        }
        impl $name {
            #[inline]
            pub fn new($($comp: $scalar), *) -> Self {
                Self { $($comp), * }
            }
            #[inline]
            pub fn splat(scalar: $scalar) -> Self {
                Self { $($comp: scalar), * }
            }
        }
    };
}
def_vec!(Float2, Vec2, f32, 8, x, y);
def_vec!(Float3, Vec3, f32, 16, x, y, z);
def_vec!(Float4, Vec4, f32, 16, x, y, z, w);

def_packed_vec!(PackedFloat2, Float2, Vec2, f32, x, y);
def_packed_vec!(PackedFloat3, Float3, Vec3, f32, x, y, z);
def_packed_vec!(PackedFloat4, Float4, Vec4, f32, x, y, z, w);

def_vec!(Uint2, UVec2, u32, 8, x, y);
def_vec!(Uint3, UVec3, u32, 16, x, y, z);
def_vec!(Uint4, UVec4, u32, 16, x, y, z, w);

def_packed_vec!(PackedUint2, Uint2, UVec2, u32, x, y);
def_packed_vec!(PackedUint3, Uint3, UVec3, u32, x, y, z);
def_packed_vec!(PackedUint4, Uint4, UVec4, u32, x, y, z, w);

def_vec!(Int2, IVec2, i32, 8, x, y);
def_vec!(Int3, IVec3, i32, 16, x, y, z);
def_vec!(Int4, IVec4, i32, 16, x, y, z, w);

def_packed_vec!(PackedInt2, Int2, IVec2, i32, x, y);
def_packed_vec!(PackedInt3, Int3, IVec3, i32, x, y, z);
def_packed_vec!(PackedInt4, Int4, IVec4, i32, x, y, z, w);

def_vec!(Double2, DVec2, f64, 16, x, y);
def_vec!(Double3, DVec3, f64, 32, x, y, z);
def_vec!(Double4, DVec4, f64, 32, x, y, z, w);

def_vec!(Bool2, BVec2, bool, 2, x, y);
def_vec!(Bool3, BVec3, bool, 4, x, y, z);
def_vec!(Bool4, BVec4, bool, 4, x, y, z, w);

def_packed_vec!(PackedBool2, Bool2, BVec2, bool, x, y);
def_packed_vec!(PackedBool3, Bool3, BVec3, bool, x, y, z);
def_packed_vec!(PackedBool4, Bool4, BVec4, bool, x, y, z, w);

def_vec_no_glam!(Ulong2, u64, 16, x, y);
def_vec_no_glam!(Ulong3, u64, 32, x, y, z);
def_vec_no_glam!(Ulong4, u64, 32, x, y, z, w);

def_packed_vec_no_glam!(PackedUlong2, Ulong2, u64, x, y);
def_packed_vec_no_glam!(PackedUlong3, Ulong3, u64, x, y, z);
def_packed_vec_no_glam!(PackedUlong4, Ulong4, u64, x, y, z, w);

def_vec_no_glam!(Long2, i64, 16, x, y);
def_vec_no_glam!(Long3, i64, 32, x, y, z);
def_vec_no_glam!(Long4, i64, 32, x, y, z, w);

def_packed_vec_no_glam!(PackedLong2, Long2, i64, x, y);
def_packed_vec_no_glam!(PackedLong3, Long3, i64, x, y, z);
def_packed_vec_no_glam!(PackedLong4, Long4, i64, x, y, z, w);

def_vec_no_glam!(Ushort2, u16, 4, x, y);
def_vec_no_glam!(Ushort3, u16, 8, x, y, z);
def_vec_no_glam!(Ushort4, u16, 8, x, y, z, w);

def_packed_vec_no_glam!(PackedUshort2, Ushort2, u16, x, y);
def_packed_vec_no_glam!(PackedUshort3, Ushort3, u16, x, y, z);
def_packed_vec_no_glam!(PackedUshort4, Ushort4, u16, x, y, z, w);

def_vec_no_glam!(Short2, i16, 4, x, y);
def_vec_no_glam!(Short3, i16, 8, x, y, z);
def_vec_no_glam!(Short4, i16, 8, x, y, z, w);

def_packed_vec_no_glam!(PackedShort2, Short2, i16, x, y);
def_packed_vec_no_glam!(PackedShort3, Short3, i16, x, y, z);
def_packed_vec_no_glam!(PackedShort4, Short4, i16, x, y, z, w);

def_vec_no_glam!(Half2, f16, 4, x, y);
def_vec_no_glam!(Half3, f16, 8, x, y, z);
def_vec_no_glam!(Half4, f16, 8, x, y, z, w);

// def_packed_vec_no_glam!(PackedHalf2, f16, x, y);
// def_packed_vec_no_glam!(PackedHalf3, f16, x, y, z);
// pub type PackHalf4 = Half4;

def_vec_no_glam!(Ubyte2, u8, 2, x, y);
def_vec_no_glam!(Ubyte3, u8, 4, x, y, z);
def_vec_no_glam!(Ubyte4, u8, 4, x, y, z, w);

// def_packed_vec_no_glam!(PackedUbyte2, u8, x, y);
// def_packed_vec_no_glam!(PackedUbyte3, u8, x, y, z);
// pub type PackUbyte4 = Ubyte4;

def_vec_no_glam!(Byte2, u8, 2, x, y);
def_vec_no_glam!(Byte3, u8, 4, x, y, z);
def_vec_no_glam!(Byte4, u8, 4, x, y, z, w);

// def_packed_vec_no_glam!(PackedByte2, u8, x, y);
// def_packed_vec_no_glam!(PackedByte3, u8, x, y, z);
// pub type PackByte4 = Byte4;

#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(8))]
pub struct Mat2 {
    pub cols: [Float2; 2],
}
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
pub struct Mat3 {
    pub cols: [Float3; 3],
}
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
pub struct Mat4 {
    pub cols: [Float4; 4],
}
impl Mat2 {
    pub const fn from_cols(c0: Float2, c1: Float2) -> Self {
        Self { cols: [c0, c1] }
    }
    pub const fn identity() -> Self {
        Self::from_cols(Float2::new(1.0, 0.0), Float2::new(0.0, 1.0))
    }
}
impl Mat3 {
    pub const fn from_cols(c0: Float3, c1: Float3, c2: Float3) -> Self {
        Self { cols: [c0, c1, c2] }
    }
    pub const fn identity() -> Self {
        Self::from_cols(
            Float3::new(1.0, 0.0, 0.0),
            Float3::new(0.0, 1.0, 0.0),
            Float3::new(0.0, 0.0, 1.0),
        )
    }
}
impl Mat4 {
    pub const fn from_cols(c0: Float4, c1: Float4, c2: Float4, c3: Float4) -> Self {
        Self {
            cols: [c0, c1, c2, c3],
        }
    }
    pub const fn identity() -> Self {
        Self::from_cols(
            Float4::new(1.0, 0.0, 0.0, 0.0),
            Float4::new(0.0, 1.0, 0.0, 0.0),
            Float4::new(0.0, 0.0, 1.0, 0.0),
            Float4::new(0.0, 0.0, 0.0, 1.0),
        )
    }
    pub fn into_affine3x4(self) -> [f32; 12] {
        // [
        //     self.cols[0].x,
        //     self.cols[0].y,
        //     self.cols[0].z,
        //     self.cols[1].x,
        //     self.cols[1].y,
        //     self.cols[1].z,
        //     self.cols[2].x,
        //     self.cols[2].y,
        //     self.cols[2].z,
        //     self.cols[3].x,
        //     self.cols[3].y,
        //     self.cols[3].z,
        // ]
        [
            self.cols[0].x,
            self.cols[1].x,
            self.cols[2].x,
            self.cols[3].x,
            self.cols[0].y,
            self.cols[1].y,
            self.cols[2].y,
            self.cols[3].y,
            self.cols[0].z,
            self.cols[1].z,
            self.cols[2].z,
            self.cols[3].z,
        ]
    }
}
impl From<Mat2> for glam::Mat2 {
    #[inline]
    fn from(m: Mat2) -> Self {
        Self::from_cols(m.cols[0].into(), m.cols[1].into())
    }
}
impl From<Mat3> for glam::Mat3 {
    #[inline]
    fn from(m: Mat3) -> Self {
        Self::from_cols(m.cols[0].into(), m.cols[1].into(), m.cols[2].into())
    }
}
impl From<Mat4> for glam::Mat4 {
    #[inline]
    fn from(m: Mat4) -> Self {
        Self::from_cols(
            m.cols[0].into(),
            m.cols[1].into(),
            m.cols[2].into(),
            m.cols[3].into(),
        )
    }
}
impl From<glam::Mat2> for Mat2 {
    #[inline]
    fn from(m: glam::Mat2) -> Self {
        Self {
            cols: [m.x_axis.into(), m.y_axis.into()],
        }
    }
}
impl From<glam::Mat3> for Mat3 {
    #[inline]
    fn from(m: glam::Mat3) -> Self {
        Self {
            cols: [m.x_axis.into(), m.y_axis.into(), m.z_axis.into()],
        }
    }
}
impl From<glam::Mat4> for Mat4 {
    #[inline]
    fn from(m: glam::Mat4) -> Self {
        Self {
            cols: [
                m.x_axis.into(),
                m.y_axis.into(),
                m.z_axis.into(),
                m.w_axis.into(),
            ],
        }
    }
}

macro_rules! impl_proxy_fields {
    ($vec:ident, $proxy:ident, $scalar:ty, x) => {
        impl $proxy {
            #[inline]
            pub fn x(&self) -> Expr<$scalar> {
                FromNode::from_node(__extract::<$scalar>(self.node, 0))
            }
            #[inline]
            pub fn set_x(&self, value: Expr<$scalar>) -> Self {
                Self::from_node(__insert::<$vec>(self.node, 0, FromNode::node(&value)))
            }
        }
    };
    ($vec:ident,$proxy:ident, $scalar:ty, y) => {
        impl $proxy {
            #[inline]
            pub fn y(&self) -> Expr<$scalar> {
                FromNode::from_node(__extract::<$scalar>(self.node, 1))
            }
            #[inline]
            pub fn set_y(&self, value: Expr<$scalar>) -> Self {
                Self::from_node(__insert::<$vec>(self.node, 1, FromNode::node(&value)))
            }
        }
    };
    ($vec:ident,$proxy:ident, $scalar:ty, z) => {
        impl $proxy {
            #[inline]
            pub fn z(&self) -> Expr<$scalar> {
                FromNode::from_node(__extract::<$scalar>(self.node, 2))
            }
            #[inline]
            pub fn set_z(&self, value: Expr<$scalar>) -> Self {
                Self::from_node(__insert::<$vec>(self.node, 2, FromNode::node(&value)))
            }
        }
    };
    ($vec:ident,$proxy:ident, $scalar:ty, w) => {
        impl $proxy {
            #[inline]
            pub fn w(&self) -> Expr<$scalar> {
                FromNode::from_node(__extract::<$scalar>(self.node, 3))
            }
            #[inline]
            pub fn set_w(&self, value: Expr<$scalar>) -> Self {
                Self::from_node(__insert::<$vec>(self.node, 3, FromNode::node(&value)))
            }
        }
    };
}
macro_rules! impl_var_proxy_fields {
    ($proxy:ident, $scalar:ty, x) => {
        impl $proxy {
            #[inline]
            pub fn x(&self) -> Var<$scalar> {
                FromNode::from_node(__extract::<$scalar>(self.node, 0))
            }
        }
    };
    ($proxy:ident, $scalar:ty, y) => {
        impl $proxy {
            #[inline]
            pub fn y(&self) -> Var<$scalar> {
                FromNode::from_node(__extract::<$scalar>(self.node, 1))
            }
        }
    };
    ($proxy:ident, $scalar:ty, z) => {
        impl $proxy {
            #[inline]
            pub fn z(&self) -> Var<$scalar> {
                FromNode::from_node(__extract::<$scalar>(self.node, 2))
            }
        }
    };
    ($proxy:ident, $scalar:ty, w) => {
        impl $proxy {
            #[inline]
            pub fn w(&self) -> Var<$scalar> {
                FromNode::from_node(__extract::<$scalar>(self.node, 3))
            }
        }
    };
}
macro_rules! impl_vec_proxy {
    ($vec:ident, $expr_proxy:ident, $var_proxy:ident, $scalar:ty, $scalar_ty:ident, $length:literal, $($comp:ident), *) => {
        #[derive(Clone, Copy)]
        pub struct $expr_proxy {
            node: NodeRef,
        }
        #[derive(Clone, Copy)]
        pub struct $var_proxy {
            node: NodeRef,
        }
        impl Value for $vec {
            type Expr = $expr_proxy;
            type Var = $var_proxy;
            fn fields() -> Vec<String> {
                vec![$(stringify!($comp).to_string()),*]
            }
        }
        impl TypeOf for $vec {
            fn type_() -> luisa_compute_ir::CArc<luisa_compute_ir::ir::Type> {
                let type_ = Type::Vector(VectorType {
                    element: VectorElementType::Scalar(Primitive::$scalar_ty),
                    length: $length,
                });
                register_type(type_)
            }
        }
        impl Aggregate for $expr_proxy {
            fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
                nodes.push(self.node);
            }
            fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
                Self {
                    node: iter.next().unwrap(),
                }
            }
        }
        impl Aggregate for $var_proxy {
            fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
                nodes.push(self.node);
            }
            fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
                Self {
                    node: iter.next().unwrap(),
                }
            }
        }
        impl FromNode for $expr_proxy {
            fn from_node(node: NodeRef) -> Self {
                Self { node }
            }
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        impl FromNode for $var_proxy {
            fn from_node(node: NodeRef) -> Self {
                Self { node }
            }
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        impl ExprProxy for $expr_proxy {
            type Value = $vec;
        }
        impl VarProxy for $var_proxy {
            type Value = $vec;
        }
        impl From<$var_proxy> for $expr_proxy {
            fn from(var: $var_proxy) -> Self {
                var.load()
            }
        }
        $(impl_proxy_fields!($vec, $expr_proxy, $scalar, $comp);)*
        $(impl_var_proxy_fields!($var_proxy, $scalar, $comp);)*
        impl $expr_proxy {
            #[inline]
            pub fn new($($comp: Expr<$scalar>), *) -> Self {
                Self {
                    node: __compose::<$vec>(&[$(FromNode::node(&$comp)), *]),
                }
            }
            pub fn at(&self, index: usize) -> Expr<$scalar> {
                FromNode::from_node(__extract::<$scalar>(self.node, index))
            }
        }
    };
}

macro_rules! impl_mat_proxy {
    ($mat:ident, $expr_proxy:ident, $var_proxy:ident, $vec:ty, $scalar_ty:ident, $length:literal, $($comp:ident), *) => {
        #[derive(Clone, Copy)]
        pub struct $expr_proxy {
            node: NodeRef,
        }
        #[derive(Clone, Copy)]
        pub struct $var_proxy {
            node: NodeRef,
        }
        impl Value for $mat {
            type Expr = $expr_proxy;
            type Var = $var_proxy;
            fn fields() -> Vec<String> {
                vec![$(stringify!($comp).to_string()),*]
            }
        }
        impl TypeOf for $mat {
            fn type_() -> luisa_compute_ir::CArc<luisa_compute_ir::ir::Type> {
                let type_ = Type::Matrix(MatrixType {
                    element: VectorElementType::Scalar(Primitive::$scalar_ty),
                    dimension: $length,
                });
                register_type(type_)
            }
        }
        impl Aggregate for $expr_proxy {
            fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
                nodes.push(self.node);
            }
            fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
                Self {
                    node: iter.next().unwrap(),
                }
            }
        }
        impl Aggregate for $var_proxy {
            fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
                nodes.push(self.node);
            }
            fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
                Self {
                    node: iter.next().unwrap(),
                }
            }
        }
        impl FromNode for $expr_proxy {
            fn from_node(node: NodeRef) -> Self {
                Self { node }
            }
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        impl ExprProxy for $expr_proxy {
            type Value = $mat;
        }
        impl FromNode for $var_proxy {
            fn from_node(node: NodeRef) -> Self {
                Self { node }
            }
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        impl VarProxy for $var_proxy {
            type Value = $mat;
        }
        impl From<$var_proxy> for $expr_proxy {
            fn from(var: $var_proxy) -> Self {
                var.load()
            }
        }
        impl $expr_proxy {
            #[inline]
            pub fn new($($comp: Expr<$vec>), *) -> Self {
                Self {
                    node: __compose::<$mat>(&[$(FromNode::node(&$comp)), *]),
                }
            }
            pub fn col(&self, index: usize) -> Expr<$vec> {
                Expr::<$vec>::from_node(__extract::<$vec>(self.node, index))
            }
        }
    };
}

impl_vec_proxy!(Bool2, Bool2Expr, Bool2Var, bool, Bool, 2, x, y);
impl_vec_proxy!(Bool3, Bool3Expr, Bool3Var, bool, Bool, 3, x, y, z);
impl_vec_proxy!(Bool4, Bool4Expr, Bool4Var, bool, Bool, 4, x, y, z, w);

impl_vec_proxy!(Float2, Float2Expr, Float2Var, f32, Float32, 2, x, y);
impl_vec_proxy!(Float3, Float3Expr, Float3Var, f32, Float32, 3, x, y, z);
impl_vec_proxy!(Float4, Float4Expr, Float4Var, f32, Float32, 4, x, y, z, w);

impl_vec_proxy!(Ushort2, Ushort2Expr, Ushort2Var, u16, Uint16, 2, x, y);
impl_vec_proxy!(Ushort3, Ushort3Expr, Ushort3Var, u16, Uint16, 3, x, y, z);
impl_vec_proxy!(Ushort4, Ushort4Expr, Ushort4Var, u16, Uint16, 4, x, y, z, w);

impl_vec_proxy!(Short2, Short2Expr, Short2Var, i16, Int16, 2, x, y);
impl_vec_proxy!(Short3, Short3Expr, Short3Var, i16, Int16, 3, x, y, z);
impl_vec_proxy!(Short4, Short4Expr, Short4Var, i16, Int16, 4, x, y, z, w);

impl_vec_proxy!(Uint2, Uint2Expr, Uint2Var, u32, Uint32, 2, x, y);
impl_vec_proxy!(Uint3, Uint3Expr, Uint3Var, u32, Uint32, 3, x, y, z);
impl_vec_proxy!(Uint4, Uint4Expr, Uint4Var, u32, Uint32, 4, x, y, z, w);

impl_vec_proxy!(Int2, Int2Expr, Int2Var, i32, Int32, 2, x, y);
impl_vec_proxy!(Int3, Int3Expr, Int3Var, i32, Int32, 3, x, y, z);
impl_vec_proxy!(Int4, Int4Expr, Int4Var, i32, Int32, 4, x, y, z, w);

impl_vec_proxy!(Ulong2, Ulong2Expr, Ulong2Var, u64, Uint64, 2, x, y);
impl_vec_proxy!(Ulong3, Ulong3Expr, Ulong3Var, u64, Uint64, 3, x, y, z);
impl_vec_proxy!(Ulong4, Ulong4Expr, Ulong4Var, u64, Uint64, 4, x, y, z, w);

impl_vec_proxy!(Long2, Long2Expr, Long2Var, i64, Int64, 2, x, y);
impl_vec_proxy!(Long3, Long3Expr, Long3Var, i64, Int64, 3, x, y, z);
impl_vec_proxy!(Long4, Long4Expr, Long4Var, i64, Int64, 4, x, y, z, w);

impl_mat_proxy!(Mat2, Mat2Expr, Mat2Var, Float2, Float32, 2, x, y);
impl_mat_proxy!(Mat3, Mat3Expr, Mat3Var, Float3, Float32, 3, x, y, z);
impl_mat_proxy!(Mat4, Mat4Expr, Mat4Var, Float4, Float32, 4, x, y, z, w);

macro_rules! impl_packed_cvt {
    ($packed:ty, $vec:ty, $($comp:ident), *) => {
        impl From<$vec> for $packed {
            fn from(v: $vec) -> Self {
                Self::new($(v.$comp()), *)
            }
        }
        impl From<$packed> for $vec {
            fn from(v: $packed) -> Self {
                Self::new($(v.$comp()), *)
            }
        }
    }
}
impl_packed_cvt!(PackedFloat2Expr, Float2Expr, x, y);
impl_packed_cvt!(PackedFloat3Expr, Float3Expr, x, y, z);
impl_packed_cvt!(PackedFloat4Expr, Float4Expr, x, y, z, w);

impl_packed_cvt!(PackedShort2Expr, Short2Expr, x, y);
impl_packed_cvt!(PackedShort3Expr, Short3Expr, x, y, z);
impl_packed_cvt!(PackedShort4Expr, Short4Expr, x, y, z, w);

// ushort
impl_packed_cvt!(PackedUshort2Expr, Ushort2Expr, x, y);
impl_packed_cvt!(PackedUshort3Expr, Ushort3Expr, x, y, z);
impl_packed_cvt!(PackedUshort4Expr, Ushort4Expr, x, y, z, w);

// int
impl_packed_cvt!(PackedInt2Expr, Int2Expr, x, y);
impl_packed_cvt!(PackedInt3Expr, Int3Expr, x, y, z);
impl_packed_cvt!(PackedInt4Expr, Int4Expr, x, y, z, w);

// uint
impl_packed_cvt!(PackedUint2Expr, Uint2Expr, x, y);
impl_packed_cvt!(PackedUint3Expr, Uint3Expr, x, y, z);
impl_packed_cvt!(PackedUint4Expr, Uint4Expr, x, y, z, w);

// long
impl_packed_cvt!(PackedLong2Expr, Long2Expr, x, y);
impl_packed_cvt!(PackedLong3Expr, Long3Expr, x, y, z);
impl_packed_cvt!(PackedLong4Expr, Long4Expr, x, y, z, w);

// ulong
impl_packed_cvt!(PackedUlong2Expr, Ulong2Expr, x, y);
impl_packed_cvt!(PackedUlong3Expr, Ulong3Expr, x, y, z);
impl_packed_cvt!(PackedUlong4Expr, Ulong4Expr, x, y, z, w);

macro_rules! impl_binop {
    ($t:ty, $scalar:ty, $proxy:ty, $tr:ident, $m:ident, $tr_assign:ident, $m_assign:ident) => {
        impl std::ops::$tr_assign<$proxy> for $proxy {
            fn $m_assign(&mut self, rhs: $proxy) {
                use std::ops::$tr;
                *self = (*self).$m(rhs);
            }
        }
        impl std::ops::$tr for $proxy {
            type Output = $proxy;
            fn $m(self, rhs: $proxy) -> Self::Output {
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::$tr, &[self.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        impl std::ops::$tr<$scalar> for $proxy {
            type Output = $proxy;
            fn $m(self, rhs: $scalar) -> Self::Output {
                let rhs = Self::splat(rhs);
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::$tr, &[self.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        impl std::ops::$tr<$proxy> for $scalar {
            type Output = $proxy;
            fn $m(self, rhs: $proxy) -> Self::Output {
                let lhs = <$proxy>::splat(self);
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::$tr, &[lhs.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        impl std::ops::$tr<PrimExpr<$scalar>> for $proxy {
            type Output = $proxy;
            fn $m(self, rhs: PrimExpr<$scalar>) -> Self::Output {
                let rhs = Self::splat(rhs);
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::$tr, &[self.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        impl std::ops::$tr<$proxy> for PrimExpr<$scalar> {
            type Output = $proxy;
            fn $m(self, rhs: $proxy) -> Self::Output {
                let lhs = <$proxy>::splat(self);
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::$tr, &[lhs.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
    };
}
macro_rules! impl_binop_for_mat {
    ($t:ty, $scalar:ty, $proxy:ty, $tr:ident, $m:ident, $tr_assign:ident, $m_assign:ident) => {
        impl std::ops::$tr_assign<$proxy> for $proxy {
            fn $m_assign(&mut self, rhs: $proxy) {
                use std::ops::$tr;
                *self = (*self).$m(rhs);
            }
        }
        impl std::ops::$tr for $proxy {
            type Output = $proxy;
            fn $m(self, rhs: $proxy) -> Self::Output {
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::$tr, &[self.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        impl std::ops::$tr<$scalar> for $proxy {
            type Output = $proxy;
            fn $m(self, rhs: $scalar) -> Self::Output {
                let rhs = Self::fill(rhs);
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::$tr, &[self.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        impl std::ops::$tr<$proxy> for $scalar {
            type Output = $proxy;
            fn $m(self, rhs: $proxy) -> Self::Output {
                let lhs = <$proxy>::fill(self);
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::$tr, &[lhs.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        impl std::ops::$tr<PrimExpr<$scalar>> for $proxy {
            type Output = $proxy;
            fn $m(self, rhs: PrimExpr<$scalar>) -> Self::Output {
                let rhs = Self::fill(rhs);
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::$tr, &[self.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        impl std::ops::$tr<$proxy> for PrimExpr<$scalar> {
            type Output = $proxy;
            fn $m(self, rhs: $proxy) -> Self::Output {
                let lhs = <$proxy>::fill(self);
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::$tr, &[lhs.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
    }
}
macro_rules! impl_arith_binop {
    ($t:ty, $scalar:ty, $proxy:ty) => {
        impl_common_op!($t, $scalar, $proxy);
        impl_binop!($t, $scalar, $proxy, Add, add, AddAssign, add_assign);
        impl_binop!($t, $scalar, $proxy, Sub, sub, SubAssign, sub_assign);
        impl_binop!($t, $scalar, $proxy, Mul, mul, MulAssign, mul_assign);
        impl_binop!($t, $scalar, $proxy, Div, div, DivAssign, div_assign);
        impl_binop!($t, $scalar, $proxy, Rem, rem, RemAssign, rem_assign);
        impl_reduce!($t, $scalar, $proxy);
        impl std::ops::Neg for $proxy {
            type Output = $proxy;
            fn neg(self) -> Self::Output {
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::Neg, &[self.node], <$t as TypeOf>::type_())
                }))
            }
        }
    };
}
macro_rules! impl_arith_binop_for_mat {
    ($t:ty, $scalar:ty, $proxy:ty) => {
        impl_binop_for_mat!($t, $scalar, $proxy, Add, add, AddAssign, add_assign);
        impl_binop_for_mat!($t, $scalar, $proxy, Sub, sub, SubAssign, sub_assign);
        impl_binop_for_mat!($t, $scalar, $proxy, Mul, mul, MulAssign, mul_assign);
        impl_binop_for_mat!($t, $scalar, $proxy, Div, div, DivAssign, div_assign);
        impl_binop_for_mat!($t, $scalar, $proxy, Rem, rem, RemAssign, rem_assign);
        impl std::ops::Neg for $proxy {
            type Output = $proxy;
            fn neg(self) -> Self::Output {
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::Neg, &[self.node], <$t as TypeOf>::type_())
                }))
            }
        }
        impl $proxy {
            pub fn comp_mul(&self, other:Self)->Self {
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::MatCompMul, &[self.node, other.node], <$t as TypeOf>::type_())
                }))
            }
        }
    }
}
macro_rules! impl_int_binop {
    ($t:ty, $scalar:ty, $proxy:ty) => {
        impl_binop!(
            $t,
            $scalar,
            $proxy,
            BitAnd,
            bitand,
            BitAndAssign,
            bitand_assign
        );
        impl_binop!($t, $scalar, $proxy, BitOr, bitor, BitOrAssign, bitor_assign);
        impl_binop!(
            $t,
            $scalar,
            $proxy,
            BitXor,
            bitxor,
            BitXorAssign,
            bitxor_assign
        );
        impl_binop!($t, $scalar, $proxy, Shl, shl, ShlAssign, shl_assign);
        impl_binop!($t, $scalar, $proxy, Shr, shr, ShrAssign, shr_assign);
        impl std::ops::Not for $proxy {
            type Output = Expr<$t>;
            fn not(self) -> Self::Output {
                __current_scope(|s| {
                    let ret = s.call(
                        Func::BitNot,
                        &[FromNode::node(&self)],
                        Self::Output::type_(),
                    );
                    Expr::<$t>::from_node(ret)
                })
            }
        }
    };
}
macro_rules! impl_bool_binop {
    ($t:ty, $proxy:ty) => {
        impl_binop!(
            $t,
            bool,
            $proxy,
            BitAnd,
            bitand,
            BitAndAssign,
            bitand_assign
        );
        impl_binop!($t, bool, $proxy, BitOr, bitor, BitOrAssign, bitor_assign);
        impl_binop!(
            $t,
            bool,
            $proxy,
            BitXor,
            bitxor,
            BitXorAssign,
            bitxor_assign
        );
        impl $proxy {
            pub fn splat<V: Into<PrimExpr<bool>>>(value: V) -> Self {
                let value = value.into();
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::Vec, &[value.node], <$t as TypeOf>::type_())
                }))
            }
            pub fn zero() -> Self {
                Self::splat(false)
            }
            pub fn one() -> Self {
                Self::splat(true)
            }
            pub fn all(&self) -> Expr<bool> {
                Expr::<bool>::from_node(__current_scope(|s| {
                    s.call(Func::All, &[self.node], <bool as TypeOf>::type_())
                }))
            }
            pub fn any(&self) -> Expr<bool> {
                Expr::<bool>::from_node(__current_scope(|s| {
                    s.call(Func::Any, &[self.node], <bool as TypeOf>::type_())
                }))
            }
        }
        impl std::ops::Not for $proxy {
            type Output = Expr<$t>;
            fn not(self) -> Self::Output {
                self ^ Self::splat(true)
            }
        }
    };
}
macro_rules! impl_reduce {
    ($t:ty, $scalar:ty, $proxy:ty) => {
        impl $proxy {
            #[inline]
            pub fn reduce_sum(&self) -> Expr<$scalar> {
                FromNode::from_node(__current_scope(|s| {
                    s.call(Func::ReduceSum, &[self.node], <$scalar as TypeOf>::type_())
                }))
            }
            #[inline]
            pub fn reduce_prod(&self) -> Expr<$scalar> {
                FromNode::from_node(__current_scope(|s| {
                    s.call(Func::ReduceProd, &[self.node], <$scalar as TypeOf>::type_())
                }))
            }
            #[inline]
            pub fn reduce_min(&self) -> Expr<$scalar> {
                FromNode::from_node(__current_scope(|s| {
                    s.call(Func::ReduceMin, &[self.node], <$scalar as TypeOf>::type_())
                }))
            }
            #[inline]
            pub fn reduce_max(&self) -> Expr<$scalar> {
                FromNode::from_node(__current_scope(|s| {
                    s.call(Func::ReduceMax, &[self.node], <$scalar as TypeOf>::type_())
                }))
            }
            #[inline]
            pub fn dot(&self, rhs: $proxy) -> Expr<$scalar> {
                FromNode::from_node(__current_scope(|s| {
                    s.call(
                        Func::Dot,
                        &[self.node, rhs.node],
                        <$scalar as TypeOf>::type_(),
                    )
                }))
            }
        }
    };
}
macro_rules! impl_common_op {
    ($t:ty, $scalar:ty, $proxy:ty) => {
        impl $proxy {
            pub fn splat<V: Into<PrimExpr<$scalar>>>(value: V) -> Self {
                let value = value.into();
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::Vec, &[value.node], <$t as TypeOf>::type_())
                }))
            }
            pub fn zero() -> Self {
                Self::splat(0.0 as $scalar)
            }
            pub fn one() -> Self {
                Self::splat(1.0 as $scalar)
            }
        }
    };
}
macro_rules! impl_vec_op {
    ($t:ty, $scalar:ty, $proxy:ty, $mat:ty) => {
        impl $proxy {
            #[inline]
            pub fn length(&self) -> Expr<$scalar> {
                FromNode::from_node(__current_scope(|s| {
                    s.call(Func::Length, &[self.node], <$scalar as TypeOf>::type_())
                }))
            }
            #[inline]
            pub fn normalize(&self) -> Self {
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::Normalize, &[self.node], <$t as TypeOf>::type_())
                }))
            }
            #[inline]
            pub fn length_squared(&self) -> Expr<$scalar> {
                FromNode::from_node(__current_scope(|s| {
                    s.call(
                        Func::LengthSquared,
                        &[self.node],
                        <$scalar as TypeOf>::type_(),
                    )
                }))
            }
            #[inline]
            pub fn distance(&self, rhs: $proxy) -> Expr<$scalar> {
                (*self - rhs).length()
            }
            #[inline]
            pub fn distance_squared(&self, rhs: $proxy) -> Expr<$scalar> {
                (*self - rhs).length_squared()
            }
            #[inline]
            pub fn fma(&self, a: $proxy, b: $proxy) -> Self {
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(
                        Func::Fma,
                        &[self.node, a.node, b.node],
                        <$t as TypeOf>::type_(),
                    )
                }))
            }
            #[inline]
            pub fn outer_product(&self, rhs: $proxy) -> Expr<$mat> {
                Expr::<$mat>::from_node(__current_scope(|s| {
                    s.call(
                        Func::OuterProduct,
                        &[self.node, rhs.node],
                        <$mat as TypeOf>::type_(),
                    )
                }))
            }
        }
    };
}
impl_arith_binop!(Float2, f32, Float2Expr);
impl_arith_binop!(Float3, f32, Float3Expr);
impl_arith_binop!(Float4, f32, Float4Expr);

impl_arith_binop!(Short2, i16, Short2Expr);
impl_arith_binop!(Short3, i16, Short3Expr);
impl_arith_binop!(Short4, i16, Short4Expr);

impl_arith_binop!(Ushort2, u16, Ushort2Expr);
impl_arith_binop!(Ushort3, u16, Ushort3Expr);
impl_arith_binop!(Ushort4, u16, Ushort4Expr);

impl_arith_binop!(Int2, i32, Int2Expr);
impl_arith_binop!(Int3, i32, Int3Expr);
impl_arith_binop!(Int4, i32, Int4Expr);

impl_arith_binop!(Uint2, u32, Uint2Expr);
impl_arith_binop!(Uint3, u32, Uint3Expr);
impl_arith_binop!(Uint4, u32, Uint4Expr);

impl_arith_binop!(Long2, i64, Long2Expr);
impl_arith_binop!(Long3, i64, Long3Expr);
impl_arith_binop!(Long4, i64, Long4Expr);

impl_arith_binop!(Ulong2, u64, Ulong2Expr);
impl_arith_binop!(Ulong3, u64, Ulong3Expr);
impl_arith_binop!(Ulong4, u64, Ulong4Expr);

impl_int_binop!(Short2, i16, Short2Expr);
impl_int_binop!(Short3, i16, Short3Expr);
impl_int_binop!(Short4, i16, Short4Expr);

impl_int_binop!(Ushort2, u16, Ushort2Expr);
impl_int_binop!(Ushort3, u16, Ushort3Expr);
impl_int_binop!(Ushort4, u16, Ushort4Expr);

impl_int_binop!(Int2, i32, Int2Expr);
impl_int_binop!(Int3, i32, Int3Expr);
impl_int_binop!(Int4, i32, Int4Expr);

impl_int_binop!(Uint2, u32, Uint2Expr);
impl_int_binop!(Uint3, u32, Uint3Expr);
impl_int_binop!(Uint4, u32, Uint4Expr);

impl_int_binop!(Long2, i64, Long2Expr);
impl_int_binop!(Long3, i64, Long3Expr);
impl_int_binop!(Long4, i64, Long4Expr);

impl_int_binop!(Ulong2, u64, Ulong2Expr);
impl_int_binop!(Ulong3, u64, Ulong3Expr);
impl_int_binop!(Ulong4, u64, Ulong4Expr);

impl_bool_binop!(Bool2, Bool2Expr);
impl_bool_binop!(Bool3, Bool3Expr);
impl_bool_binop!(Bool4, Bool4Expr);

macro_rules! impl_select {
    ($bvec:ty, $vec:ty, $proxy:ty) => {
        impl $proxy {
            pub fn select(mask: Expr<$bvec>, a: Expr<$vec>, b: Expr<$vec>) -> Expr<$vec> {
                Expr::<$vec>::from_node(__current_scope(|s| {
                    s.call(
                        Func::Select,
                        &[mask.node(), a.node(), b.node()],
                        <$vec as TypeOf>::type_(),
                    )
                }))
            }
        }
    };
}

impl_select!(Bool2, Bool2, Bool2Expr);
impl_select!(Bool3, Bool3, Bool3Expr);
impl_select!(Bool4, Bool4, Bool4Expr);

impl_select!(Bool2, Float2, Float2Expr);
impl_select!(Bool3, Float3, Float3Expr);
impl_select!(Bool4, Float4, Float4Expr);

impl_select!(Bool2, Int2, Int2Expr);
impl_select!(Bool3, Int3, Int3Expr);
impl_select!(Bool4, Int4, Int4Expr);

impl_select!(Bool2, Uint2, Uint2Expr);
impl_select!(Bool3, Uint3, Uint3Expr);
impl_select!(Bool4, Uint4, Uint4Expr);

impl_select!(Bool2, Short2, Short2Expr);
impl_select!(Bool3, Short3, Short3Expr);
impl_select!(Bool4, Short4, Short4Expr);

impl_select!(Bool2, Ushort2, Ushort2Expr);
impl_select!(Bool3, Ushort3, Ushort3Expr);
impl_select!(Bool4, Ushort4, Ushort4Expr);

impl_select!(Bool2, Long2, Long2Expr);
impl_select!(Bool3, Long3, Long3Expr);
impl_select!(Bool4, Long4, Long4Expr);

impl_select!(Bool2, Ulong2, Ulong2Expr);
impl_select!(Bool3, Ulong3, Ulong3Expr);
impl_select!(Bool4, Ulong4, Ulong4Expr);

macro_rules! impl_permute {
    ($tr:ident, $proxy:ty,$len:expr, $v2:ty, $v3:ty, $v4:ty) => {
        impl $tr for $proxy {
            type Vec2 = Expr<$v2>;
            type Vec3 = Expr<$v3>;
            type Vec4 = Expr<$v4>;
            fn permute2(&self, x: i32, y: i32) -> Self::Vec2 {
                assert!(x < $len);
                assert!(y < $len);
                let x: Expr<i32> = x.into();
                let y: Expr<i32> = y.into();
                Expr::<$v2>::from_node(__current_scope(|s| {
                    s.call(
                        Func::Permute,
                        &[self.node, FromNode::node(&x), FromNode::node(&y)],
                        <$v2 as TypeOf>::type_(),
                    )
                }))
            }
            fn permute3(&self, x: i32, y: i32, z: i32) -> Self::Vec3 {
                assert!(x < $len);
                assert!(y < $len);
                assert!(z < $len);
                let x: Expr<i32> = x.into();
                let y: Expr<i32> = y.into();
                let z: Expr<i32> = z.into();
                Expr::<$v3>::from_node(__current_scope(|s| {
                    s.call(
                        Func::Permute,
                        &[
                            self.node,
                            FromNode::node(&x),
                            FromNode::node(&y),
                            FromNode::node(&z),
                        ],
                        <$v3 as TypeOf>::type_(),
                    )
                }))
            }
            fn permute4(&self, x: i32, y: i32, z: i32, w: i32) -> Self::Vec4 {
                assert!(x < $len);
                assert!(y < $len);
                assert!(z < $len);
                assert!(w < $len);
                let x: Expr<i32> = x.into();
                let y: Expr<i32> = y.into();
                let z: Expr<i32> = z.into();
                let w: Expr<i32> = w.into();
                Expr::<$v4>::from_node(__current_scope(|s| {
                    s.call(
                        Func::Permute,
                        &[
                            self.node,
                            FromNode::node(&x),
                            FromNode::node(&y),
                            FromNode::node(&z),
                            FromNode::node(&w),
                        ],
                        <$v4 as TypeOf>::type_(),
                    )
                }))
            }
        }
    };
}
impl_permute!(Vec2Swizzle, Float2Expr, 2, Float2, Float3, Float4);
impl_permute!(Vec3Swizzle, Float3Expr, 3, Float2, Float3, Float4);
impl_permute!(Vec4Swizzle, Float4Expr, 4, Float2, Float3, Float4);

impl_permute!(Vec2Swizzle, Short2Expr, 2, Short2, Short3, Short4);
impl_permute!(Vec3Swizzle, Short3Expr, 3, Short2, Short3, Short4);
impl_permute!(Vec4Swizzle, Short4Expr, 4, Short2, Short3, Short4);

impl_permute!(Vec2Swizzle, Ushort2Expr, 2, Ushort2, Ushort3, Ushort4);
impl_permute!(Vec3Swizzle, Ushort3Expr, 3, Ushort2, Ushort3, Ushort4);
impl_permute!(Vec4Swizzle, Ushort4Expr, 4, Ushort2, Ushort3, Ushort4);

impl_permute!(Vec2Swizzle, Int2Expr, 2, Int2, Int3, Int4);
impl_permute!(Vec3Swizzle, Int3Expr, 3, Int2, Int3, Int4);
impl_permute!(Vec4Swizzle, Int4Expr, 4, Int2, Int3, Int4);

impl_permute!(Vec2Swizzle, Uint2Expr, 2, Uint2, Uint3, Uint4);
impl_permute!(Vec3Swizzle, Uint3Expr, 3, Uint2, Uint3, Uint4);
impl_permute!(Vec4Swizzle, Uint4Expr, 4, Uint2, Uint3, Uint4);

impl_permute!(Vec2Swizzle, Long2Expr, 2, Long2, Long3, Long4);
impl_permute!(Vec3Swizzle, Long3Expr, 3, Long2, Long3, Long4);
impl_permute!(Vec4Swizzle, Long4Expr, 4, Long2, Long3, Long4);

impl_permute!(Vec2Swizzle, Ulong2Expr, 2, Ulong2, Ulong3, Ulong4);
impl_permute!(Vec3Swizzle, Ulong3Expr, 3, Ulong2, Ulong3, Ulong4);
impl_permute!(Vec4Swizzle, Ulong4Expr, 4, Ulong2, Ulong3, Ulong4);

impl Float3Expr {
    #[inline]
    pub fn cross(&self, rhs: Float3Expr) -> Self {
        Float3Expr::from_node(__current_scope(|s| {
            s.call(
                Func::Cross,
                &[self.node, rhs.node],
                <Float3 as TypeOf>::type_(),
            )
        }))
    }
}
impl_vec_op!(Float2, f32, Float2Expr, Mat2);
impl_vec_op!(Float3, f32, Float3Expr, Mat3);
impl_vec_op!(Float4, f32, Float4Expr, Mat4);

macro_rules! impl_var_trait2 {
    ($t:ty, $v:ty) => {
        impl VarTrait for $t {
            type Value = $v;
            type Short = Short2Expr;
            type Ushort = Ushort2Expr;
            type Int = Int2Expr;
            type Uint = Uint2Expr;
            type Float = Float2Expr;
            type Bool = Bool2Expr;
            // type Double = Double2Expr;
            type Long = Long2Expr;
            type Ulong = Ulong2Expr;
        }
        impl CommonVarOp for $t {}
        impl VarCmp for $t {}
        impl VarCmpEq for $t {}
        impl From<$v> for $t {
            fn from(v: $v) -> Self {
                Self::new(const_(v.x), const_(v.y))
            }
        }
    };
}
macro_rules! impl_var_trait3 {
    ($t:ty, $v:ty) => {
        impl VarTrait for $t {
            type Value = $v;
            type Short = Short3Expr;
            type Ushort = Ushort3Expr;
            type Int = Int3Expr;
            type Uint = Uint3Expr;
            type Float = Float3Expr;
            type Bool = Bool3Expr;
            type Long = Long3Expr;
            type Ulong = Ulong3Expr;
        }
        impl CommonVarOp for $t {}
        impl VarCmp for $t {}
        impl VarCmpEq for $t {}
        impl From<$v> for $t {
            fn from(v: $v) -> Self {
                Self::new(const_(v.x), const_(v.y), const_(v.z))
            }
        }
    };
}
macro_rules! impl_var_trait4 {
    ($t:ty, $v:ty) => {
        impl VarTrait for $t {
            type Value = $v;
            type Short = Short4Expr;
            type Ushort = Ushort4Expr;
            type Int = Int2Expr;
            type Uint = Uint2Expr;
            type Float = Float2Expr;
            type Bool = Bool2Expr;
            type Long = Long2Expr;
            type Ulong = Ulong2Expr;
        }
        impl CommonVarOp for $t {}
        impl VarCmp for $t {}
        impl VarCmpEq for $t {}
        impl From<$v> for $t {
            fn from(v: $v) -> Self {
                Self::new(const_(v.x), const_(v.y), const_(v.z), const_(v.w))
            }
        }
    };
}
impl_var_trait2!(Float2Expr, Float2);
impl_var_trait2!(Short2Expr, Short2);
impl_var_trait2!(Ushort2Expr, Ushort2);
impl_var_trait2!(Int2Expr, Int2);
impl_var_trait2!(Uint2Expr, Uint2);
impl_var_trait2!(Bool2Expr, Bool2);
impl_var_trait2!(Long2Expr, Long2);
impl_var_trait2!(Ulong2Expr, Ulong2);

impl_var_trait3!(Float3Expr, Float3);
impl_var_trait3!(Short3Expr, Short3);
impl_var_trait3!(Ushort3Expr, Ushort3);
impl_var_trait3!(Int3Expr, Int3);
impl_var_trait3!(Uint3Expr, Uint3);
impl_var_trait3!(Bool3Expr, Bool3);
impl_var_trait3!(Long3Expr, Long3);
impl_var_trait3!(Ulong3Expr, Ulong3);

impl_var_trait4!(Float4Expr, Float4);
impl_var_trait4!(Short4Expr, Short4);
impl_var_trait4!(Ushort4Expr, Ushort4);
impl_var_trait4!(Int4Expr, Int4);
impl_var_trait4!(Uint4Expr, Uint4);
impl_var_trait4!(Bool4Expr, Bool4);
impl_var_trait4!(Long4Expr, Long4);
impl_var_trait4!(Ulong4Expr, Ulong4);

macro_rules! impl_float_trait {
    ($t:ty) => {
        impl From<f32> for $t {
            fn from(v: f32) -> Self {
                Self::splat(v)
            }
        }
        impl FloatVarTrait for $t {}
    };
}
impl_float_trait!(Float2Expr);
impl_float_trait!(Float3Expr);
impl_float_trait!(Float4Expr);
macro_rules! impl_int_trait {
    ($t:ty) => {
        impl From<i64> for $t {
            fn from(v: i64) -> Self {
                Self::splat(v)
            }
        }
        impl IntVarTrait for $t {}
    };
}
impl_int_trait!(Int2Expr);
impl_int_trait!(Int3Expr);
impl_int_trait!(Int4Expr);
impl_int_trait!(Long2Expr);
impl_int_trait!(Long3Expr);
impl_int_trait!(Long4Expr);
impl_int_trait!(Uint2Expr);
impl_int_trait!(Uint3Expr);
impl_int_trait!(Uint4Expr);
impl_int_trait!(Ulong2Expr);
impl_int_trait!(Ulong3Expr);
impl_int_trait!(Ulong4Expr);

impl Mul<Float2Expr> for Mat2Expr {
    type Output = Float2Expr;
    #[inline]
    fn mul(self, rhs: Float2Expr) -> Self::Output {
        Float2Expr::from_node(__current_scope(|s| {
            s.call(
                Func::Mul,
                &[self.node, rhs.node],
                <Float2 as TypeOf>::type_(),
            )
        }))
    }
}
impl Mat2Expr {
    pub fn fill(e: impl Into<Expr<f32>> + Copy) -> Self {
        Self::new(make_float2(e, e), make_float2(e, e))
    }
    pub fn eye(e: Expr<Float2>) -> Self {
        Self::new(make_float2(e.x(), 0.0), make_float2(0.0, e.y()))
    }
    pub fn inverse(&self) -> Self {
        Mat2Expr::from_node(__current_scope(|s| {
            s.call(Func::Inverse, &[self.node], <Mat2 as TypeOf>::type_())
        }))
    }
    pub fn transpose(&self) -> Self {
        Mat2Expr::from_node(__current_scope(|s| {
            s.call(Func::Transpose, &[self.node], <Mat2 as TypeOf>::type_())
        }))
    }
    pub fn determinant(&self) -> Float {
        FromNode::from_node(__current_scope(|s| {
            s.call(Func::Determinant, &[self.node], <f32 as TypeOf>::type_())
        }))
    }
}
impl_arith_binop_for_mat!(Mat2, f32, Mat2Expr);
impl Mul<Float3Expr> for Mat3Expr {
    type Output = Float3Expr;
    #[inline]
    fn mul(self, rhs: Float3Expr) -> Self::Output {
        Float3Expr::from_node(__current_scope(|s| {
            s.call(
                Func::Mul,
                &[self.node, rhs.node],
                <Float3 as TypeOf>::type_(),
            )
        }))
    }
}
impl Mat3Expr {
    pub fn fill(e: impl Into<PrimExpr<f32>> + Copy) -> Self {
        Self::new(
            make_float3(e, e, e),
            make_float3(e, e, e),
            make_float3(e, e, e),
        )
    }
    pub fn eye(e: Expr<Float3>) -> Self {
        Self::new(
            make_float3(e.x(), 0.0, 0.0),
            make_float3(0.0, e.y(), 0.0),
            make_float3(0.0, 0.0, e.z()),
        )
    }
    pub fn inverse(&self) -> Self {
        Self::from_node(__current_scope(|s| {
            s.call(Func::Inverse, &[self.node], <Mat3 as TypeOf>::type_())
        }))
    }
    pub fn transpose(&self) -> Self {
        Self::from_node(__current_scope(|s| {
            s.call(Func::Transpose, &[self.node], <Mat3 as TypeOf>::type_())
        }))
    }
    pub fn determinant(&self) -> Float {
        FromNode::from_node(__current_scope(|s| {
            s.call(Func::Determinant, &[self.node], <f32 as TypeOf>::type_())
        }))
    }
}
impl Mul<Float4Expr> for Mat4Expr {
    type Output = Float4Expr;
    #[inline]
    fn mul(self, rhs: Float4Expr) -> Self::Output {
        Float4Expr::from_node(__current_scope(|s| {
            s.call(
                Func::Mul,
                &[self.node, rhs.node],
                <Float4 as TypeOf>::type_(),
            )
        }))
    }
}
impl_arith_binop_for_mat!(Mat3, f32, Mat3Expr);
impl Mat4Expr {
    pub fn fill(e: impl Into<PrimExpr<f32>> + Copy) -> Self {
        Self::new(
            make_float4(e, e, e, e),
            make_float4(e, e, e, e),
            make_float4(e, e, e, e),
            make_float4(e, e, e, e),
        )
    }
    pub fn eye(e: Expr<Float4>) -> Self {
        Self::new(
            make_float4(e.x(), 0.0, 0.0, 0.0),
            make_float4(0.0, e.y(), 0.0, 0.0),
            make_float4(0.0, 0.0, e.z(), 0.0),
            make_float4(0.0, 0.0, 0.0, e.w()),
        )
    }
    pub fn inverse(&self) -> Self {
        Self::from_node(__current_scope(|s| {
            s.call(Func::Inverse, &[self.node], <Mat4 as TypeOf>::type_())
        }))
    }
    pub fn transpose(&self) -> Self {
        Self::from_node(__current_scope(|s| {
            s.call(Func::Transpose, &[self.node], <Mat4 as TypeOf>::type_())
        }))
    }
    pub fn determinant(&self) -> Float {
        FromNode::from_node(__current_scope(|s| {
            s.call(Func::Determinant, &[self.node], <f32 as TypeOf>::type_())
        }))
    }
}
impl_arith_binop_for_mat!(Mat4, f32, Mat4Expr);
#[inline]
pub fn make_float2<X: Into<PrimExpr<f32>>, Y: Into<PrimExpr<f32>>>(x: X, y: Y) -> Expr<Float2> {
    Expr::<Float2>::new(x.into(), y.into())
}
#[inline]
pub fn make_float3<X: Into<PrimExpr<f32>>, Y: Into<PrimExpr<f32>>, Z: Into<PrimExpr<f32>>>(
    x: X,
    y: Y,
    z: Z,
) -> Expr<Float3> {
    Expr::<Float3>::new(x.into(), y.into(), z.into())
}
#[inline]
pub fn make_float4<
    X: Into<PrimExpr<f32>>,
    Y: Into<PrimExpr<f32>>,
    Z: Into<PrimExpr<f32>>,
    W: Into<PrimExpr<f32>>,
>(
    x: X,
    y: Y,
    z: Z,
    w: W,
) -> Expr<Float4> {
    Expr::<Float4>::new(x.into(), y.into(), z.into(), w.into())
}
#[inline]
pub fn make_float2x2<X: Into<Expr<Float2>>, Y: Into<Expr<Float2>>>(x: X, y: Y) -> Expr<Mat2> {
    Expr::<Mat2>::new(x.into(), y.into())
}
#[inline]
pub fn make_float3x3<X: Into<Expr<Float3>>, Y: Into<Expr<Float3>>, Z: Into<Expr<Float3>>>(
    x: X,
    y: Y,
    z: Z,
) -> Expr<Mat3> {
    Expr::<Mat3>::new(x.into(), y.into(), z.into())
}
#[inline]
pub fn make_float4x4<
    X: Into<Expr<Float4>>,
    Y: Into<Expr<Float4>>,
    Z: Into<Expr<Float4>>,
    W: Into<Expr<Float4>>,
>(
    x: X,
    y: Y,
    z: Z,
    w: W,
) -> Expr<Mat4> {
    Expr::<Mat4>::new(x.into(), y.into(), z.into(), w.into())
}

#[inline]
pub fn make_int2<X: Into<PrimExpr<i32>>, Y: Into<PrimExpr<i32>>>(x: X, y: Y) -> Expr<Int2> {
    Expr::<Int2>::new(x.into(), y.into())
}
#[inline]
pub fn make_int3<X: Into<PrimExpr<i32>>, Y: Into<PrimExpr<i32>>, Z: Into<PrimExpr<i32>>>(
    x: X,
    y: Y,
    z: Z,
) -> Expr<Int3> {
    Expr::<Int3>::new(x.into(), y.into(), z.into())
}
#[inline]
pub fn make_int4<
    X: Into<PrimExpr<i32>>,
    Y: Into<PrimExpr<i32>>,
    Z: Into<PrimExpr<i32>>,
    W: Into<PrimExpr<i32>>,
>(
    x: X,
    y: Y,
    z: Z,
    w: W,
) -> Expr<Int4> {
    Expr::<Int4>::new(x.into(), y.into(), z.into(), w.into())
}
#[inline]
pub fn make_uint2<X: Into<PrimExpr<u32>>, Y: Into<PrimExpr<u32>>>(x: X, y: Y) -> Expr<Uint2> {
    Expr::<Uint2>::new(x.into(), y.into())
}
#[inline]
pub fn make_uint3<X: Into<PrimExpr<u32>>, Y: Into<PrimExpr<u32>>, Z: Into<PrimExpr<u32>>>(
    x: X,
    y: Y,
    z: Z,
) -> Expr<Uint3> {
    Expr::<Uint3>::new(x.into(), y.into(), z.into())
}
#[inline]
pub fn make_uint4<
    X: Into<PrimExpr<u32>>,
    Y: Into<PrimExpr<u32>>,
    Z: Into<PrimExpr<u32>>,
    W: Into<PrimExpr<u32>>,
>(
    x: X,
    y: Y,
    z: Z,
    w: W,
) -> Expr<Uint4> {
    Expr::<Uint4>::new(x.into(), y.into(), z.into(), w.into())
}

#[cfg(test)]
mod test {
    #[test]
    fn test_size() {
        use crate::prelude::*;
        use crate::*;
        macro_rules! assert_size {
            ($ty:ty) => {
                {assert_eq!(std::mem::size_of::<$ty>(), <$ty as TypeOf>::type_().size());}
            };
            ($ty:ty, $($rest:ty),*) => {
                assert_size!($ty);
                assert_size!($($rest),*);
            };
        }
        assert_size!(f32, f64, bool, u16, u32, u64, i16, i32, i64);
        assert_size!(Float2, Float3, Float4, Int2, Int3, Int4, Uint2, Uint3, Uint4);
        assert_size!(Short2, Short3, Short4, Ushort2, Ushort3, Ushort4);
        assert_size!(Long2, Long3, Long4, Ulong2, Ulong3, Ulong4);
        assert_size!(Mat2, Mat3, Mat4);
        assert_size!(PackedFloat2, PackedFloat3, PackedFloat4);
        assert_eq!(std::mem::size_of::<PackedFloat3>(), 12);
    }
}
