use super::alignment::*;
use super::core::*;
use super::*;
use ir::{MatrixType, VectorElementType, VectorType};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::ops::Mul;

#[cfg(feature = "glam")]
mod glam;
#[cfg(feature = "nalgebra")]
mod nalgebra;

pub mod coords;

trait VectorElement<const N: usize, const PACKED: bool>: Primitive {
    type A: Alignment;
}

#[repr(C)]
#[derive(Copy, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Vector<const N: usize, T: VectorElement<N, PACKED>, const PACKED: bool = false> {
    _align: T::A,
    elements: [T; N],
}
impl<const N: usize, T: Debug + VectorElement<N, P>, const P: bool> Debug for Vector<N, T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.elements.fmt(f)
    }
}

impl<const N: usize, T: VectorElement<N, PACKED>, const PACKED: bool = false> 

macro_rules! element {
    ($t:ty [ $l:literal ]: $a: ident, $p: ident) => {
        impl VectorElement<$l, false> for $t {
            type A = $a;
        }
        impl VectorElement<$l, true> for $t {
            type A = $p;
        }
    };
    ($t:ty [ $l:literal ]: $a: ident) => {
        element!($t [ $l ] : $a, Align1);
    }
}

element!(bool[2]: Align2);
element!(bool[3]: Align4);
element!(bool[4]: Align4);
// TODO: Make u8 support ir::TypeOf.
// element!(u8[2]: Align2);
// element!(u8[3]: Align4);
// element!(u8[4]: Align4);
// element!(i8[2]: Align2);
// element!(i8[3]: Align4);
// element!(i8[4]: Align4);

element!(f16[2]: Align4);
element!(f16[3]: Align8);
element!(f16[4]: Align8);
element!(u16[2]: Align4);
element!(u16[3]: Align8);
element!(u16[4]: Align8);
element!(i16[2]: Align4);
element!(i16[3]: Align8);
element!(i16[4]: Align8);

element!(f32[2]: Align8);
element!(f32[3]: Align16);
element!(f32[4]: Align16);
element!(u32[2]: Align8);
element!(u32[3]: Align16);
element!(u32[4]: Align16);
element!(i32[2]: Align8);
element!(i32[3]: Align16);
element!(i32[4]: Align16);

// TODO: Check whether size 8 alignment on packed f32 is necessary.
// This is an x86 feature though.
element!(f64[2]: Align16, Align8);
element!(f64[3]: Align32, Align8);
element!(f64[4]: Align32, Align8);
element!(u64[2]: Align16, Align8);
element!(u64[3]: Align32, Align8);
element!(u64[4]: Align32, Align8);
element!(i64[2]: Align16, Align8);
element!(i64[3]: Align32, Align8);
element!(i64[4]: Align32, Align8);


macro_rules! impl_proxy_fields {
    ($vec:ident, $proxy:ident, $scalar:ty, x) => {
        impl $proxy {
            #[inline]
            pub fn x(&self) -> prim::Expr<$scalar> {
                FromNode::from_node(__extract::<$scalar>(self.node, 0))
            }
            #[inline]
            pub fn set_x(&self, value: prim::Expr<$scalar>) -> Self {
                Self::from_node(__insert::<$vec>(self.node, 0, ToNode::node(&value)))
            }
        }
    };
    ($vec:ident,$proxy:ident, $scalar:ty, y) => {
        impl $proxy {
            #[inline]
            pub fn y(&self) -> prim::Expr<$scalar> {
                FromNode::from_node(__extract::<$scalar>(self.node, 1))
            }
            #[inline]
            pub fn set_y(&self, value: prim::Expr<$scalar>) -> Self {
                Self::from_node(__insert::<$vec>(self.node, 1, ToNode::node(&value)))
            }
        }
    };
    ($vec:ident,$proxy:ident, $scalar:ty, z) => {
        impl $proxy {
            #[inline]
            pub fn z(&self) -> prim::Expr<$scalar> {
                FromNode::from_node(__extract::<$scalar>(self.node, 2))
            }
            #[inline]
            pub fn set_z(&self, value: prim::Expr<$scalar>) -> Self {
                Self::from_node(__insert::<$vec>(self.node, 2, ToNode::node(&value)))
            }
        }
    };
    ($vec:ident,$proxy:ident, $scalar:ty, w) => {
        impl $proxy {
            #[inline]
            pub fn w(&self) -> prim::Expr<$scalar> {
                FromNode::from_node(__extract::<$scalar>(self.node, 3))
            }
            #[inline]
            pub fn set_w(&self, value: prim::Expr<$scalar>) -> Self {
                Self::from_node(__insert::<$vec>(self.node, 3, ToNode::node(&value)))
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
        impl VectorVarTrait for $expr_proxy { }
        impl ScalarOrVector for $expr_proxy {
            type Element = prim::Expr<$scalar>;
            type ElementHost = $scalar;
        }
        impl BuiltinVarTrait for $expr_proxy { }
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
        }
        impl ToNode for $expr_proxy {
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        impl FromNode for $var_proxy {
            fn from_node(node: NodeRef) -> Self {
                Self { node }
            }
        }
        impl ToNode for $var_proxy {
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
        impl std::ops::Deref for $var_proxy {
            type Target = $expr_proxy;
            fn deref(&self) -> &Self::Target {
                self._deref()
            }
        }
        impl From<$var_proxy> for $expr_proxy {
            fn from(var: $var_proxy) -> Self {
                var.load()
            }
        }
        impl_callable_param!($vec, $expr_proxy, $var_proxy);
        $(impl_proxy_fields!($vec, $expr_proxy, $scalar, $comp);)*
        $(impl_var_proxy_fields!($var_proxy, $scalar, $comp);)*
        impl $expr_proxy {
            #[inline]
            pub fn new($($comp: prim::Expr<$scalar>), *) -> Self {
                Self {
                    node: __compose::<$vec>(&[$(ToNode::node(&$comp)), *]),
                }
            }
            pub fn at(&self, index: usize) -> prim::Expr<$scalar> {
                FromNode::from_node(__extract::<$scalar>(self.node, index))
            }
        }
        impl $vec {
            #[inline]
            pub fn expr($($comp: impl Into<prim::Expr<$scalar>>), *) -> $expr_proxy {
                $expr_proxy::new($($comp.into()), *)
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
        impl MatrixVarTrait for $expr_proxy { }
        impl BuiltinVarTrait for $expr_proxy { }
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
        }
        impl ToNode for $expr_proxy {
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
        }
        impl ToNode for $var_proxy {
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        impl VarProxy for $var_proxy {
            type Value = $mat;
        }
        impl std::ops::Deref for $var_proxy {
            type Target = $expr_proxy;
            fn deref(&self) -> &Self::Target {
                self._deref()
            }
        }
        impl From<$var_proxy> for $expr_proxy {
            fn from(var: $var_proxy) -> Self {
                var.load()
            }
        }
        impl_callable_param!($mat, $expr_proxy, $var_proxy);
        impl $expr_proxy {
            #[inline]
            pub fn new($($comp: Expr<$vec>), *) -> Self {
                Self {
                    node: __compose::<$mat>(&[$(ToNode::node(&$comp)), *]),
                }
            }
            pub fn col(&self, index: usize) -> Expr<$vec> {
                Expr::<$vec>::from_node(__extract::<$vec>(self.node, index))
            }
        }
        impl $mat {
            #[inline]
            pub fn expr($($comp: impl Into<Expr<$vec>>), *) -> $expr_proxy {
                $expr_proxy::new($($comp.into()), *)
            }
        }
    };
}

impl_vec_proxy!(Bool2, Bool2Expr, Bool2Var, bool, Bool, 2, x, y);
impl_vec_proxy!(Bool3, Bool3Expr, Bool3Var, bool, Bool, 3, x, y, z);
impl_vec_proxy!(Bool4, Bool4Expr, Bool4Var, bool, Bool, 4, x, y, z, w);

impl_vec_proxy!(Half2, Half2Expr, Half2Var, f16, Float16, 2, x, y);
impl_vec_proxy!(Half3, Half3Expr, Half3Var, f16, Float16, 3, x, y, z);
impl_vec_proxy!(Half4, Half4Expr, Half4Var, f16, Float16, 4, x, y, z, w);

impl_vec_proxy!(Float2, Float2Expr, Float2Var, f32, Float32, 2, x, y);
impl_vec_proxy!(Float3, Float3Expr, Float3Var, f32, Float32, 3, x, y, z);
impl_vec_proxy!(Float4, Float4Expr, Float4Var, f32, Float32, 4, x, y, z, w);

impl_vec_proxy!(Double2, Double2Expr, Double2Var, f64, Float64, 2, x, y);
impl_vec_proxy!(Double3, Double3Expr, Double3Var, f64, Float64, 3, x, y, z);
impl_vec_proxy!(
    Double4,
    Double4Expr,
    Double4Var,
    f64,
    Float64,
    4,
    x,
    y,
    z,
    w
);

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
        impl $packed {
            pub fn unpack(&self) -> $vec {
                (*self).into()
            }
        }
        impl From<$packed> for $vec {
            fn from(v: $packed) -> Self {
                Self::new($(v.$comp()), *)
            }
        }
        impl $vec {
            pub fn pack(&self) -> $packed {
                (*self).into()
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
        impl std::ops::$tr<prim::Expr<$scalar>> for $proxy {
            type Output = $proxy;
            fn $m(self, rhs: prim::Expr<$scalar>) -> Self::Output {
                let rhs = Self::splat(rhs);
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::$tr, &[self.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        impl std::ops::$tr<$proxy> for prim::Expr<$scalar> {
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
        impl std::ops::$tr<prim::Expr<$scalar>> for $proxy {
            type Output = $proxy;
            fn $m(self, rhs: prim::Expr<$scalar>) -> Self::Output {
                let rhs = Self::fill(rhs);
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::$tr, &[self.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        impl std::ops::$tr<$proxy> for prim::Expr<$scalar> {
            type Output = $proxy;
            fn $m(self, rhs: $proxy) -> Self::Output {
                let lhs = <$proxy>::fill(self);
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::$tr, &[lhs.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
    };
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
        // Mat * Mat
        impl std::ops::MulAssign<$proxy> for $proxy {
            fn mul_assign(&mut self, rhs: $proxy) {
                use std::ops::Mul;
                *self = (*self).mul(rhs);
            }
        }
        impl std::ops::Mul for $proxy {
            type Output = $proxy;
            fn mul(self, rhs: $proxy) -> Self::Output {
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::Mul, &[self.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        // Mat * Scalar
        impl std::ops::MulAssign<$scalar> for $proxy {
            fn mul_assign(&mut self, rhs: $scalar) {
                use std::ops::Mul;
                *self = (*self).mul(rhs);
            }
        }
        impl std::ops::Mul<$scalar> for $proxy {
            type Output = $proxy;
            fn mul(self, rhs: $scalar) -> Self::Output {
                let rhs = Self::fill(rhs);
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(
                        Func::MatCompMul,
                        &[self.node, rhs.node],
                        <$t as TypeOf>::type_(),
                    )
                }))
            }
        }
        impl std::ops::Mul<$proxy> for $scalar {
            type Output = $proxy;
            fn mul(self, rhs: $proxy) -> Self::Output {
                let lhs = <$proxy>::fill(self);
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(
                        Func::MatCompMul,
                        &[lhs.node, rhs.node],
                        <$t as TypeOf>::type_(),
                    )
                }))
            }
        }
        impl std::ops::Mul<prim::Expr<$scalar>> for $proxy {
            type Output = $proxy;
            fn mul(self, rhs: prim::Expr<$scalar>) -> Self::Output {
                let rhs = Self::fill(rhs);
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(
                        Func::MatCompMul,
                        &[self.node, rhs.node],
                        <$t as TypeOf>::type_(),
                    )
                }))
            }
        }
        impl std::ops::Mul<$proxy> for prim::Expr<$scalar> {
            type Output = $proxy;
            fn mul(self, rhs: $proxy) -> Self::Output {
                let lhs = <$proxy>::fill(self);
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(
                        Func::MatCompMul,
                        &[lhs.node, rhs.node],
                        <$t as TypeOf>::type_(),
                    )
                }))
            }
        }
        // Rem
        impl std::ops::RemAssign<$scalar> for $proxy {
            fn rem_assign(&mut self, rhs: $scalar) {
                use std::ops::Rem;
                *self = (*self).rem(rhs);
            }
        }
        impl std::ops::Rem<$scalar> for $proxy {
            type Output = $proxy;
            fn rem(self, rhs: $scalar) -> Self::Output {
                let rhs: prim::Expr<$scalar> = rhs.into();
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::Rem, &[self.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        impl std::ops::Rem<prim::Expr<$scalar>> for $proxy {
            type Output = $proxy;
            fn rem(self, rhs: prim::Expr<$scalar>) -> Self::Output {
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::Rem, &[self.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        // Div
        impl std::ops::DivAssign<$scalar> for $proxy {
            fn div_assign(&mut self, rhs: $scalar) {
                use std::ops::Div;
                *self = (*self).div(rhs);
            }
        }
        impl std::ops::Div<$scalar> for $proxy {
            type Output = $proxy;
            fn div(self, rhs: $scalar) -> Self::Output {
                let rhs: prim::Expr<$scalar> = rhs.into();
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::Div, &[self.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        impl std::ops::Div<prim::Expr<$scalar>> for $proxy {
            type Output = $proxy;
            fn div(self, rhs: prim::Expr<$scalar>) -> Self::Output {
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::Div, &[self.node, rhs.node], <$t as TypeOf>::type_())
                }))
            }
        }
        // Neg
        impl std::ops::Neg for $proxy {
            type Output = $proxy;
            fn neg(self) -> Self::Output {
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::Neg, &[self.node], <$t as TypeOf>::type_())
                }))
            }
        }
        impl $proxy {
            pub fn comp_mul(&self, other: Self) -> Self {
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(
                        Func::MatCompMul,
                        &[self.node, other.node],
                        <$t as TypeOf>::type_(),
                    )
                }))
            }
        }
    };
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
                    let ret = s.call(Func::BitNot, &[ToNode::node(&self)], Self::Output::type_());
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
            pub fn splat<V: Into<prim::Expr<bool>>>(value: V) -> Self {
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
            pub fn all(&self) -> prim::Expr<bool> {
                Expr::<bool>::from_node(__current_scope(|s| {
                    s.call(Func::All, &[self.node], <bool as TypeOf>::type_())
                }))
            }
            pub fn any(&self) -> prim::Expr<bool> {
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
            pub fn reduce_sum(&self) -> prim::Expr<$scalar> {
                FromNode::from_node(__current_scope(|s| {
                    s.call(Func::ReduceSum, &[self.node], <$scalar as TypeOf>::type_())
                }))
            }
            #[inline]
            pub fn reduce_prod(&self) -> prim::Expr<$scalar> {
                FromNode::from_node(__current_scope(|s| {
                    s.call(Func::ReduceProd, &[self.node], <$scalar as TypeOf>::type_())
                }))
            }
            #[inline]
            pub fn reduce_min(&self) -> prim::Expr<$scalar> {
                FromNode::from_node(__current_scope(|s| {
                    s.call(Func::ReduceMin, &[self.node], <$scalar as TypeOf>::type_())
                }))
            }
            #[inline]
            pub fn reduce_max(&self) -> prim::Expr<$scalar> {
                FromNode::from_node(__current_scope(|s| {
                    s.call(Func::ReduceMax, &[self.node], <$scalar as TypeOf>::type_())
                }))
            }
            #[inline]
            pub fn dot(&self, rhs: $proxy) -> prim::Expr<$scalar> {
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
            pub fn splat<V: Into<prim::Expr<$scalar>>>(value: V) -> Self {
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
            pub fn length(&self) -> prim::Expr<$scalar> {
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
            pub fn length_squared(&self) -> prim::Expr<$scalar> {
                FromNode::from_node(__current_scope(|s| {
                    s.call(
                        Func::LengthSquared,
                        &[self.node],
                        <$scalar as TypeOf>::type_(),
                    )
                }))
            }
            #[inline]
            pub fn distance(&self, rhs: $proxy) -> prim::Expr<$scalar> {
                (*self - rhs).length()
            }
            #[inline]
            pub fn distance_squared(&self, rhs: $proxy) -> prim::Expr<$scalar> {
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

// a little shit
macro_rules! impl_arith_binop_f16 {
    ($t:ty, $scalar:ty, $proxy:ty) => {
        impl_common_op_f16!($t, $scalar, $proxy);
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
macro_rules! impl_common_op_f16 {
    ($t:ty, $scalar:ty, $proxy:ty) => {
        impl $proxy {
            pub fn splat<V: Into<prim::Expr<$scalar>>>(value: V) -> Self {
                let value = value.into();
                <$proxy>::from_node(__current_scope(|s| {
                    s.call(Func::Vec, &[value.node], <$t as TypeOf>::type_())
                }))
            }
            pub fn zero() -> Self {
                Self::splat(f16::from_f32(0.0f32))
            }
            pub fn one() -> Self {
                Self::splat(f16::from_f32(1.0f32))
            }
        }
    };
}

impl_arith_binop_f16!(Half2, f16, Half2Expr);
impl_arith_binop_f16!(Half3, f16, Half3Expr);
impl_arith_binop_f16!(Half4, f16, Half4Expr);

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

impl_select!(Bool2, Half2, Half2Expr);
impl_select!(Bool3, Half3, Half3Expr);
impl_select!(Bool4, Half4, Half4Expr);

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
                        &[self.node, ToNode::node(&x), ToNode::node(&y)],
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
                            ToNode::node(&x),
                            ToNode::node(&y),
                            ToNode::node(&z),
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
                            ToNode::node(&x),
                            ToNode::node(&y),
                            ToNode::node(&z),
                            ToNode::node(&w),
                        ],
                        <$v4 as TypeOf>::type_(),
                    )
                }))
            }
        }
    };
}

impl_permute!(Vec2Swizzle, Half2Expr, 2, Half2, Half3, Half4);
impl_permute!(Vec3Swizzle, Half3Expr, 3, Half2, Half3, Half4);
impl_permute!(Vec4Swizzle, Half4Expr, 4, Half2, Half3, Half4);

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
            type Half = Half2Expr;
            type Bool = Bool2Expr;
            type Double = Double2Expr;
            type Long = Long2Expr;
            type Ulong = Ulong2Expr;
        }
        impl CommonVarOp for $t {}
        impl VarCmp for $t {}
        impl VarCmpEq for $t {}
        impl From<$v> for $t {
            fn from(v: $v) -> Self {
                Self::new((v.x).expr(), (v.y).expr())
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
            type Half = Half3Expr;
            type Bool = Bool3Expr;
            type Double = Double3Expr;
            type Long = Long3Expr;
            type Ulong = Ulong3Expr;
        }
        impl CommonVarOp for $t {}
        impl VarCmp for $t {}
        impl VarCmpEq for $t {}
        impl From<$v> for $t {
            fn from(v: $v) -> Self {
                Self::new(v.x.expr(), v.y.expr(), v.z.expr())
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
            type Int = Int4Expr;
            type Uint = Uint4Expr;
            type Float = Float4Expr;
            type Double = Double4Expr;
            type Half = Half4Expr;
            type Bool = Bool4Expr;
            type Long = Long4Expr;
            type Ulong = Ulong4Expr;
        }
        impl CommonVarOp for $t {}
        impl VarCmp for $t {}
        impl VarCmpEq for $t {}
        impl From<$v> for $t {
            fn from(v: $v) -> Self {
                Self::new(v.x.expr(), v.y.expr(), v.z.expr(), v.w.expr())
            }
        }
    };
}

impl_var_trait2!(Half2Expr, Half2);
impl_var_trait2!(Float2Expr, Float2);
impl_var_trait2!(Double2Expr, Double2);
impl_var_trait2!(Short2Expr, Short2);
impl_var_trait2!(Ushort2Expr, Ushort2);
impl_var_trait2!(Int2Expr, Int2);
impl_var_trait2!(Uint2Expr, Uint2);
impl_var_trait2!(Bool2Expr, Bool2);
impl_var_trait2!(Long2Expr, Long2);
impl_var_trait2!(Ulong2Expr, Ulong2);

impl_var_trait3!(Half3Expr, Half3);
impl_var_trait3!(Float3Expr, Float3);
impl_var_trait3!(Double3Expr, Double3);
impl_var_trait3!(Short3Expr, Short3);
impl_var_trait3!(Ushort3Expr, Ushort3);
impl_var_trait3!(Int3Expr, Int3);
impl_var_trait3!(Uint3Expr, Uint3);
impl_var_trait3!(Bool3Expr, Bool3);
impl_var_trait3!(Long3Expr, Long3);
impl_var_trait3!(Ulong3Expr, Ulong3);

impl_var_trait4!(Half4Expr, Half4);
impl_var_trait4!(Float4Expr, Float4);
impl_var_trait4!(Double4Expr, Double4);
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

impl_float_trait!(Half2Expr);
impl_float_trait!(Half3Expr);
impl_float_trait!(Half4Expr);
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

impl_int_trait!(Short2Expr);
impl_int_trait!(Short3Expr);
impl_int_trait!(Short4Expr);

impl_int_trait!(Ushort2Expr);
impl_int_trait!(Ushort3Expr);
impl_int_trait!(Ushort4Expr);

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
    pub fn fill(e: impl Into<prim::Expr<f32>> + Copy) -> Self {
        Self::new(Float2::expr(e, e), Float2::expr(e, e))
    }
    pub fn eye(e: Expr<Float2>) -> Self {
        Self::new(Float2::expr(e.x(), 0.0), Float2::expr(0.0, e.y()))
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
    pub fn determinant(&self) -> prim::Expr<f32> {
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
    pub fn fill(e: impl Into<prim::Expr<f32>> + Copy) -> Self {
        Self::new(
            Float3::expr(e, e, e),
            Float3::expr(e, e, e),
            Float3::expr(e, e, e),
        )
    }
    pub fn eye(e: Expr<Float3>) -> Self {
        Self::new(
            Float3::expr(e.x(), 0.0, 0.0),
            Float3::expr(0.0, e.y(), 0.0),
            Float3::expr(0.0, 0.0, e.z()),
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
    pub fn determinant(&self) -> prim::Expr<f32> {
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
    pub fn fill(e: impl Into<prim::Expr<f32>> + Copy) -> Self {
        Self::new(
            Float4::expr(e, e, e, e),
            Float4::expr(e, e, e, e),
            Float4::expr(e, e, e, e),
            Float4::expr(e, e, e, e),
        )
    }
    pub fn eye(e: Expr<Float4>) -> Self {
        Self::new(
            Float4::expr(e.x(), 0.0, 0.0, 0.0),
            Float4::expr(0.0, e.y(), 0.0, 0.0),
            Float4::expr(0.0, 0.0, e.z(), 0.0),
            Float4::expr(0.0, 0.0, 0.0, e.w()),
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
    pub fn determinant(&self) -> prim::Expr<f32> {
        FromNode::from_node(__current_scope(|s| {
            s.call(Func::Determinant, &[self.node], <f32 as TypeOf>::type_())
        }))
    }
}
impl_arith_binop_for_mat!(Mat4, f32, Mat4Expr);

#[cfg(test)]
mod test {
    #[test]
    fn test_size() {
        use crate::internal_prelude::*;
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
