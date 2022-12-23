pub use super::math_impl::*;
use super::{Aggregate, Expr, ExprProxy, Value, VarProxy, __extract, traits::*};
use crate::prelude::PrimProxy;
use luisa_compute_ir::{
    context::register_type,
    ir::{MatrixType, NodeRef, Primitive, Type, VectorElementType, VectorType},
    TypeOf,
};
macro_rules! impl_proxy_fields {
    ($proxy:ident, $scalar:ty, x) => {
        impl $proxy {
            #[inline]
            pub fn x(&self) -> Expr<$scalar> {
                <PrimProxy<$scalar> as VarTrait>::from_node(__extract::<$scalar>(self.node, 0))
            }
        }
    };
    ($proxy:ident, $scalar:ty, y) => {
        impl $proxy {
            #[inline]
            pub fn y(&self) -> Expr<$scalar> {
                <PrimProxy<$scalar> as VarTrait>::from_node(__extract::<$scalar>(self.node, 1))
            }
        }
    };
    ($proxy:ident, $scalar:ty, z) => {
        impl $proxy {
            #[inline]
            pub fn z(&self) -> Expr<$scalar> {
                <PrimProxy<$scalar> as VarTrait>::from_node(__extract::<$scalar>(self.node, 2))
            }
        }
    };
    ($proxy:ident, $scalar:ty, w) => {
        impl $proxy {
            #[inline]
            pub fn w(&self) -> Expr<$scalar> {
                <PrimProxy<$scalar> as VarTrait>::from_node(__extract::<$scalar>(self.node, 3))
            }
        }
    };
}
macro_rules! impl_var_proxy_fields {
    ($proxy:ident, $scalar:ty, x) => {
        impl $proxy {
            #[inline]
            pub fn x(&self) -> Var<$scalar> {
                <PrimProxy<$scalar> as VarTrait>::from_node(__extract::<$scalar>(self.node, 0))
            }
        }
    };
    ($proxy:ident, $scalar:ty, y) => {
        impl $proxy {
            #[inline]
            pub fn y(&self) -> Var<$scalar> {
                <PrimProxy<$scalar> as VarTrait>::from_node(__extract::<$scalar>(self.node, 1))
            }
        }
    };
    ($proxy:ident, $scalar:ty, z) => {
        impl $proxy {
            #[inline]
            pub fn z(&self) -> Var<$scalar> {
                <PrimProxy<$scalar> as VarTrait>::from_node(__extract::<$scalar>(self.node, 2))
            }
        }
    };
    ($proxy:ident, $scalar:ty, w) => {
        impl $proxy {
            #[inline]
            pub fn w(&self) -> Var<$scalar> {
                <PrimProxy<$scalar> as VarTrait>::from_node(__extract::<$scalar>(self.node, 3))
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
        }
        impl TypeOf for $vec {
            fn type_() -> luisa_compute_ir::Gc<luisa_compute_ir::ir::Type> {
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
        impl ExprProxy<$vec> for $expr_proxy {
            fn from_node(node: NodeRef) -> Self {
                Self { node }
            }
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        impl VarProxy<$vec> for $var_proxy {
            fn from_node(node: NodeRef) -> Self {
                Self { node }
            }
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        $(impl_proxy_fields!($expr_proxy, $scalar, $comp);)*
    };
}

macro_rules! impl_mat_proxy {
    ($mat:ident, $expr_proxy:ident, $var_proxy:ident, $vec:ty, $scalar_ty:ident, $length:literal) => {
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
        }
        impl TypeOf for $mat {
            fn type_() -> luisa_compute_ir::Gc<luisa_compute_ir::ir::Type> {
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
        impl ExprProxy<$mat> for $expr_proxy {
            fn from_node(node: NodeRef) -> Self {
                Self { node }
            }
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        impl VarProxy<$mat> for $var_proxy {
            fn from_node(node: NodeRef) -> Self {
                Self { node }
            }
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        impl $expr_proxy {
            pub fn col(&self, index: usize) -> Expr<$vec> {
                Expr::<$vec>::from_node(__extract::<$vec>(self.node, index))
            }
        }
    };
}

impl_vec_proxy!(Vec2, Vec2Expr, Vec2Var, f32, Float32, 2, x, y);
impl_vec_proxy!(Vec3, Vec3Expr, Vec3Var, f32, Float32, 3, x, y, z);
impl_vec_proxy!(Vec4, Vec4Expr, Vec4Var, f32, Float32, 4, x, y, z, w);

impl_vec_proxy!(UVec2, UVec2Expr, UVec2Var, u32, Uint32, 2, x, y);
impl_vec_proxy!(UVec3, UVec3Expr, UVec3Var, u32, Uint32, 3, x, y, z);
impl_vec_proxy!(UVec4, UVec4Expr, UVec4Var, u32, Uint32, 4, x, y, z, w);

impl_vec_proxy!(IVec2, IVec2Expr, IVec2Var, i32, Int32, 2, x, y);
impl_vec_proxy!(IVec3, IVec3Expr, IVec3Var, i32, Int32, 3, x, y, z);
impl_vec_proxy!(IVec4, IVec4Expr, IVec4Var, i32, Int32, 4, x, y, z, w);

impl_mat_proxy!(Mat2, Mat2Expr, Mat2Var, Vec2, Float32, 2);
impl_mat_proxy!(Mat3, Mat3Expr, Mat3Var, Vec3, Float32, 3);
impl_mat_proxy!(Mat4, Mat4Expr, Mat4Var, Vec4, Float32, 4);
