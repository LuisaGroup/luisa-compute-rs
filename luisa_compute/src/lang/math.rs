use luisa_compute_ir::{
    context::register_type,
    ir::{MatrixType, NodeRef, Primitive, Type, VectorElementType, VectorType},
    TypeOf,
};

pub use super::math_impl::*;
use super::{Aggregate, Expr, Proxy, Value, __extract};
macro_rules! impl_proxy_fields {
    ($proxy:ident, $scalar:ty, x) => {
        impl $proxy {
            #[inline]
            pub fn x(&self) -> Expr<$scalar> {
                Expr::from_proxy(<$scalar as Value>::Proxy::from_node(__extract::<$scalar>(
                    self.node, 1,
                )))
            }
        }
    };
    ($proxy:ident, $scalar:ty, y) => {
        impl $proxy {
            #[inline]
            pub fn y(&self) -> Expr<$scalar> {
                Expr::from_proxy(<$scalar as Value>::Proxy::from_node(__extract::<$scalar>(
                    self.node, 2,
                )))
            }
        }
    };
    ($proxy:ident, $scalar:ty, z) => {
        impl $proxy {
            #[inline]
            pub fn z(&self) -> Expr<$scalar> {
                Expr::from_proxy(<$scalar as Value>::Proxy::from_node(__extract::<$scalar>(
                    self.node, 3,
                )))
            }
        }
    };
    ($proxy:ident, $scalar:ty, w) => {
        impl $proxy {
            #[inline]
            pub fn w(&self) -> Expr<$scalar> {
                Expr::from_proxy(<$scalar as Value>::Proxy::from_node(__extract::<$scalar>(
                    self.node, 4,
                )))
            }
        }
    };
}
macro_rules! impl_vec_proxy {
    ($vec:ident, $proxy:ident, $scalar:ty, $scalar_ty:ident, $length:literal, $($comp:ident), *) => {
        #[derive(Clone, Copy)]
        pub struct $proxy {
            node: NodeRef,
        }
        impl Value for $vec {
            type Proxy = $proxy;
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
        impl Aggregate for $proxy {
            fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
                nodes.push(self.node);
            }
            fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
                Self {
                    node: iter.next().unwrap(),
                }
            }
        }
        impl Proxy<$vec> for $proxy {
            fn from_node(node: NodeRef) -> Self {
                Self { node }
            }
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        $(impl_proxy_fields!($proxy, $scalar, $comp);)*
    };
}

macro_rules! impl_mat_proxy {
    ($mat:ident, $proxy:ident, $vec:ty, $scalar_ty:ident, $length:literal) => {
        #[derive(Clone, Copy)]
        pub struct $proxy {
            node: NodeRef,
        }
        impl Value for $mat {
            type Proxy = $proxy;
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
        impl Aggregate for $proxy {
            fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
                nodes.push(self.node);
            }
            fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
                Self {
                    node: iter.next().unwrap(),
                }
            }
        }
        impl Proxy<$mat> for $proxy {
            fn from_node(node: NodeRef) -> Self {
                Self { node }
            }
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        impl $proxy {
            pub fn col(&self, index: usize) -> Expr<$vec> {
                Expr::from_proxy(<$vec as Value>::Proxy::from_node(__extract::<$vec>(
                    self.node, index,
                )))
            }
        }
    };
}

impl_vec_proxy!(Vec2, Vec2Proxy, f32, Float32, 2, x, y);
impl_vec_proxy!(Vec3, Vec3Proxy, f32, Float32, 3, x, y, z);
impl_vec_proxy!(Vec4, Vec4Proxy, f32, Float32, 4, x, y, z, w);

impl_vec_proxy!(UVec2, UVec2Proxy, u32, Uint32, 2, x, y);
impl_vec_proxy!(UVec3, UVec3Proxy, u32, Uint32, 3, x, y, z);
impl_vec_proxy!(UVec4, UVec4Proxy, u32, Uint32, 4, x, y, z, w);

impl_vec_proxy!(IVec2, IVec2Proxy, i32, Int32, 2, x, y);
impl_vec_proxy!(IVec3, IVec3Proxy, i32, Int32, 3, x, y, z);
impl_vec_proxy!(IVec4, IVec4Proxy, i32, Int32, 4, x, y, z, w);

impl_mat_proxy!(Mat2, Mat2Proxy, Vec2, Float32, 2);
impl_mat_proxy!(Mat3, Mat3Proxy, Vec3, Float32, 3);
impl_mat_proxy!(Mat4, Mat4Proxy, Vec4, Float32, 4);
