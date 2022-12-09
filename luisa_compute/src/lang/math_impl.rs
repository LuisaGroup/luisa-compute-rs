use crate::prelude::*;

impl Vec2Proxy {
    pub fn dot(&self, other: Vec2Proxy) -> Expr<f32> {
        self.x * other.x + self.y * other.y
    }
    pub fn max_element(&self) -> Expr<f32> {
        self.x.max(self.y)
    }
    pub fn min_element(&self) -> Expr<f32> {
        self.x.min(self.y)
    }
    pub fn sum(&self) -> Expr<f32> {
        self.x + self.y
    }
    pub fn length(&self) -> Expr<f32> {
        self.dot(*self).sqrt()
    }
    pub fn length_squared(&self) -> Expr<f32> {
        self.dot(*self)
    }
}
impl Vec3Proxy {
    pub fn dot(&self, other: Vec3Proxy) -> Expr<f32> {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    pub fn cross(&self, other: Vec3Proxy) -> Expr<Vec3> {
        make_float3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
    pub fn max_element(&self) -> Expr<f32> {
        self.x.max(self.y).max(self.z)
    }
    pub fn min_element(&self) -> Expr<f32> {
        self.x.min(self.y).min(self.z)
    }
    pub fn sum(&self) -> Expr<f32> {
        self.x + self.y + self.z
    }
    pub fn length(&self) -> Expr<f32> {
        self.dot(*self).sqrt()
    }
    pub fn length_squared(&self) -> Expr<f32> {
        self.dot(*self)
    }
    // pub fn normalize(&self) -> Expr<Vec3> {
    //     self / self.length()
    // }
}

macro_rules! vec_binop {
    ($trait_:ident, $method:ident, $op:tt, $name:ident, $proxy:ident, $scalar:ident, $($comps:ident), *) => {
        impl std::ops::$trait_ for $proxy {
            type Output = Expr<$name>;
            fn $method(self, other: Self) -> Self::Output {
                struct_!($name {
                    $($comps: self.$comps $op other.$comps), *
                })
            }
        }
        impl std::ops::$trait_<$scalar> for $proxy {
            type Output = Expr<$name>;
            fn $method(self, other: $scalar) -> Self::Output {
                struct_!($name {
                    $($comps: self.$comps $op other), *
                })
            }
        }
        impl std::ops::$trait_<Expr<$scalar>> for $proxy {
            type Output = Expr<$name>;
            fn $method(self, other: Expr<$scalar>) -> Self::Output {
                struct_!($name {
                    $($comps: self.$comps $op other), *
                })
            }
        }
        impl std::ops::$trait_<$proxy> for $scalar {
            type Output = Expr<$name>;
            fn $method(self, other: $proxy) -> Self::Output {
                struct_!($name {
                    $($comps: self $op other.$comps), *
                })
            }
        }
        impl std::ops::$trait_<Expr<$name>> for $scalar {
            type Output = Expr<$name>;
            fn $method(self, other: Expr<$name>) -> Self::Output {
                struct_!($name {
                    $($comps: self $op other.$comps), *
                })
            }
        }
    };
}

macro_rules! vec_binop_same_type {
    ($name:ident, $proxy:ident,  $scalar:ident, $($comps:ident), *) => {
        vec_binop!(Add, add, +, $name, $proxy, $scalar, $($comps), *);
        vec_binop!(Sub, sub, -, $name, $proxy, $scalar, $($comps), *);
        vec_binop!(Mul, mul, *, $name, $proxy, $scalar, $($comps), *);
        vec_binop!(Div, div, /, $name, $proxy, $scalar, $($comps), *);
        vec_binop!(Rem, rem, /, $name, $proxy, $scalar, $($comps), *);
    };
}
macro_rules! ivec_binop_same_type {
    ($name:ident, $proxy:ident, $scalar:ident, $($comps:ident), *) => {
        vec_binop!(BitAnd, bitand, +, $name, $proxy, $scalar, $($comps), *);
        vec_binop!(BitXor, bitxor, -, $name, $proxy, $scalar, $($comps), *);
        vec_binop!(BitOr, bitor, *, $name, $proxy, $scalar, $($comps), *);
        vec_binop!(Shl, shl, /, $name, $proxy, $scalar, $($comps), *);
        vec_binop!(Shr, shr, /, $name, $proxy, $scalar, $($comps), *);
    };
}
macro_rules! vec_binop_cmp {
    ($name:ident, $proxy:ident,$bool_v:ident, $($comps:ident), *) => {
        impl $proxy {
            pub fn cmplt(&self, other: Self) -> Expr<$bool_v> {
                struct_!($bool_v {
                    $($comps: self.$comps.cmplt(other.$comps)), *
                })
            }
            pub fn cmple(&self, other: Self) -> Expr<$bool_v> {
                struct_!($bool_v {
                    $($comps: self.$comps.cmple(other.$comps)), *
                })
            }
            pub fn cmpgt(&self, other: Self) -> Expr<$bool_v> {
                struct_!($bool_v {
                    $($comps: self.$comps.cmpgt(other.$comps)), *
                })
            }
            pub fn cmpge(&self, other: Self) -> Expr<$bool_v> {
                struct_!($bool_v {
                    $($comps: self.$comps.cmpge(other.$comps)), *
                })
            }
            pub fn cmpeq(&self, other: Self) -> Expr<$bool_v> {
                struct_!($bool_v {
                    $($comps: self.$comps.cmpeq(other.$comps)), *
                })
            }
            pub fn cmpne(&self, other: Self) -> Expr<$bool_v> {
                struct_!($bool_v {
                    $($comps: self.$comps.cmpne(other.$comps)), *
                })
            }
        }
    };
}

vec_binop_same_type!(Vec2, Vec2Proxy, f32, x, y);
vec_binop_same_type!(Vec3, Vec3Proxy, f32, x, y, z);
vec_binop_same_type!(Vec4, Vec4Proxy, f32, x, y, z, w);

vec_binop_same_type!(IVec2, IVec2Proxy, i32, x, y);
vec_binop_same_type!(IVec3, IVec3Proxy, i32, x, y, z);
vec_binop_same_type!(IVec4, IVec4Proxy, i32, x, y, z, w);

ivec_binop_same_type!(IVec2, IVec2Proxy, i32, x, y);
ivec_binop_same_type!(IVec3, IVec3Proxy, i32, x, y, z);
ivec_binop_same_type!(IVec4, IVec4Proxy, i32, x, y, z, w);

vec_binop_cmp!(Vec2, Vec2Proxy, BVec2, x, y);
vec_binop_cmp!(Vec3, Vec3Proxy, BVec3, x, y, z);
vec_binop_cmp!(Vec4, Vec4Proxy, BVec4, x, y, z, w);

vec_binop_cmp!(IVec2, IVec2Proxy, BVec2, x, y);
vec_binop_cmp!(IVec3, IVec3Proxy, BVec3, x, y, z);
vec_binop_cmp!(IVec4, IVec4Proxy, BVec4, x, y, z, w);
