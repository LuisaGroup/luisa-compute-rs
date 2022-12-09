use std::ops::{Index, Mul, Add};

use crate::prelude::*;

impl Vec2Proxy {
    pub fn dot(&self, other: Expr<Vec2>) -> Expr<f32> {
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
        self.dot(Expr::from_proxy(*self)).sqrt()
    }
    pub fn length_squared(&self) -> Expr<f32> {
        self.dot(Expr::from_proxy(*self))
    }
    pub fn normalize(&self) -> Expr<Vec2> {
        *self / self.length()
    }
}
impl Vec3Proxy {
    pub fn dot(&self, other: Expr<Vec3>) -> Expr<f32> {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    pub fn cross(&self, other: Expr<Vec3>) -> Expr<Vec3> {
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
        self.dot(Expr::from_proxy(*self)).sqrt()
    }
    pub fn length_squared(&self) -> Expr<f32> {
        self.dot(Expr::from_proxy(*self))
    }
    pub fn normalize(&self) -> Expr<Vec3> {
        *self / self.length()
    }
}
impl Vec4Proxy {
    pub fn dot(&self, other: Expr<Vec4>) -> Expr<f32> {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }
    pub fn max_element(&self) -> Expr<f32> {
        self.x.max(self.y).max(self.z).max(self.w)
    }
    pub fn min_element(&self) -> Expr<f32> {
        self.x.min(self.y).min(self.z).min(self.w)
    }
    pub fn sum(&self) -> Expr<f32> {
        self.x + self.y + self.z + self.w
    }
    pub fn length(&self) -> Expr<f32> {
        self.dot(Expr::from_proxy(*self)).sqrt()
    }
    pub fn length_squared(&self) -> Expr<f32> {
        self.dot(Expr::from_proxy(*self))
    }
    pub fn normalize(&self) -> Expr<Vec4> {
        *self / self.length()
    }
}
impl IVec2Proxy {
    pub fn dot(&self, other: Expr<IVec2>) -> Expr<i32> {
        self.x * other.x + self.y * other.y
    }
    pub fn max_element(&self) -> Expr<i32> {
        self.x.max(self.y)
    }
    pub fn min_element(&self) -> Expr<i32> {
        self.x.min(self.y)
    }
    pub fn sum(&self) -> Expr<i32> {
        self.x + self.y
    }
}
impl IVec3Proxy {
    pub fn dot(&self, other: Expr<IVec3>) -> Expr<i32> {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    pub fn max_element(&self) -> Expr<i32> {
        self.x.max(self.y).max(self.z)
    }
    pub fn min_element(&self) -> Expr<i32> {
        self.x.min(self.y).min(self.z)
    }
    pub fn sum(&self) -> Expr<i32> {
        self.x + self.y + self.z
    }
}
impl IVec4Proxy {
    pub fn dot(&self, other: Expr<IVec4>) -> Expr<i32> {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }
    pub fn max_element(&self) -> Expr<i32> {
        self.x.max(self.y).max(self.z).max(self.w)
    }
    pub fn min_element(&self) -> Expr<i32> {
        self.x.min(self.y).min(self.z).min(self.w)
    }
    pub fn sum(&self) -> Expr<i32> {
        self.x + self.y + self.z + self.w
    }
}
impl BVec2Proxy {
    pub fn any(&self) -> Expr<bool> {
        self.x | self.y
    }
    pub fn all(&self) -> Expr<bool> {
        self.x & self.y
    }
}
impl BVec3Proxy {
    pub fn any(&self) -> Expr<bool> {
        self.x | self.y | self.z
    }
    pub fn all(&self) -> Expr<bool> {
        self.x & self.y & self.z
    }
}
impl BVec4Proxy {
    pub fn any(&self) -> Expr<bool> {
        self.x | self.y | self.z | self.w
    }
    pub fn all(&self) -> Expr<bool> {
        self.x & self.y & self.z & self.w
    }
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
        // impl std::ops::$trait_<Expr<$name>> for $scalar {
        //     type Output = Expr<$name>;
        //     fn $method(self, other: Expr<$name>) -> Self::Output {
        //         struct_!($name {
        //             $($comps: self $op other.$comps), *
        //         })
        //     }
        // }
    };
}

macro_rules! vec_binop_same_type {
    ($name:ident, $proxy:ident,  $scalar:ident, $($comps:ident), *) => {
        vec_binop!(Add, add, +, $name, $proxy, $scalar, $($comps), *);
        vec_binop!(Sub, sub, -, $name, $proxy, $scalar, $($comps), *);
        vec_binop!(Mul, mul, *, $name, $proxy, $scalar, $($comps), *);
        vec_binop!(Div, div, /, $name, $proxy, $scalar, $($comps), *);
        vec_binop!(Rem, rem, %, $name, $proxy, $scalar, $($comps), *);
    };
}
macro_rules! ivec_binop_same_type {
    ($name:ident, $proxy:ident, $scalar:ident, $($comps:ident), *) => {
        vec_binop!(BitAnd, bitand, &, $name, $proxy, $scalar, $($comps), *);
        vec_binop!(BitXor, bitxor, ^, $name, $proxy, $scalar, $($comps), *);
        vec_binop!(BitOr, bitor, |, $name, $proxy, $scalar, $($comps), *);
        vec_binop!(Shl, shl, <<, $name, $proxy, $scalar, $($comps), *);
        vec_binop!(Shr, shr, >>, $name, $proxy, $scalar, $($comps), *);
    };
}
macro_rules! bvec_binop_same_type {
    ($name:ident, $proxy:ident, $scalar:ident, $($comps:ident), *) => {
        vec_binop!(BitAnd, bitand, &, $name, $proxy, $scalar, $($comps), *);
        vec_binop!(BitXor, bitxor, ^, $name, $proxy, $scalar, $($comps), *);
        vec_binop!(BitOr, bitor, |, $name, $proxy, $scalar, $($comps), *);
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
impl Index<usize> for Vec2Proxy {
    type Output = Expr<f32>;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("index out of bounds"),
        }
    }
}
impl Index<usize> for Vec3Proxy {
    type Output = Expr<f32>;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("index out of bounds"),
        }
    }
}

impl Index<usize> for Vec4Proxy {
    type Output = Expr<f32>;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("index out of bounds"),
        }
    }
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

bvec_binop_same_type!(BVec2, BVec2Proxy, bool, x, y);
bvec_binop_same_type!(BVec3, BVec3Proxy, bool, x, y, z);
bvec_binop_same_type!(BVec4, BVec4Proxy, bool, x, y, z, w);

vec_binop_cmp!(Vec2, Vec2Proxy, BVec2, x, y);
vec_binop_cmp!(Vec3, Vec3Proxy, BVec3, x, y, z);
vec_binop_cmp!(Vec4, Vec4Proxy, BVec4, x, y, z, w);

vec_binop_cmp!(IVec2, IVec2Proxy, BVec2, x, y);
vec_binop_cmp!(IVec3, IVec3Proxy, BVec3, x, y, z);
vec_binop_cmp!(IVec4, IVec4Proxy, BVec4, x, y, z, w);

impl Mat3Proxy {
    pub fn col(&self, i: usize) -> Expr<Vec3> {
        match i {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => panic!("index out of bounds"),
        }
    }
    pub fn row(&self, i: usize) -> Expr<Vec3> {
        match i {
            0 => struct_!(Vec3 {
                x: self.x.x,
                y: self.y.x,
                z: self.z.x
            }),
            1 => struct_!(Vec3 {
                x: self.x.y,
                y: self.y.y,
                z: self.z.y
            }),
            2 => struct_!(Vec3 {
                x: self.x.z,
                y: self.y.z,
                z: self.z.z
            }),
            _ => panic!("index out of bounds"),
        }
    }
    pub fn transpose(&self) -> Expr<Mat3> {
        struct_!(Mat3 {
            x: self.row(0),
            y: self.row(1),
            z: self.row(2)
        })
    }
}
// impl Add for Mat3Proxy {
//     type Output = Expr<Mat3>;
//     fn add(self, rhs: Self) -> Self::Output {
//         struct_!(Mat3 {
//             x: self.x + rhs.x,
//             y: self.y + rhs.y,
//             z: self.z + rhs.z
//         })
//     }
// }
impl Mul<Expr<Vec3>> for Mat3Proxy {
    type Output = Expr<Vec3>;
    fn mul(self, rhs: Expr<Vec3>) -> Self::Output {
        struct_!(Vec3 {
            x: self.x.dot(rhs),
            y: self.y.dot(rhs),
            z: self.z.dot(rhs)
        })
    }
}
impl Mul for Mat3Proxy {
    type Output = Expr<Mat3>;
    fn mul(self, rhs: Self) -> Self::Output {
        struct_!(Mat3 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z
        })
    }
}