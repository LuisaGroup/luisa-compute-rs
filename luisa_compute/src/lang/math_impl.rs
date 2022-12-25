use std::ops::*;

trait ArithCmp {
    type Output;
    fn cmplt(self, rhs: Self) -> Self::Output;
    fn cmple(self, rhs: Self) -> Self::Output;
    fn cmpgt(self, rhs: Self) -> Self::Output;
    fn cmpge(self, rhs: Self) -> Self::Output;
    fn cmpne(self, rhs: Self) -> Self::Output;
    fn cmpeq(self, rhs: Self) -> Self::Output;
}
macro_rules! impl_cmp_for_primitive {
    ($t:ty) => {
        impl ArithCmp for $t {
            type Output = bool;
            #[inline]
            fn cmplt(self, rhs: Self) -> Self::Output {
                self < rhs
            }
            #[inline]
            fn cmple(self, rhs: Self) -> Self::Output {
                self <= rhs
            }
            #[inline]
            fn cmpgt(self, rhs: Self) -> Self::Output {
                self > rhs
            }
            #[inline]
            fn cmpge(self, rhs: Self) -> Self::Output {
                self >= rhs
            }
            #[inline]
            fn cmpne(self, rhs: Self) -> Self::Output {
                self != rhs
            }
            #[inline]
            fn cmpeq(self, rhs: Self) -> Self::Output {
                self == rhs
            }
        }
    };
}
impl_cmp_for_primitive!(f32);
impl_cmp_for_primitive!(f64);
impl_cmp_for_primitive!(i32);
impl_cmp_for_primitive!(i64);
impl_cmp_for_primitive!(u32);
impl_cmp_for_primitive!(u64);
macro_rules! def_vec {
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
def_vec!(Vec2, f32, 8, x, y);
def_vec!(Vec3, f32, 16, x, y, z);
def_vec!(Vec4, f32, 16, x, y, z, w);

def_vec!(UVec2, u32, 8, x, y);
def_vec!(UVec3, u32, 16, x, y, z);
def_vec!(UVec4, u32, 16, x, y, z, w);

def_vec!(IVec2, i32, 8, x, y);
def_vec!(IVec3, i32, 16, x, y, z);
def_vec!(IVec4, i32, 16, x, y, z, w);

def_vec!(DVec2, f64, 16, x, y);
def_vec!(DVec3, f64, 32, x, y, z);
def_vec!(DVec4, f64, 32, x, y, z, w);

def_vec!(BVec2, bool, 2, x, y);
def_vec!(BVec3, bool, 4, x, y, z);
def_vec!(BVec4, bool, 4, x, y, z, w);

macro_rules! impl_ops{
    ($vec:ident, $trait_:ident, $method:ident, $op_assign_trait:ident, $assign_method:ident, $scalar:ty, $($comp:ident), *)=>{
        impl $trait_ for $vec {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: Self) -> Self {
                Self {
                    $($comp: self.$comp.$method(rhs.$comp)), *
                }
            }
        }
        impl $trait_<$scalar> for $vec {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: $scalar) -> Self {
                Self {
                    $($comp: self.$comp.$method(rhs)), *
                }
            }
        }
        impl $trait_<$vec> for $scalar {
            type Output = $vec;
            #[inline]
            fn $method(self, rhs: $vec) -> $vec {
                $vec {
                    $($comp: self.$method(rhs.$comp)), *
                }
            }
        }
        impl $op_assign_trait for $vec {
            #[inline]
            fn $assign_method(&mut self, rhs: Self) {
                $(self.$comp.$assign_method(rhs.$comp)); *
            }
        }
        impl $op_assign_trait<$scalar> for $vec {
            #[inline]
            fn $assign_method(&mut self, rhs: $scalar) {
                $(self.$comp.$assign_method(rhs)); *
            }
        }
    }
}
macro_rules! common_binop {
    ($vec:ident,$scalar:ty, $($comp:ident), *) => {
        impl_ops!(
            $vec, Add, add, AddAssign, add_assign, $scalar, $($comp), *
        );
        impl_ops!(
            $vec, Sub, sub, SubAssign, sub_assign, $scalar, $($comp), *
        );
        impl_ops!(
            $vec, Mul, mul, MulAssign, mul_assign, $scalar, $($comp), *
        );
        impl_ops!(
            $vec, Div, div, DivAssign, div_assign, $scalar, $($comp), *
        );
        impl_ops!(
            $vec, Rem, rem, RemAssign, rem_assign, $scalar, $($comp), *
        );
        impl $vec {
            #[inline]
            pub fn dot(self, rhs: Self) -> $scalar {
                (self * rhs).reduce_sum()
            }
        }
    };
}
macro_rules! cmp_op {
    ($vec:ident,$bvec:ident, $($comp:ident), *) => {
        impl ArithCmp for $vec {
            type Output = $bvec;
            #[inline]
            fn cmplt(self, rhs: Self) -> $bvec {
                $bvec {
                    $($comp: self.$comp < rhs.$comp), *
                }
            }
            #[inline]
            fn cmple(self, rhs: Self) -> $bvec {
                $bvec {
                    $($comp: self.$comp <= rhs.$comp), *
                }
            }
            #[inline]
            fn cmpgt(self, rhs: Self) -> $bvec {
                $bvec {
                    $($comp: self.$comp > rhs.$comp), *
                }
            }
            #[inline]
            fn cmpge(self, rhs: Self) -> $bvec {
                $bvec {
                    $($comp: self.$comp >= rhs.$comp), *
                }
            }
            #[inline]
            fn cmpne(self, rhs: Self) -> $bvec {
                $bvec {
                    $($comp: self.$comp != rhs.$comp), *
                }
            }
            #[inline]
            fn cmpeq(self, rhs: Self) -> $bvec {
                $bvec {
                    $($comp: self.$comp == rhs.$comp), *
                }
            }
        }
        impl $vec {
            #[inline]
            pub fn max(self, rhs: Self) -> Self {
                Self {
                    $($comp: self.$comp.max(rhs.$comp)), *
                }
            }
            #[inline]
            pub fn min(self, rhs: Self) -> Self {
                Self {
                    $($comp: self.$comp.min(rhs.$comp)), *
                }
            }

        }
    };
}
macro_rules! math_func1 {
    ($vec:ident,$func:ident, $($comp:ident), *) => {
        impl $vec {
            #[inline]
            pub fn $func(self) -> Self {
                Self {
                    $($comp: self.$comp.$func()), *
                }
            }
        }
    };
}
macro_rules! math_funcs {
    ($vec:ident,$($comp:ident), *) => {
        math_func1!($vec, abs, $($comp), *);
        math_func1!($vec, acos, $($comp), *);
        math_func1!($vec, acosh, $($comp), *);
        math_func1!($vec, asin, $($comp), *);
        math_func1!($vec, asinh, $($comp), *);
        math_func1!($vec, atan, $($comp), *);
        math_func1!($vec, atanh, $($comp), *);
        math_func1!($vec, cbrt, $($comp), *);
        math_func1!($vec, ceil, $($comp), *);
        math_func1!($vec, cos, $($comp), *);
        math_func1!($vec, cosh, $($comp), *);
        math_func1!($vec, exp, $($comp), *);
        math_func1!($vec, exp2, $($comp), *);
        math_func1!($vec, exp_m1, $($comp), *);
        math_func1!($vec, floor, $($comp), *);
        math_func1!($vec, fract, $($comp), *);
        math_func1!($vec, ln, $($comp), *);
        math_func1!($vec, ln_1p, $($comp), *);
        math_func1!($vec, log10, $($comp), *);
        math_func1!($vec, log2, $($comp), *);
        math_func1!($vec, round, $($comp), *);
        math_func1!($vec, signum, $($comp), *);
        math_func1!($vec, sin, $($comp), *);
        math_func1!($vec, sinh, $($comp), *);
        math_func1!($vec, sqrt, $($comp), *);
        math_func1!($vec, tan, $($comp), *);
        math_func1!($vec, tanh, $($comp), *);
        math_func1!($vec, to_degrees, $($comp), *);
        math_func1!($vec, to_radians, $($comp), *);
        math_func1!($vec, trunc, $($comp), *);
    };
}
macro_rules! int_binop {
    ($vec:ident,$scalar:ty, $($comp:ident), *) => {
        impl_ops!($vec, Shl, shl, ShlAssign, shl_assign, $scalar, $($comp), *);
        impl_ops!($vec, Shr, shr, ShrAssign, shr_assign, $scalar, $($comp), *);
        impl_ops!($vec, BitAnd, bitand, BitAndAssign, bitand_assign, $scalar, $($comp), *);
        impl_ops!($vec, BitOr, bitor, BitOrAssign, bitor_assign, $scalar, $($comp), *);
        impl_ops!($vec, BitXor, bitxor, BitXorAssign, bitxor_assign, $scalar, $($comp), *);
    };
}
macro_rules! impl_reduce {
    ($vec:ident,$scalar:ty, $x:ident, $y:ident) => {
        impl $vec {
            #[inline]
            pub fn reduce_sum(self) -> $scalar {
                self.$x + self.$y
            }
            #[inline]
            pub fn reduce_prod(self) -> $scalar {
                self.$x * self.$y
            }
            #[inline]
            pub fn reduce_min(self) -> $scalar {
                self.$x.min(self.$y)
            }
            #[inline]
            pub fn reduce_max(self) -> $scalar {
                self.$x.max(self.$y)
            }
        }
    };
    ($vec:ident,$scalar:ty, $x:ident, $y:ident, $z:ident) => {
        impl $vec {
            #[inline]
            pub fn reduce_sum(self) -> $scalar {
                self.$x + self.$y + self.$z
            }
            #[inline]
            pub fn reduce_prod(self) -> $scalar {
                self.$x * self.$y * self.$z
            }
            #[inline]
            pub fn reduce_min(self) -> $scalar {
                self.$x.min(self.$y).min(self.$z)
            }
            #[inline]
            pub fn reduce_max(self) -> $scalar {
                self.$x.max(self.$y).max(self.$z)
            }
        }
    };
    ($vec:ident,$scalar:ty, $x:ident, $y:ident, $z:ident, $w:ident) => {
        impl $vec {
            #[inline]
            pub fn reduce_sum(self) -> $scalar {
                self.$x + self.$y + self.$z + self.$w
            }
            #[inline]
            pub fn reduce_prod(self) -> $scalar {
                self.$x * self.$y * self.$z * self.$w
            }
            #[inline]
            pub fn reduce_min(self) -> $scalar {
                self.$x.min(self.$y).min(self.$z).min(self.$w)
            }
            #[inline]
            pub fn reduce_max(self) -> $scalar {
                self.$x.max(self.$y).max(self.$z).max(self.$w)
            }
        }
    };
}
macro_rules! impl_index2 {
    ($vec:ident,$scalar:ty) => {
        impl Index<usize> for $vec {
            type Output = $scalar;
            #[inline]
            fn index(&self, i: usize) -> &$scalar {
                match i {
                    0 => &self.x,
                    1 => &self.y,
                    _ => panic!("index out of bounds"),
                }
            }
        }
        impl IndexMut<usize> for $vec {
            #[inline]
            fn index_mut(&mut self, i: usize) -> &mut $scalar {
                match i {
                    0 => &mut self.x,
                    1 => &mut self.y,
                    _ => panic!("index out of bounds"),
                }
            }
        }
    };
}
macro_rules! impl_index3 {
    ($vec:ident,$scalar:ty) => {
        impl Index<usize> for $vec {
            type Output = $scalar;
            #[inline]
            fn index(&self, i: usize) -> &$scalar {
                match i {
                    0 => &self.x,
                    1 => &self.y,
                    2 => &self.z,
                    _ => panic!("index out of bounds"),
                }
            }
        }
        impl IndexMut<usize> for $vec {
            #[inline]
            fn index_mut(&mut self, i: usize) -> &mut $scalar {
                match i {
                    0 => &mut self.x,
                    1 => &mut self.y,
                    2 => &mut self.z,
                    _ => panic!("index out of bounds"),
                }
            }
        }
    };
}
macro_rules! impl_index4 {
    ($vec:ident,$scalar:ty) => {
        impl Index<usize> for $vec {
            type Output = $scalar;
            #[inline]
            fn index(&self, i: usize) -> &$scalar {
                match i {
                    0 => &self.x,
                    1 => &self.y,
                    2 => &self.z,
                    3 => &self.w,
                    _ => panic!("index out of bounds"),
                }
            }
        }
        impl IndexMut<usize> for $vec {
            #[inline]
            fn index_mut(&mut self, i: usize) -> &mut $scalar {
                match i {
                    0 => &mut self.x,
                    1 => &mut self.y,
                    2 => &mut self.z,
                    3 => &mut self.w,
                    _ => panic!("index out of bounds"),
                }
            }
        }
    };
}
common_binop!(Vec2, f32, x, y);
common_binop!(Vec3, f32, x, y, z);
common_binop!(Vec4, f32, x, y, z, w);

common_binop!(UVec2, u32, x, y);
common_binop!(UVec3, u32, x, y, z);
common_binop!(UVec4, u32, x, y, z, w);

common_binop!(IVec2, i32, x, y);
common_binop!(IVec3, i32, x, y, z);
common_binop!(IVec4, i32, x, y, z, w);

common_binop!(DVec2, f64, x, y);
common_binop!(DVec3, f64, x, y, z);
common_binop!(DVec4, f64, x, y, z, w);

int_binop!(IVec2, i32, x, y);
int_binop!(IVec3, i32, x, y, z);
int_binop!(IVec4, i32, x, y, z, w);

int_binop!(UVec2, u32, x, y);
int_binop!(UVec3, u32, x, y, z);
int_binop!(UVec4, u32, x, y, z, w);

impl_index2!(Vec2, f32);
impl_index3!(Vec3, f32);
impl_index4!(Vec4, f32);

impl_index2!(UVec2, u32);
impl_index3!(UVec3, u32);
impl_index4!(UVec4, u32);

impl_index2!(IVec2, i32);
impl_index3!(IVec3, i32);
impl_index4!(IVec4, i32);

impl_ops!(
    BVec2,
    BitAnd,
    bitand,
    BitAndAssign,
    bitand_assign,
    bool,
    x,
    y
);
impl_ops!(
    BVec3,
    BitAnd,
    bitand,
    BitAndAssign,
    bitand_assign,
    bool,
    x,
    y,
    z
);
impl_ops!(
    BVec4,
    BitAnd,
    bitand,
    BitAndAssign,
    bitand_assign,
    bool,
    x,
    y,
    z,
    w
);

impl_ops!(BVec2, BitOr, bitor, BitOrAssign, bitor_assign, bool, x, y);
impl_ops!(
    BVec3,
    BitOr,
    bitor,
    BitOrAssign,
    bitor_assign,
    bool,
    x,
    y,
    z
);
impl_ops!(
    BVec4,
    BitOr,
    bitor,
    BitOrAssign,
    bitor_assign,
    bool,
    x,
    y,
    z,
    w
);

impl_ops!(
    BVec2,
    BitXor,
    bitxor,
    BitXorAssign,
    bitxor_assign,
    bool,
    x,
    y
);
impl_ops!(
    BVec3,
    BitXor,
    bitxor,
    BitXorAssign,
    bitxor_assign,
    bool,
    x,
    y,
    z
);
impl_ops!(
    BVec4,
    BitXor,
    bitxor,
    BitXorAssign,
    bitxor_assign,
    bool,
    x,
    y,
    z,
    w
);

cmp_op!(Vec2, BVec2, x, y);
cmp_op!(Vec3, BVec3, x, y, z);
cmp_op!(Vec4, BVec4, x, y, z, w);

cmp_op!(IVec2, BVec2, x, y);
cmp_op!(IVec3, BVec3, x, y, z);
cmp_op!(IVec4, BVec4, x, y, z, w);

cmp_op!(UVec2, BVec2, x, y);
cmp_op!(UVec3, BVec3, x, y, z);
cmp_op!(UVec4, BVec4, x, y, z, w);

math_funcs!(Vec2, x, y);
math_funcs!(Vec3, x, y, z);
math_funcs!(Vec4, x, y, z, w);

impl_reduce!(Vec2, f32, x, y);
impl_reduce!(Vec3, f32, x, y, z);
impl_reduce!(Vec4, f32, x, y, z, w);

impl_reduce!(DVec2, f64, x, y);
impl_reduce!(DVec3, f64, x, y, z);
impl_reduce!(DVec4, f64, x, y, z, w);

impl_reduce!(IVec2, i32, x, y);
impl_reduce!(IVec3, i32, x, y, z);
impl_reduce!(IVec4, i32, x, y, z, w);

impl_reduce!(UVec2, u32, x, y);
impl_reduce!(UVec3, u32, x, y, z);
impl_reduce!(UVec4, u32, x, y, z, w);

impl BVec2 {
    #[inline]
    pub fn any(self) -> bool {
        self.x || self.y
    }
    #[inline]
    pub fn all(self) -> bool {
        self.x && self.y
    }
}
impl BVec3 {
    #[inline]
    pub fn any(self) -> bool {
        self.x || self.y || self.z
    }
    #[inline]
    pub fn all(self) -> bool {
        self.x && self.y && self.z
    }
}
impl BVec4 {
    #[inline]
    pub fn any(self) -> bool {
        self.x || self.y || self.z || self.w
    }
    #[inline]
    pub fn all(self) -> bool {
        self.x && self.y && self.z && self.w
    }
}
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(8))]
pub struct Mat2 {
    pub cols: [Vec2; 2],
}
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
pub struct Mat3 {
    pub cols: [Vec3; 3],
}
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
pub struct Mat4 {
    pub cols: [Vec4; 4],
}

impl Mul<Vec2> for Mat2 {
    type Output = Vec2;
    #[inline]
    fn mul(self, rhs: Vec2) -> Vec2 {
        Vec2 {
            x: self.cols[0].dot(rhs),
            y: self.cols[1].dot(rhs),
        }
    }
}
impl Mul<Vec3> for Mat3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.cols[0].dot(rhs),
            y: self.cols[1].dot(rhs),
            z: self.cols[2].dot(rhs),
        }
    }
}
impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    #[inline]
    fn mul(self, rhs: Vec4) -> Vec4 {
        Vec4 {
            x: self.cols[0].dot(rhs),
            y: self.cols[1].dot(rhs),
            z: self.cols[2].dot(rhs),
            w: self.cols[3].dot(rhs),
        }
    }
}
impl Mul for Mat2 {
    type Output = Mat2;
    #[inline]
    fn mul(self, rhs: Mat2) -> Mat2 {
        Mat2 {
            cols: [self * rhs.cols[0], self * rhs.cols[1]],
        }
    }
}
impl Mul for Mat3 {
    type Output = Mat3;
    #[inline]
    fn mul(self, rhs: Mat3) -> Mat3 {
        Mat3 {
            cols: [self * rhs.cols[0], self * rhs.cols[1], self * rhs.cols[2]],
        }
    }
}
impl Mul for Mat4 {
    type Output = Mat4;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Mat4 {
            cols: [
                self * rhs.cols[0],
                self * rhs.cols[1],
                self * rhs.cols[2],
                self * rhs.cols[3],
            ],
        }
    }
}
impl Add for Mat2 {
    type Output = Mat2;
    #[inline]
    fn add(self, rhs: Mat2) -> Mat2 {
        Mat2 {
            cols: [self.cols[0] + rhs.cols[0], self.cols[1] + rhs.cols[1]],
        }
    }
}
impl Add for Mat3 {
    type Output = Mat3;
    #[inline]
    fn add(self, rhs: Mat3) -> Mat3 {
        Mat3 {
            cols: [
                self.cols[0] + rhs.cols[0],
                self.cols[1] + rhs.cols[1],
                self.cols[2] + rhs.cols[2],
            ],
        }
    }
}
impl Add for Mat4 {
    type Output = Mat4;
    #[inline]
    fn add(self, rhs: Mat4) -> Mat4 {
        Mat4 {
            cols: [
                self.cols[0] + rhs.cols[0],
                self.cols[1] + rhs.cols[1],
                self.cols[2] + rhs.cols[2],
                self.cols[3] + rhs.cols[3],
            ],
        }
    }
}
impl Sub for Mat2 {
    type Output = Mat2;
    #[inline]
    fn sub(self, rhs: Mat2) -> Mat2 {
        Mat2 {
            cols: [self.cols[0] - rhs.cols[0], self.cols[1] - rhs.cols[1]],
        }
    }
}
impl Sub for Mat3 {
    type Output = Mat3;
    #[inline]
    fn sub(self, rhs: Mat3) -> Mat3 {
        Mat3 {
            cols: [
                self.cols[0] - rhs.cols[0],
                self.cols[1] - rhs.cols[1],
                self.cols[2] - rhs.cols[2],
            ],
        }
    }
}
impl Sub for Mat4 {
    type Output = Mat4;
    #[inline]
    fn sub(self, rhs: Mat4) -> Mat4 {
        Mat4 {
            cols: [
                self.cols[0] - rhs.cols[0],
                self.cols[1] - rhs.cols[1],
                self.cols[2] - rhs.cols[2],
                self.cols[3] - rhs.cols[3],
            ],
        }
    }
}

impl Mat2 {
    #[inline]
    pub fn row(&self, i: usize) -> Vec2 {
        Vec2 {
            x: self.cols[0][i],
            y: self.cols[1][i],
        }
    }
    #[inline]
    pub fn col(&self, i: usize) -> Vec2 {
        self.cols[i]
    }
    #[inline]
    pub fn transpose(&self) -> Mat2 {
        Mat2 {
            cols: [self.row(0), self.row(1)],
        }
    }
    #[inline]
    pub fn determinant(&self) -> f32 {
        self.cols[0].x * self.cols[1].y - self.cols[0].y * self.cols[1].x
    }
    #[inline]
    pub fn inverse(&self) -> Mat2 {
        let det = self.determinant();
        Mat2 {
            cols: [
                Vec2 {
                    x: self.cols[1].y / det,
                    y: -self.cols[0].y / det,
                },
                Vec2 {
                    x: -self.cols[1].x / det,
                    y: self.cols[0].x / det,
                },
            ],
        }
    }
}
impl Mat3 {
    #[inline]
    pub fn diag(v: Vec3) -> Mat3 {
        Mat3 {
            cols: [
                Vec3::new(v.x, 0.0, 0.0),
                Vec3::new(0.0, v.y, 0.0),
                Vec3::new(0.0, 0.0, v.z),
            ],
        }
    }
    #[inline]
    pub fn identity() -> Mat3 {
        Mat3::diag(Vec3::new(1.0, 1.0, 1.0))
    }
    #[inline]
    pub fn row(&self, i: usize) -> Vec3 {
        Vec3 {
            x: self.cols[0][i],
            y: self.cols[1][i],
            z: self.cols[2][i],
        }
    }
    #[inline]
    pub fn col(&self, i: usize) -> Vec3 {
        self.cols[i]
    }
    #[inline]
    pub fn transpose(&self) -> Mat3 {
        Mat3 {
            cols: [self.row(0), self.row(1), self.row(2)],
        }
    }
    #[inline]
    pub fn determinant(&self) -> f32 {
        self.cols[0][0] * (self.cols[1][1] * self.cols[2][2] - self.cols[2][1] * self.cols[1][2])
            - self.cols[0][1]
                * (self.cols[1][0] * self.cols[2][2] - self.cols[1][2] * self.cols[2][0])
            + self.cols[0][2]
                * (self.cols[1][0] * self.cols[2][1] - self.cols[1][1] * self.cols[2][0])
    }
    #[inline]
    pub fn inverse(&self) -> Mat3 {
        let det = self.determinant();
        let invdet = 1.0 / det;
        let mut inv = Mat3::identity();
        inv.cols[0][0] =
            (self.cols[1][1] * self.cols[2][2] - self.cols[2][1] * self.cols[1][2]) * invdet;
        inv.cols[0][1] =
            (self.cols[0][2] * self.cols[2][1] - self.cols[0][1] * self.cols[2][2]) * invdet;
        inv.cols[0][2] =
            (self.cols[0][1] * self.cols[1][2] - self.cols[0][2] * self.cols[1][1]) * invdet;
        inv.cols[1][0] =
            (self.cols[1][2] * self.cols[2][0] - self.cols[1][0] * self.cols[2][2]) * invdet;
        inv.cols[1][1] =
            (self.cols[0][0] * self.cols[2][2] - self.cols[0][2] * self.cols[2][0]) * invdet;
        inv.cols[1][2] =
            (self.cols[1][0] * self.cols[0][2] - self.cols[0][0] * self.cols[1][2]) * invdet;
        inv.cols[2][0] =
            (self.cols[1][0] * self.cols[2][1] - self.cols[2][0] * self.cols[1][1]) * invdet;
        inv.cols[2][1] =
            (self.cols[2][0] * self.cols[0][1] - self.cols[0][0] * self.cols[2][1]) * invdet;
        inv.cols[2][2] =
            (self.cols[0][0] * self.cols[1][1] - self.cols[1][0] * self.cols[0][1]) * invdet;
        inv
    }
}
impl Mat4 {
    #[inline]
    pub fn diag(v: Vec4) -> Mat4 {
        Mat4 {
            cols: [
                Vec4::new(v.x, 0.0, 0.0, 0.0),
                Vec4::new(0.0, v.y, 0.0, 0.0),
                Vec4::new(0.0, 0.0, v.z, 0.0),
                Vec4::new(0.0, 0.0, 0.0, v.w),
            ],
        }
    }
    #[inline]
    pub fn identity() -> Mat4 {
        Mat4::diag(Vec4::new(1.0, 1.0, 1.0, 1.0))
    }
    #[inline]
    pub fn row(&self, i: usize) -> Vec4 {
        Vec4 {
            x: self.cols[0][i],
            y: self.cols[1][i],
            z: self.cols[2][i],
            w: self.cols[3][i],
        }
    }
    #[inline]
    pub fn col(&self, i: usize) -> Vec4 {
        self.cols[i]
    }
    #[inline]
    pub fn transpose(&self) -> Mat4 {
        Mat4 {
            cols: [self.row(0), self.row(1), self.row(2), self.row(3)],
        }
    }
    #[inline]
    pub fn determinant(&self) -> f32 {
        todo!()
    }
}
