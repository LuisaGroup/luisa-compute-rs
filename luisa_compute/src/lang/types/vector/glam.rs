use super::*;

macro_rules! impl_glam_conversions {
    ($($Vt:ty = $Gt:path),+ $(,)?) => {
        $(
            impl From<$Vt> for $Gt {
                fn from(value: $Vt) -> Self {
                    Self::from_array(value.elements)
                }
            }
            impl From<$Gt> for $Vt {
                fn from(value: $Gt) -> Self {
                    Self::from_elements(value.to_array())
                }
            }
        )+
    };
}
impl_glam_conversions!(
    Vec2<f32> = ::glam::Vec2,
    Vec3<f32> = ::glam::Vec3,
    Vec3<f32> = ::glam::Vec3A,
    Vec4<f32> = ::glam::Vec4,
    Vec2<f64> = ::glam::DVec2,
    Vec3<f64> = ::glam::DVec3,
    Vec4<f64> = ::glam::DVec4,
    Vec2<i32> = ::glam::IVec2,
    Vec3<i32> = ::glam::IVec3,
    Vec4<i32> = ::glam::IVec4,
    Vec2<u32> = ::glam::UVec2,
    Vec3<u32> = ::glam::UVec3,
    Vec4<u32> = ::glam::UVec4,
    Vec2<i64> = ::glam::I64Vec2,
    Vec3<i64> = ::glam::I64Vec3,
    Vec4<i64> = ::glam::I64Vec4,
    Vec2<u64> = ::glam::U64Vec2,
    Vec3<u64> = ::glam::U64Vec3,
    Vec4<u64> = ::glam::U64Vec4,
);

impl From<Vec2<bool>> for ::glam::BVec2 {
    fn from(value: Vec2<bool>) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}
impl From<::glam::BVec2> for Vec2<bool> {
    fn from(value: ::glam::BVec2) -> Self {
        Self::from_elements([value.x, value.y])
    }
}

impl From<Vec3<bool>> for ::glam::BVec3 {
    fn from(value: Vec3<bool>) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}
impl From<::glam::BVec3> for Vec3<bool> {
    fn from(value: ::glam::BVec3) -> Self {
        Self::from_elements([value.x, value.y, value.z])
    }
}

impl From<Vec4<bool>> for ::glam::BVec4 {
    fn from(value: Vec4<bool>) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
            w: value.w,
        }
    }
}
impl From<::glam::BVec4> for Vec4<bool> {
    fn from(value: ::glam::BVec4) -> Self {
        Self::from_elements([value.x, value.y, value.z, value.w])
    }
}

impl From<Mat2> for ::glam::Mat2 {
    fn from(value: Mat2) -> Self {
        ::glam::Mat2::from_cols_array_2d(&value.to_column_array())
    }
}
impl From<::glam::Mat2> for Mat2 {
    fn from(value: ::glam::Mat2) -> Self {
        Self::from_column_array(&value.to_cols_array_2d())
    }
}

impl From<Mat3> for ::glam::Mat3 {
    fn from(value: Mat3) -> Self {
        ::glam::Mat3::from_cols_array_2d(&value.to_column_array())
    }
}
impl From<::glam::Mat3> for Mat3 {
    fn from(value: ::glam::Mat3) -> Self {
        Self::from_column_array(&value.to_cols_array_2d())
    }
}

impl From<Mat4> for ::glam::Mat4 {
    fn from(value: Mat4) -> Self {
        ::glam::Mat4::from_cols_array_2d(&value.to_column_array())
    }
}
impl From<::glam::Mat4> for Mat4 {
    fn from(value: ::glam::Mat4) -> Self {
        Self::from_column_array(&value.to_cols_array_2d())
    }
}
