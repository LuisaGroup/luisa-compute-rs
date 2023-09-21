pub trait Vec2Swizzle {
    type Vec2;
    type Vec3;
    type Vec4;
    fn permute2(&self, x: u32, y: u32) -> Self::Vec2;
    fn permute3(&self, x: u32, y: u32, z: u32) -> Self::Vec3;
    fn permute4(&self, x: u32, y: u32, z: u32, w: u32) -> Self::Vec4;
    fn xx(&self) -> Self::Vec2 {
        self.permute2(0, 0)
    }
    fn xy(&self) -> Self::Vec2 {
        self.permute2(0, 1)
    }
    fn yx(&self) -> Self::Vec2 {
        self.permute2(1, 0)
    }
    fn yy(&self) -> Self::Vec2 {
        self.permute2(1, 1)
    }
    fn xxx(&self) -> Self::Vec3 {
        self.permute3(0, 0, 0)
    }
    fn xxy(&self) -> Self::Vec3 {
        self.permute3(0, 0, 1)
    }
    fn xyx(&self) -> Self::Vec3 {
        self.permute3(0, 1, 0)
    }
    fn xyy(&self) -> Self::Vec3 {
        self.permute3(0, 1, 1)
    }
    fn yxx(&self) -> Self::Vec3 {
        self.permute3(1, 0, 0)
    }
    fn yxy(&self) -> Self::Vec3 {
        self.permute3(1, 0, 1)
    }
    fn yyx(&self) -> Self::Vec3 {
        self.permute3(1, 1, 0)
    }
    fn yyy(&self) -> Self::Vec3 {
        self.permute3(1, 1, 1)
    }
    fn xxxx(&self) -> Self::Vec4 {
        self.permute4(0, 0, 0, 0)
    }
    fn xxxy(&self) -> Self::Vec4 {
        self.permute4(0, 0, 0, 1)
    }
    fn xxyx(&self) -> Self::Vec4 {
        self.permute4(0, 0, 1, 0)
    }
    fn xxyy(&self) -> Self::Vec4 {
        self.permute4(0, 0, 1, 1)
    }
    fn xyxx(&self) -> Self::Vec4 {
        self.permute4(0, 1, 0, 0)
    }
    fn xyxy(&self) -> Self::Vec4 {
        self.permute4(0, 1, 0, 1)
    }
    fn xyyx(&self) -> Self::Vec4 {
        self.permute4(0, 1, 1, 0)
    }
    fn xyyy(&self) -> Self::Vec4 {
        self.permute4(0, 1, 1, 1)
    }
    fn yxxx(&self) -> Self::Vec4 {
        self.permute4(1, 0, 0, 0)
    }
    fn yxxy(&self) -> Self::Vec4 {
        self.permute4(1, 0, 0, 1)
    }
    fn yxyx(&self) -> Self::Vec4 {
        self.permute4(1, 0, 1, 0)
    }
    fn yxyy(&self) -> Self::Vec4 {
        self.permute4(1, 0, 1, 1)
    }
    fn yyxx(&self) -> Self::Vec4 {
        self.permute4(1, 1, 0, 0)
    }
    fn yyxy(&self) -> Self::Vec4 {
        self.permute4(1, 1, 0, 1)
    }
    fn yyyx(&self) -> Self::Vec4 {
        self.permute4(1, 1, 1, 0)
    }
    fn yyyy(&self) -> Self::Vec4 {
        self.permute4(1, 1, 1, 1)
    }
}
pub trait Vec3Swizzle {
    type Vec2;
    type Vec3;
    type Vec4;
    fn permute2(&self, x: u32, y: u32) -> Self::Vec2;
    fn permute3(&self, x: u32, y: u32, z: u32) -> Self::Vec3;
    fn permute4(&self, x: u32, y: u32, z: u32, w: u32) -> Self::Vec4;
    fn xx(&self) -> Self::Vec2 {
        self.permute2(0, 0)
    }
    fn xy(&self) -> Self::Vec2 {
        self.permute2(0, 1)
    }
    fn xz(&self) -> Self::Vec2 {
        self.permute2(0, 2)
    }
    fn yx(&self) -> Self::Vec2 {
        self.permute2(1, 0)
    }
    fn yy(&self) -> Self::Vec2 {
        self.permute2(1, 1)
    }
    fn yz(&self) -> Self::Vec2 {
        self.permute2(1, 2)
    }
    fn zx(&self) -> Self::Vec2 {
        self.permute2(2, 0)
    }
    fn zy(&self) -> Self::Vec2 {
        self.permute2(2, 1)
    }
    fn zz(&self) -> Self::Vec2 {
        self.permute2(2, 2)
    }
    fn xxx(&self) -> Self::Vec3 {
        self.permute3(0, 0, 0)
    }
    fn xxy(&self) -> Self::Vec3 {
        self.permute3(0, 0, 1)
    }
    fn xxz(&self) -> Self::Vec3 {
        self.permute3(0, 0, 2)
    }
    fn xyx(&self) -> Self::Vec3 {
        self.permute3(0, 1, 0)
    }
    fn xyy(&self) -> Self::Vec3 {
        self.permute3(0, 1, 1)
    }
    fn xyz(&self) -> Self::Vec3 {
        self.permute3(0, 1, 2)
    }
    fn xzx(&self) -> Self::Vec3 {
        self.permute3(0, 2, 0)
    }
    fn xzy(&self) -> Self::Vec3 {
        self.permute3(0, 2, 1)
    }
    fn xzz(&self) -> Self::Vec3 {
        self.permute3(0, 2, 2)
    }
    fn yxx(&self) -> Self::Vec3 {
        self.permute3(1, 0, 0)
    }
    fn yxy(&self) -> Self::Vec3 {
        self.permute3(1, 0, 1)
    }
    fn yxz(&self) -> Self::Vec3 {
        self.permute3(1, 0, 2)
    }
    fn yyx(&self) -> Self::Vec3 {
        self.permute3(1, 1, 0)
    }
    fn yyy(&self) -> Self::Vec3 {
        self.permute3(1, 1, 1)
    }
    fn yyz(&self) -> Self::Vec3 {
        self.permute3(1, 1, 2)
    }
    fn yzx(&self) -> Self::Vec3 {
        self.permute3(1, 2, 0)
    }
    fn yzy(&self) -> Self::Vec3 {
        self.permute3(1, 2, 1)
    }
    fn yzz(&self) -> Self::Vec3 {
        self.permute3(1, 2, 2)
    }
    fn zxx(&self) -> Self::Vec3 {
        self.permute3(2, 0, 0)
    }
    fn zxy(&self) -> Self::Vec3 {
        self.permute3(2, 0, 1)
    }
    fn zxz(&self) -> Self::Vec3 {
        self.permute3(2, 0, 2)
    }
    fn zyx(&self) -> Self::Vec3 {
        self.permute3(2, 1, 0)
    }
    fn zyy(&self) -> Self::Vec3 {
        self.permute3(2, 1, 1)
    }
    fn zyz(&self) -> Self::Vec3 {
        self.permute3(2, 1, 2)
    }
    fn zzx(&self) -> Self::Vec3 {
        self.permute3(2, 2, 0)
    }
    fn zzy(&self) -> Self::Vec3 {
        self.permute3(2, 2, 1)
    }
    fn zzz(&self) -> Self::Vec3 {
        self.permute3(2, 2, 2)
    }
    fn xxxx(&self) -> Self::Vec4 {
        self.permute4(0, 0, 0, 0)
    }
    fn xxxy(&self) -> Self::Vec4 {
        self.permute4(0, 0, 0, 1)
    }
    fn xxxz(&self) -> Self::Vec4 {
        self.permute4(0, 0, 0, 2)
    }
    fn xxyx(&self) -> Self::Vec4 {
        self.permute4(0, 0, 1, 0)
    }
    fn xxyy(&self) -> Self::Vec4 {
        self.permute4(0, 0, 1, 1)
    }
    fn xxyz(&self) -> Self::Vec4 {
        self.permute4(0, 0, 1, 2)
    }
    fn xxzx(&self) -> Self::Vec4 {
        self.permute4(0, 0, 2, 0)
    }
    fn xxzy(&self) -> Self::Vec4 {
        self.permute4(0, 0, 2, 1)
    }
    fn xxzz(&self) -> Self::Vec4 {
        self.permute4(0, 0, 2, 2)
    }
    fn xyxx(&self) -> Self::Vec4 {
        self.permute4(0, 1, 0, 0)
    }
    fn xyxy(&self) -> Self::Vec4 {
        self.permute4(0, 1, 0, 1)
    }
    fn xyxz(&self) -> Self::Vec4 {
        self.permute4(0, 1, 0, 2)
    }
    fn xyyx(&self) -> Self::Vec4 {
        self.permute4(0, 1, 1, 0)
    }
    fn xyyy(&self) -> Self::Vec4 {
        self.permute4(0, 1, 1, 1)
    }
    fn xyyz(&self) -> Self::Vec4 {
        self.permute4(0, 1, 1, 2)
    }
    fn xyzx(&self) -> Self::Vec4 {
        self.permute4(0, 1, 2, 0)
    }
    fn xyzy(&self) -> Self::Vec4 {
        self.permute4(0, 1, 2, 1)
    }
    fn xyzz(&self) -> Self::Vec4 {
        self.permute4(0, 1, 2, 2)
    }
    fn xzxx(&self) -> Self::Vec4 {
        self.permute4(0, 2, 0, 0)
    }
    fn xzxy(&self) -> Self::Vec4 {
        self.permute4(0, 2, 0, 1)
    }
    fn xzxz(&self) -> Self::Vec4 {
        self.permute4(0, 2, 0, 2)
    }
    fn xzyx(&self) -> Self::Vec4 {
        self.permute4(0, 2, 1, 0)
    }
    fn xzyy(&self) -> Self::Vec4 {
        self.permute4(0, 2, 1, 1)
    }
    fn xzyz(&self) -> Self::Vec4 {
        self.permute4(0, 2, 1, 2)
    }
    fn xzzx(&self) -> Self::Vec4 {
        self.permute4(0, 2, 2, 0)
    }
    fn xzzy(&self) -> Self::Vec4 {
        self.permute4(0, 2, 2, 1)
    }
    fn xzzz(&self) -> Self::Vec4 {
        self.permute4(0, 2, 2, 2)
    }
    fn yxxx(&self) -> Self::Vec4 {
        self.permute4(1, 0, 0, 0)
    }
    fn yxxy(&self) -> Self::Vec4 {
        self.permute4(1, 0, 0, 1)
    }
    fn yxxz(&self) -> Self::Vec4 {
        self.permute4(1, 0, 0, 2)
    }
    fn yxyx(&self) -> Self::Vec4 {
        self.permute4(1, 0, 1, 0)
    }
    fn yxyy(&self) -> Self::Vec4 {
        self.permute4(1, 0, 1, 1)
    }
    fn yxyz(&self) -> Self::Vec4 {
        self.permute4(1, 0, 1, 2)
    }
    fn yxzx(&self) -> Self::Vec4 {
        self.permute4(1, 0, 2, 0)
    }
    fn yxzy(&self) -> Self::Vec4 {
        self.permute4(1, 0, 2, 1)
    }
    fn yxzz(&self) -> Self::Vec4 {
        self.permute4(1, 0, 2, 2)
    }
    fn yyxx(&self) -> Self::Vec4 {
        self.permute4(1, 1, 0, 0)
    }
    fn yyxy(&self) -> Self::Vec4 {
        self.permute4(1, 1, 0, 1)
    }
    fn yyxz(&self) -> Self::Vec4 {
        self.permute4(1, 1, 0, 2)
    }
    fn yyyx(&self) -> Self::Vec4 {
        self.permute4(1, 1, 1, 0)
    }
    fn yyyy(&self) -> Self::Vec4 {
        self.permute4(1, 1, 1, 1)
    }
    fn yyyz(&self) -> Self::Vec4 {
        self.permute4(1, 1, 1, 2)
    }
    fn yyzx(&self) -> Self::Vec4 {
        self.permute4(1, 1, 2, 0)
    }
    fn yyzy(&self) -> Self::Vec4 {
        self.permute4(1, 1, 2, 1)
    }
    fn yyzz(&self) -> Self::Vec4 {
        self.permute4(1, 1, 2, 2)
    }
    fn yzxx(&self) -> Self::Vec4 {
        self.permute4(1, 2, 0, 0)
    }
    fn yzxy(&self) -> Self::Vec4 {
        self.permute4(1, 2, 0, 1)
    }
    fn yzxz(&self) -> Self::Vec4 {
        self.permute4(1, 2, 0, 2)
    }
    fn yzyx(&self) -> Self::Vec4 {
        self.permute4(1, 2, 1, 0)
    }
    fn yzyy(&self) -> Self::Vec4 {
        self.permute4(1, 2, 1, 1)
    }
    fn yzyz(&self) -> Self::Vec4 {
        self.permute4(1, 2, 1, 2)
    }
    fn yzzx(&self) -> Self::Vec4 {
        self.permute4(1, 2, 2, 0)
    }
    fn yzzy(&self) -> Self::Vec4 {
        self.permute4(1, 2, 2, 1)
    }
    fn yzzz(&self) -> Self::Vec4 {
        self.permute4(1, 2, 2, 2)
    }
    fn zxxx(&self) -> Self::Vec4 {
        self.permute4(2, 0, 0, 0)
    }
    fn zxxy(&self) -> Self::Vec4 {
        self.permute4(2, 0, 0, 1)
    }
    fn zxxz(&self) -> Self::Vec4 {
        self.permute4(2, 0, 0, 2)
    }
    fn zxyx(&self) -> Self::Vec4 {
        self.permute4(2, 0, 1, 0)
    }
    fn zxyy(&self) -> Self::Vec4 {
        self.permute4(2, 0, 1, 1)
    }
    fn zxyz(&self) -> Self::Vec4 {
        self.permute4(2, 0, 1, 2)
    }
    fn zxzx(&self) -> Self::Vec4 {
        self.permute4(2, 0, 2, 0)
    }
    fn zxzy(&self) -> Self::Vec4 {
        self.permute4(2, 0, 2, 1)
    }
    fn zxzz(&self) -> Self::Vec4 {
        self.permute4(2, 0, 2, 2)
    }
    fn zyxx(&self) -> Self::Vec4 {
        self.permute4(2, 1, 0, 0)
    }
    fn zyxy(&self) -> Self::Vec4 {
        self.permute4(2, 1, 0, 1)
    }
    fn zyxz(&self) -> Self::Vec4 {
        self.permute4(2, 1, 0, 2)
    }
    fn zyyx(&self) -> Self::Vec4 {
        self.permute4(2, 1, 1, 0)
    }
    fn zyyy(&self) -> Self::Vec4 {
        self.permute4(2, 1, 1, 1)
    }
    fn zyyz(&self) -> Self::Vec4 {
        self.permute4(2, 1, 1, 2)
    }
    fn zyzx(&self) -> Self::Vec4 {
        self.permute4(2, 1, 2, 0)
    }
    fn zyzy(&self) -> Self::Vec4 {
        self.permute4(2, 1, 2, 1)
    }
    fn zyzz(&self) -> Self::Vec4 {
        self.permute4(2, 1, 2, 2)
    }
    fn zzxx(&self) -> Self::Vec4 {
        self.permute4(2, 2, 0, 0)
    }
    fn zzxy(&self) -> Self::Vec4 {
        self.permute4(2, 2, 0, 1)
    }
    fn zzxz(&self) -> Self::Vec4 {
        self.permute4(2, 2, 0, 2)
    }
    fn zzyx(&self) -> Self::Vec4 {
        self.permute4(2, 2, 1, 0)
    }
    fn zzyy(&self) -> Self::Vec4 {
        self.permute4(2, 2, 1, 1)
    }
    fn zzyz(&self) -> Self::Vec4 {
        self.permute4(2, 2, 1, 2)
    }
    fn zzzx(&self) -> Self::Vec4 {
        self.permute4(2, 2, 2, 0)
    }
    fn zzzy(&self) -> Self::Vec4 {
        self.permute4(2, 2, 2, 1)
    }
    fn zzzz(&self) -> Self::Vec4 {
        self.permute4(2, 2, 2, 2)
    }
}
pub trait Vec4Swizzle {
    type Vec2;
    type Vec3;
    type Vec4;
    fn permute2(&self, x: u32, y: u32) -> Self::Vec2;
    fn permute3(&self, x: u32, y: u32, z: u32) -> Self::Vec3;
    fn permute4(&self, x: u32, y: u32, z: u32, w: u32) -> Self::Vec4;
    fn xx(&self) -> Self::Vec2 {
        self.permute2(0, 0)
    }
    fn xy(&self) -> Self::Vec2 {
        self.permute2(0, 1)
    }
    fn xz(&self) -> Self::Vec2 {
        self.permute2(0, 2)
    }
    fn xw(&self) -> Self::Vec2 {
        self.permute2(0, 3)
    }
    fn yx(&self) -> Self::Vec2 {
        self.permute2(1, 0)
    }
    fn yy(&self) -> Self::Vec2 {
        self.permute2(1, 1)
    }
    fn yz(&self) -> Self::Vec2 {
        self.permute2(1, 2)
    }
    fn yw(&self) -> Self::Vec2 {
        self.permute2(1, 3)
    }
    fn zx(&self) -> Self::Vec2 {
        self.permute2(2, 0)
    }
    fn zy(&self) -> Self::Vec2 {
        self.permute2(2, 1)
    }
    fn zz(&self) -> Self::Vec2 {
        self.permute2(2, 2)
    }
    fn zw(&self) -> Self::Vec2 {
        self.permute2(2, 3)
    }
    fn wx(&self) -> Self::Vec2 {
        self.permute2(3, 0)
    }
    fn wy(&self) -> Self::Vec2 {
        self.permute2(3, 1)
    }
    fn wz(&self) -> Self::Vec2 {
        self.permute2(3, 2)
    }
    fn ww(&self) -> Self::Vec2 {
        self.permute2(3, 3)
    }
    fn xxx(&self) -> Self::Vec3 {
        self.permute3(0, 0, 0)
    }
    fn xxy(&self) -> Self::Vec3 {
        self.permute3(0, 0, 1)
    }
    fn xxz(&self) -> Self::Vec3 {
        self.permute3(0, 0, 2)
    }
    fn xxw(&self) -> Self::Vec3 {
        self.permute3(0, 0, 3)
    }
    fn xyx(&self) -> Self::Vec3 {
        self.permute3(0, 1, 0)
    }
    fn xyy(&self) -> Self::Vec3 {
        self.permute3(0, 1, 1)
    }
    fn xyz(&self) -> Self::Vec3 {
        self.permute3(0, 1, 2)
    }
    fn xyw(&self) -> Self::Vec3 {
        self.permute3(0, 1, 3)
    }
    fn xzx(&self) -> Self::Vec3 {
        self.permute3(0, 2, 0)
    }
    fn xzy(&self) -> Self::Vec3 {
        self.permute3(0, 2, 1)
    }
    fn xzz(&self) -> Self::Vec3 {
        self.permute3(0, 2, 2)
    }
    fn xzw(&self) -> Self::Vec3 {
        self.permute3(0, 2, 3)
    }
    fn xwx(&self) -> Self::Vec3 {
        self.permute3(0, 3, 0)
    }
    fn xwy(&self) -> Self::Vec3 {
        self.permute3(0, 3, 1)
    }
    fn xwz(&self) -> Self::Vec3 {
        self.permute3(0, 3, 2)
    }
    fn xww(&self) -> Self::Vec3 {
        self.permute3(0, 3, 3)
    }
    fn yxx(&self) -> Self::Vec3 {
        self.permute3(1, 0, 0)
    }
    fn yxy(&self) -> Self::Vec3 {
        self.permute3(1, 0, 1)
    }
    fn yxz(&self) -> Self::Vec3 {
        self.permute3(1, 0, 2)
    }
    fn yxw(&self) -> Self::Vec3 {
        self.permute3(1, 0, 3)
    }
    fn yyx(&self) -> Self::Vec3 {
        self.permute3(1, 1, 0)
    }
    fn yyy(&self) -> Self::Vec3 {
        self.permute3(1, 1, 1)
    }
    fn yyz(&self) -> Self::Vec3 {
        self.permute3(1, 1, 2)
    }
    fn yyw(&self) -> Self::Vec3 {
        self.permute3(1, 1, 3)
    }
    fn yzx(&self) -> Self::Vec3 {
        self.permute3(1, 2, 0)
    }
    fn yzy(&self) -> Self::Vec3 {
        self.permute3(1, 2, 1)
    }
    fn yzz(&self) -> Self::Vec3 {
        self.permute3(1, 2, 2)
    }
    fn yzw(&self) -> Self::Vec3 {
        self.permute3(1, 2, 3)
    }
    fn ywx(&self) -> Self::Vec3 {
        self.permute3(1, 3, 0)
    }
    fn ywy(&self) -> Self::Vec3 {
        self.permute3(1, 3, 1)
    }
    fn ywz(&self) -> Self::Vec3 {
        self.permute3(1, 3, 2)
    }
    fn yww(&self) -> Self::Vec3 {
        self.permute3(1, 3, 3)
    }
    fn zxx(&self) -> Self::Vec3 {
        self.permute3(2, 0, 0)
    }
    fn zxy(&self) -> Self::Vec3 {
        self.permute3(2, 0, 1)
    }
    fn zxz(&self) -> Self::Vec3 {
        self.permute3(2, 0, 2)
    }
    fn zxw(&self) -> Self::Vec3 {
        self.permute3(2, 0, 3)
    }
    fn zyx(&self) -> Self::Vec3 {
        self.permute3(2, 1, 0)
    }
    fn zyy(&self) -> Self::Vec3 {
        self.permute3(2, 1, 1)
    }
    fn zyz(&self) -> Self::Vec3 {
        self.permute3(2, 1, 2)
    }
    fn zyw(&self) -> Self::Vec3 {
        self.permute3(2, 1, 3)
    }
    fn zzx(&self) -> Self::Vec3 {
        self.permute3(2, 2, 0)
    }
    fn zzy(&self) -> Self::Vec3 {
        self.permute3(2, 2, 1)
    }
    fn zzz(&self) -> Self::Vec3 {
        self.permute3(2, 2, 2)
    }
    fn zzw(&self) -> Self::Vec3 {
        self.permute3(2, 2, 3)
    }
    fn zwx(&self) -> Self::Vec3 {
        self.permute3(2, 3, 0)
    }
    fn zwy(&self) -> Self::Vec3 {
        self.permute3(2, 3, 1)
    }
    fn zwz(&self) -> Self::Vec3 {
        self.permute3(2, 3, 2)
    }
    fn zww(&self) -> Self::Vec3 {
        self.permute3(2, 3, 3)
    }
    fn wxx(&self) -> Self::Vec3 {
        self.permute3(3, 0, 0)
    }
    fn wxy(&self) -> Self::Vec3 {
        self.permute3(3, 0, 1)
    }
    fn wxz(&self) -> Self::Vec3 {
        self.permute3(3, 0, 2)
    }
    fn wxw(&self) -> Self::Vec3 {
        self.permute3(3, 0, 3)
    }
    fn wyx(&self) -> Self::Vec3 {
        self.permute3(3, 1, 0)
    }
    fn wyy(&self) -> Self::Vec3 {
        self.permute3(3, 1, 1)
    }
    fn wyz(&self) -> Self::Vec3 {
        self.permute3(3, 1, 2)
    }
    fn wyw(&self) -> Self::Vec3 {
        self.permute3(3, 1, 3)
    }
    fn wzx(&self) -> Self::Vec3 {
        self.permute3(3, 2, 0)
    }
    fn wzy(&self) -> Self::Vec3 {
        self.permute3(3, 2, 1)
    }
    fn wzz(&self) -> Self::Vec3 {
        self.permute3(3, 2, 2)
    }
    fn wzw(&self) -> Self::Vec3 {
        self.permute3(3, 2, 3)
    }
    fn wwx(&self) -> Self::Vec3 {
        self.permute3(3, 3, 0)
    }
    fn wwy(&self) -> Self::Vec3 {
        self.permute3(3, 3, 1)
    }
    fn wwz(&self) -> Self::Vec3 {
        self.permute3(3, 3, 2)
    }
    fn www(&self) -> Self::Vec3 {
        self.permute3(3, 3, 3)
    }
    fn xxxx(&self) -> Self::Vec4 {
        self.permute4(0, 0, 0, 0)
    }
    fn xxxy(&self) -> Self::Vec4 {
        self.permute4(0, 0, 0, 1)
    }
    fn xxxz(&self) -> Self::Vec4 {
        self.permute4(0, 0, 0, 2)
    }
    fn xxxw(&self) -> Self::Vec4 {
        self.permute4(0, 0, 0, 3)
    }
    fn xxyx(&self) -> Self::Vec4 {
        self.permute4(0, 0, 1, 0)
    }
    fn xxyy(&self) -> Self::Vec4 {
        self.permute4(0, 0, 1, 1)
    }
    fn xxyz(&self) -> Self::Vec4 {
        self.permute4(0, 0, 1, 2)
    }
    fn xxyw(&self) -> Self::Vec4 {
        self.permute4(0, 0, 1, 3)
    }
    fn xxzx(&self) -> Self::Vec4 {
        self.permute4(0, 0, 2, 0)
    }
    fn xxzy(&self) -> Self::Vec4 {
        self.permute4(0, 0, 2, 1)
    }
    fn xxzz(&self) -> Self::Vec4 {
        self.permute4(0, 0, 2, 2)
    }
    fn xxzw(&self) -> Self::Vec4 {
        self.permute4(0, 0, 2, 3)
    }
    fn xxwx(&self) -> Self::Vec4 {
        self.permute4(0, 0, 3, 0)
    }
    fn xxwy(&self) -> Self::Vec4 {
        self.permute4(0, 0, 3, 1)
    }
    fn xxwz(&self) -> Self::Vec4 {
        self.permute4(0, 0, 3, 2)
    }
    fn xxww(&self) -> Self::Vec4 {
        self.permute4(0, 0, 3, 3)
    }
    fn xyxx(&self) -> Self::Vec4 {
        self.permute4(0, 1, 0, 0)
    }
    fn xyxy(&self) -> Self::Vec4 {
        self.permute4(0, 1, 0, 1)
    }
    fn xyxz(&self) -> Self::Vec4 {
        self.permute4(0, 1, 0, 2)
    }
    fn xyxw(&self) -> Self::Vec4 {
        self.permute4(0, 1, 0, 3)
    }
    fn xyyx(&self) -> Self::Vec4 {
        self.permute4(0, 1, 1, 0)
    }
    fn xyyy(&self) -> Self::Vec4 {
        self.permute4(0, 1, 1, 1)
    }
    fn xyyz(&self) -> Self::Vec4 {
        self.permute4(0, 1, 1, 2)
    }
    fn xyyw(&self) -> Self::Vec4 {
        self.permute4(0, 1, 1, 3)
    }
    fn xyzx(&self) -> Self::Vec4 {
        self.permute4(0, 1, 2, 0)
    }
    fn xyzy(&self) -> Self::Vec4 {
        self.permute4(0, 1, 2, 1)
    }
    fn xyzz(&self) -> Self::Vec4 {
        self.permute4(0, 1, 2, 2)
    }
    fn xyzw(&self) -> Self::Vec4 {
        self.permute4(0, 1, 2, 3)
    }
    fn xywx(&self) -> Self::Vec4 {
        self.permute4(0, 1, 3, 0)
    }
    fn xywy(&self) -> Self::Vec4 {
        self.permute4(0, 1, 3, 1)
    }
    fn xywz(&self) -> Self::Vec4 {
        self.permute4(0, 1, 3, 2)
    }
    fn xyww(&self) -> Self::Vec4 {
        self.permute4(0, 1, 3, 3)
    }
    fn xzxx(&self) -> Self::Vec4 {
        self.permute4(0, 2, 0, 0)
    }
    fn xzxy(&self) -> Self::Vec4 {
        self.permute4(0, 2, 0, 1)
    }
    fn xzxz(&self) -> Self::Vec4 {
        self.permute4(0, 2, 0, 2)
    }
    fn xzxw(&self) -> Self::Vec4 {
        self.permute4(0, 2, 0, 3)
    }
    fn xzyx(&self) -> Self::Vec4 {
        self.permute4(0, 2, 1, 0)
    }
    fn xzyy(&self) -> Self::Vec4 {
        self.permute4(0, 2, 1, 1)
    }
    fn xzyz(&self) -> Self::Vec4 {
        self.permute4(0, 2, 1, 2)
    }
    fn xzyw(&self) -> Self::Vec4 {
        self.permute4(0, 2, 1, 3)
    }
    fn xzzx(&self) -> Self::Vec4 {
        self.permute4(0, 2, 2, 0)
    }
    fn xzzy(&self) -> Self::Vec4 {
        self.permute4(0, 2, 2, 1)
    }
    fn xzzz(&self) -> Self::Vec4 {
        self.permute4(0, 2, 2, 2)
    }
    fn xzzw(&self) -> Self::Vec4 {
        self.permute4(0, 2, 2, 3)
    }
    fn xzwx(&self) -> Self::Vec4 {
        self.permute4(0, 2, 3, 0)
    }
    fn xzwy(&self) -> Self::Vec4 {
        self.permute4(0, 2, 3, 1)
    }
    fn xzwz(&self) -> Self::Vec4 {
        self.permute4(0, 2, 3, 2)
    }
    fn xzww(&self) -> Self::Vec4 {
        self.permute4(0, 2, 3, 3)
    }
    fn xwxx(&self) -> Self::Vec4 {
        self.permute4(0, 3, 0, 0)
    }
    fn xwxy(&self) -> Self::Vec4 {
        self.permute4(0, 3, 0, 1)
    }
    fn xwxz(&self) -> Self::Vec4 {
        self.permute4(0, 3, 0, 2)
    }
    fn xwxw(&self) -> Self::Vec4 {
        self.permute4(0, 3, 0, 3)
    }
    fn xwyx(&self) -> Self::Vec4 {
        self.permute4(0, 3, 1, 0)
    }
    fn xwyy(&self) -> Self::Vec4 {
        self.permute4(0, 3, 1, 1)
    }
    fn xwyz(&self) -> Self::Vec4 {
        self.permute4(0, 3, 1, 2)
    }
    fn xwyw(&self) -> Self::Vec4 {
        self.permute4(0, 3, 1, 3)
    }
    fn xwzx(&self) -> Self::Vec4 {
        self.permute4(0, 3, 2, 0)
    }
    fn xwzy(&self) -> Self::Vec4 {
        self.permute4(0, 3, 2, 1)
    }
    fn xwzz(&self) -> Self::Vec4 {
        self.permute4(0, 3, 2, 2)
    }
    fn xwzw(&self) -> Self::Vec4 {
        self.permute4(0, 3, 2, 3)
    }
    fn xwwx(&self) -> Self::Vec4 {
        self.permute4(0, 3, 3, 0)
    }
    fn xwwy(&self) -> Self::Vec4 {
        self.permute4(0, 3, 3, 1)
    }
    fn xwwz(&self) -> Self::Vec4 {
        self.permute4(0, 3, 3, 2)
    }
    fn xwww(&self) -> Self::Vec4 {
        self.permute4(0, 3, 3, 3)
    }
    fn yxxx(&self) -> Self::Vec4 {
        self.permute4(1, 0, 0, 0)
    }
    fn yxxy(&self) -> Self::Vec4 {
        self.permute4(1, 0, 0, 1)
    }
    fn yxxz(&self) -> Self::Vec4 {
        self.permute4(1, 0, 0, 2)
    }
    fn yxxw(&self) -> Self::Vec4 {
        self.permute4(1, 0, 0, 3)
    }
    fn yxyx(&self) -> Self::Vec4 {
        self.permute4(1, 0, 1, 0)
    }
    fn yxyy(&self) -> Self::Vec4 {
        self.permute4(1, 0, 1, 1)
    }
    fn yxyz(&self) -> Self::Vec4 {
        self.permute4(1, 0, 1, 2)
    }
    fn yxyw(&self) -> Self::Vec4 {
        self.permute4(1, 0, 1, 3)
    }
    fn yxzx(&self) -> Self::Vec4 {
        self.permute4(1, 0, 2, 0)
    }
    fn yxzy(&self) -> Self::Vec4 {
        self.permute4(1, 0, 2, 1)
    }
    fn yxzz(&self) -> Self::Vec4 {
        self.permute4(1, 0, 2, 2)
    }
    fn yxzw(&self) -> Self::Vec4 {
        self.permute4(1, 0, 2, 3)
    }
    fn yxwx(&self) -> Self::Vec4 {
        self.permute4(1, 0, 3, 0)
    }
    fn yxwy(&self) -> Self::Vec4 {
        self.permute4(1, 0, 3, 1)
    }
    fn yxwz(&self) -> Self::Vec4 {
        self.permute4(1, 0, 3, 2)
    }
    fn yxww(&self) -> Self::Vec4 {
        self.permute4(1, 0, 3, 3)
    }
    fn yyxx(&self) -> Self::Vec4 {
        self.permute4(1, 1, 0, 0)
    }
    fn yyxy(&self) -> Self::Vec4 {
        self.permute4(1, 1, 0, 1)
    }
    fn yyxz(&self) -> Self::Vec4 {
        self.permute4(1, 1, 0, 2)
    }
    fn yyxw(&self) -> Self::Vec4 {
        self.permute4(1, 1, 0, 3)
    }
    fn yyyx(&self) -> Self::Vec4 {
        self.permute4(1, 1, 1, 0)
    }
    fn yyyy(&self) -> Self::Vec4 {
        self.permute4(1, 1, 1, 1)
    }
    fn yyyz(&self) -> Self::Vec4 {
        self.permute4(1, 1, 1, 2)
    }
    fn yyyw(&self) -> Self::Vec4 {
        self.permute4(1, 1, 1, 3)
    }
    fn yyzx(&self) -> Self::Vec4 {
        self.permute4(1, 1, 2, 0)
    }
    fn yyzy(&self) -> Self::Vec4 {
        self.permute4(1, 1, 2, 1)
    }
    fn yyzz(&self) -> Self::Vec4 {
        self.permute4(1, 1, 2, 2)
    }
    fn yyzw(&self) -> Self::Vec4 {
        self.permute4(1, 1, 2, 3)
    }
    fn yywx(&self) -> Self::Vec4 {
        self.permute4(1, 1, 3, 0)
    }
    fn yywy(&self) -> Self::Vec4 {
        self.permute4(1, 1, 3, 1)
    }
    fn yywz(&self) -> Self::Vec4 {
        self.permute4(1, 1, 3, 2)
    }
    fn yyww(&self) -> Self::Vec4 {
        self.permute4(1, 1, 3, 3)
    }
    fn yzxx(&self) -> Self::Vec4 {
        self.permute4(1, 2, 0, 0)
    }
    fn yzxy(&self) -> Self::Vec4 {
        self.permute4(1, 2, 0, 1)
    }
    fn yzxz(&self) -> Self::Vec4 {
        self.permute4(1, 2, 0, 2)
    }
    fn yzxw(&self) -> Self::Vec4 {
        self.permute4(1, 2, 0, 3)
    }
    fn yzyx(&self) -> Self::Vec4 {
        self.permute4(1, 2, 1, 0)
    }
    fn yzyy(&self) -> Self::Vec4 {
        self.permute4(1, 2, 1, 1)
    }
    fn yzyz(&self) -> Self::Vec4 {
        self.permute4(1, 2, 1, 2)
    }
    fn yzyw(&self) -> Self::Vec4 {
        self.permute4(1, 2, 1, 3)
    }
    fn yzzx(&self) -> Self::Vec4 {
        self.permute4(1, 2, 2, 0)
    }
    fn yzzy(&self) -> Self::Vec4 {
        self.permute4(1, 2, 2, 1)
    }
    fn yzzz(&self) -> Self::Vec4 {
        self.permute4(1, 2, 2, 2)
    }
    fn yzzw(&self) -> Self::Vec4 {
        self.permute4(1, 2, 2, 3)
    }
    fn yzwx(&self) -> Self::Vec4 {
        self.permute4(1, 2, 3, 0)
    }
    fn yzwy(&self) -> Self::Vec4 {
        self.permute4(1, 2, 3, 1)
    }
    fn yzwz(&self) -> Self::Vec4 {
        self.permute4(1, 2, 3, 2)
    }
    fn yzww(&self) -> Self::Vec4 {
        self.permute4(1, 2, 3, 3)
    }
    fn ywxx(&self) -> Self::Vec4 {
        self.permute4(1, 3, 0, 0)
    }
    fn ywxy(&self) -> Self::Vec4 {
        self.permute4(1, 3, 0, 1)
    }
    fn ywxz(&self) -> Self::Vec4 {
        self.permute4(1, 3, 0, 2)
    }
    fn ywxw(&self) -> Self::Vec4 {
        self.permute4(1, 3, 0, 3)
    }
    fn ywyx(&self) -> Self::Vec4 {
        self.permute4(1, 3, 1, 0)
    }
    fn ywyy(&self) -> Self::Vec4 {
        self.permute4(1, 3, 1, 1)
    }
    fn ywyz(&self) -> Self::Vec4 {
        self.permute4(1, 3, 1, 2)
    }
    fn ywyw(&self) -> Self::Vec4 {
        self.permute4(1, 3, 1, 3)
    }
    fn ywzx(&self) -> Self::Vec4 {
        self.permute4(1, 3, 2, 0)
    }
    fn ywzy(&self) -> Self::Vec4 {
        self.permute4(1, 3, 2, 1)
    }
    fn ywzz(&self) -> Self::Vec4 {
        self.permute4(1, 3, 2, 2)
    }
    fn ywzw(&self) -> Self::Vec4 {
        self.permute4(1, 3, 2, 3)
    }
    fn ywwx(&self) -> Self::Vec4 {
        self.permute4(1, 3, 3, 0)
    }
    fn ywwy(&self) -> Self::Vec4 {
        self.permute4(1, 3, 3, 1)
    }
    fn ywwz(&self) -> Self::Vec4 {
        self.permute4(1, 3, 3, 2)
    }
    fn ywww(&self) -> Self::Vec4 {
        self.permute4(1, 3, 3, 3)
    }
    fn zxxx(&self) -> Self::Vec4 {
        self.permute4(2, 0, 0, 0)
    }
    fn zxxy(&self) -> Self::Vec4 {
        self.permute4(2, 0, 0, 1)
    }
    fn zxxz(&self) -> Self::Vec4 {
        self.permute4(2, 0, 0, 2)
    }
    fn zxxw(&self) -> Self::Vec4 {
        self.permute4(2, 0, 0, 3)
    }
    fn zxyx(&self) -> Self::Vec4 {
        self.permute4(2, 0, 1, 0)
    }
    fn zxyy(&self) -> Self::Vec4 {
        self.permute4(2, 0, 1, 1)
    }
    fn zxyz(&self) -> Self::Vec4 {
        self.permute4(2, 0, 1, 2)
    }
    fn zxyw(&self) -> Self::Vec4 {
        self.permute4(2, 0, 1, 3)
    }
    fn zxzx(&self) -> Self::Vec4 {
        self.permute4(2, 0, 2, 0)
    }
    fn zxzy(&self) -> Self::Vec4 {
        self.permute4(2, 0, 2, 1)
    }
    fn zxzz(&self) -> Self::Vec4 {
        self.permute4(2, 0, 2, 2)
    }
    fn zxzw(&self) -> Self::Vec4 {
        self.permute4(2, 0, 2, 3)
    }
    fn zxwx(&self) -> Self::Vec4 {
        self.permute4(2, 0, 3, 0)
    }
    fn zxwy(&self) -> Self::Vec4 {
        self.permute4(2, 0, 3, 1)
    }
    fn zxwz(&self) -> Self::Vec4 {
        self.permute4(2, 0, 3, 2)
    }
    fn zxww(&self) -> Self::Vec4 {
        self.permute4(2, 0, 3, 3)
    }
    fn zyxx(&self) -> Self::Vec4 {
        self.permute4(2, 1, 0, 0)
    }
    fn zyxy(&self) -> Self::Vec4 {
        self.permute4(2, 1, 0, 1)
    }
    fn zyxz(&self) -> Self::Vec4 {
        self.permute4(2, 1, 0, 2)
    }
    fn zyxw(&self) -> Self::Vec4 {
        self.permute4(2, 1, 0, 3)
    }
    fn zyyx(&self) -> Self::Vec4 {
        self.permute4(2, 1, 1, 0)
    }
    fn zyyy(&self) -> Self::Vec4 {
        self.permute4(2, 1, 1, 1)
    }
    fn zyyz(&self) -> Self::Vec4 {
        self.permute4(2, 1, 1, 2)
    }
    fn zyyw(&self) -> Self::Vec4 {
        self.permute4(2, 1, 1, 3)
    }
    fn zyzx(&self) -> Self::Vec4 {
        self.permute4(2, 1, 2, 0)
    }
    fn zyzy(&self) -> Self::Vec4 {
        self.permute4(2, 1, 2, 1)
    }
    fn zyzz(&self) -> Self::Vec4 {
        self.permute4(2, 1, 2, 2)
    }
    fn zyzw(&self) -> Self::Vec4 {
        self.permute4(2, 1, 2, 3)
    }
    fn zywx(&self) -> Self::Vec4 {
        self.permute4(2, 1, 3, 0)
    }
    fn zywy(&self) -> Self::Vec4 {
        self.permute4(2, 1, 3, 1)
    }
    fn zywz(&self) -> Self::Vec4 {
        self.permute4(2, 1, 3, 2)
    }
    fn zyww(&self) -> Self::Vec4 {
        self.permute4(2, 1, 3, 3)
    }
    fn zzxx(&self) -> Self::Vec4 {
        self.permute4(2, 2, 0, 0)
    }
    fn zzxy(&self) -> Self::Vec4 {
        self.permute4(2, 2, 0, 1)
    }
    fn zzxz(&self) -> Self::Vec4 {
        self.permute4(2, 2, 0, 2)
    }
    fn zzxw(&self) -> Self::Vec4 {
        self.permute4(2, 2, 0, 3)
    }
    fn zzyx(&self) -> Self::Vec4 {
        self.permute4(2, 2, 1, 0)
    }
    fn zzyy(&self) -> Self::Vec4 {
        self.permute4(2, 2, 1, 1)
    }
    fn zzyz(&self) -> Self::Vec4 {
        self.permute4(2, 2, 1, 2)
    }
    fn zzyw(&self) -> Self::Vec4 {
        self.permute4(2, 2, 1, 3)
    }
    fn zzzx(&self) -> Self::Vec4 {
        self.permute4(2, 2, 2, 0)
    }
    fn zzzy(&self) -> Self::Vec4 {
        self.permute4(2, 2, 2, 1)
    }
    fn zzzz(&self) -> Self::Vec4 {
        self.permute4(2, 2, 2, 2)
    }
    fn zzzw(&self) -> Self::Vec4 {
        self.permute4(2, 2, 2, 3)
    }
    fn zzwx(&self) -> Self::Vec4 {
        self.permute4(2, 2, 3, 0)
    }
    fn zzwy(&self) -> Self::Vec4 {
        self.permute4(2, 2, 3, 1)
    }
    fn zzwz(&self) -> Self::Vec4 {
        self.permute4(2, 2, 3, 2)
    }
    fn zzww(&self) -> Self::Vec4 {
        self.permute4(2, 2, 3, 3)
    }
    fn zwxx(&self) -> Self::Vec4 {
        self.permute4(2, 3, 0, 0)
    }
    fn zwxy(&self) -> Self::Vec4 {
        self.permute4(2, 3, 0, 1)
    }
    fn zwxz(&self) -> Self::Vec4 {
        self.permute4(2, 3, 0, 2)
    }
    fn zwxw(&self) -> Self::Vec4 {
        self.permute4(2, 3, 0, 3)
    }
    fn zwyx(&self) -> Self::Vec4 {
        self.permute4(2, 3, 1, 0)
    }
    fn zwyy(&self) -> Self::Vec4 {
        self.permute4(2, 3, 1, 1)
    }
    fn zwyz(&self) -> Self::Vec4 {
        self.permute4(2, 3, 1, 2)
    }
    fn zwyw(&self) -> Self::Vec4 {
        self.permute4(2, 3, 1, 3)
    }
    fn zwzx(&self) -> Self::Vec4 {
        self.permute4(2, 3, 2, 0)
    }
    fn zwzy(&self) -> Self::Vec4 {
        self.permute4(2, 3, 2, 1)
    }
    fn zwzz(&self) -> Self::Vec4 {
        self.permute4(2, 3, 2, 2)
    }
    fn zwzw(&self) -> Self::Vec4 {
        self.permute4(2, 3, 2, 3)
    }
    fn zwwx(&self) -> Self::Vec4 {
        self.permute4(2, 3, 3, 0)
    }
    fn zwwy(&self) -> Self::Vec4 {
        self.permute4(2, 3, 3, 1)
    }
    fn zwwz(&self) -> Self::Vec4 {
        self.permute4(2, 3, 3, 2)
    }
    fn zwww(&self) -> Self::Vec4 {
        self.permute4(2, 3, 3, 3)
    }
    fn wxxx(&self) -> Self::Vec4 {
        self.permute4(3, 0, 0, 0)
    }
    fn wxxy(&self) -> Self::Vec4 {
        self.permute4(3, 0, 0, 1)
    }
    fn wxxz(&self) -> Self::Vec4 {
        self.permute4(3, 0, 0, 2)
    }
    fn wxxw(&self) -> Self::Vec4 {
        self.permute4(3, 0, 0, 3)
    }
    fn wxyx(&self) -> Self::Vec4 {
        self.permute4(3, 0, 1, 0)
    }
    fn wxyy(&self) -> Self::Vec4 {
        self.permute4(3, 0, 1, 1)
    }
    fn wxyz(&self) -> Self::Vec4 {
        self.permute4(3, 0, 1, 2)
    }
    fn wxyw(&self) -> Self::Vec4 {
        self.permute4(3, 0, 1, 3)
    }
    fn wxzx(&self) -> Self::Vec4 {
        self.permute4(3, 0, 2, 0)
    }
    fn wxzy(&self) -> Self::Vec4 {
        self.permute4(3, 0, 2, 1)
    }
    fn wxzz(&self) -> Self::Vec4 {
        self.permute4(3, 0, 2, 2)
    }
    fn wxzw(&self) -> Self::Vec4 {
        self.permute4(3, 0, 2, 3)
    }
    fn wxwx(&self) -> Self::Vec4 {
        self.permute4(3, 0, 3, 0)
    }
    fn wxwy(&self) -> Self::Vec4 {
        self.permute4(3, 0, 3, 1)
    }
    fn wxwz(&self) -> Self::Vec4 {
        self.permute4(3, 0, 3, 2)
    }
    fn wxww(&self) -> Self::Vec4 {
        self.permute4(3, 0, 3, 3)
    }
    fn wyxx(&self) -> Self::Vec4 {
        self.permute4(3, 1, 0, 0)
    }
    fn wyxy(&self) -> Self::Vec4 {
        self.permute4(3, 1, 0, 1)
    }
    fn wyxz(&self) -> Self::Vec4 {
        self.permute4(3, 1, 0, 2)
    }
    fn wyxw(&self) -> Self::Vec4 {
        self.permute4(3, 1, 0, 3)
    }
    fn wyyx(&self) -> Self::Vec4 {
        self.permute4(3, 1, 1, 0)
    }
    fn wyyy(&self) -> Self::Vec4 {
        self.permute4(3, 1, 1, 1)
    }
    fn wyyz(&self) -> Self::Vec4 {
        self.permute4(3, 1, 1, 2)
    }
    fn wyyw(&self) -> Self::Vec4 {
        self.permute4(3, 1, 1, 3)
    }
    fn wyzx(&self) -> Self::Vec4 {
        self.permute4(3, 1, 2, 0)
    }
    fn wyzy(&self) -> Self::Vec4 {
        self.permute4(3, 1, 2, 1)
    }
    fn wyzz(&self) -> Self::Vec4 {
        self.permute4(3, 1, 2, 2)
    }
    fn wyzw(&self) -> Self::Vec4 {
        self.permute4(3, 1, 2, 3)
    }
    fn wywx(&self) -> Self::Vec4 {
        self.permute4(3, 1, 3, 0)
    }
    fn wywy(&self) -> Self::Vec4 {
        self.permute4(3, 1, 3, 1)
    }
    fn wywz(&self) -> Self::Vec4 {
        self.permute4(3, 1, 3, 2)
    }
    fn wyww(&self) -> Self::Vec4 {
        self.permute4(3, 1, 3, 3)
    }
    fn wzxx(&self) -> Self::Vec4 {
        self.permute4(3, 2, 0, 0)
    }
    fn wzxy(&self) -> Self::Vec4 {
        self.permute4(3, 2, 0, 1)
    }
    fn wzxz(&self) -> Self::Vec4 {
        self.permute4(3, 2, 0, 2)
    }
    fn wzxw(&self) -> Self::Vec4 {
        self.permute4(3, 2, 0, 3)
    }
    fn wzyx(&self) -> Self::Vec4 {
        self.permute4(3, 2, 1, 0)
    }
    fn wzyy(&self) -> Self::Vec4 {
        self.permute4(3, 2, 1, 1)
    }
    fn wzyz(&self) -> Self::Vec4 {
        self.permute4(3, 2, 1, 2)
    }
    fn wzyw(&self) -> Self::Vec4 {
        self.permute4(3, 2, 1, 3)
    }
    fn wzzx(&self) -> Self::Vec4 {
        self.permute4(3, 2, 2, 0)
    }
    fn wzzy(&self) -> Self::Vec4 {
        self.permute4(3, 2, 2, 1)
    }
    fn wzzz(&self) -> Self::Vec4 {
        self.permute4(3, 2, 2, 2)
    }
    fn wzzw(&self) -> Self::Vec4 {
        self.permute4(3, 2, 2, 3)
    }
    fn wzwx(&self) -> Self::Vec4 {
        self.permute4(3, 2, 3, 0)
    }
    fn wzwy(&self) -> Self::Vec4 {
        self.permute4(3, 2, 3, 1)
    }
    fn wzwz(&self) -> Self::Vec4 {
        self.permute4(3, 2, 3, 2)
    }
    fn wzww(&self) -> Self::Vec4 {
        self.permute4(3, 2, 3, 3)
    }
    fn wwxx(&self) -> Self::Vec4 {
        self.permute4(3, 3, 0, 0)
    }
    fn wwxy(&self) -> Self::Vec4 {
        self.permute4(3, 3, 0, 1)
    }
    fn wwxz(&self) -> Self::Vec4 {
        self.permute4(3, 3, 0, 2)
    }
    fn wwxw(&self) -> Self::Vec4 {
        self.permute4(3, 3, 0, 3)
    }
    fn wwyx(&self) -> Self::Vec4 {
        self.permute4(3, 3, 1, 0)
    }
    fn wwyy(&self) -> Self::Vec4 {
        self.permute4(3, 3, 1, 1)
    }
    fn wwyz(&self) -> Self::Vec4 {
        self.permute4(3, 3, 1, 2)
    }
    fn wwyw(&self) -> Self::Vec4 {
        self.permute4(3, 3, 1, 3)
    }
    fn wwzx(&self) -> Self::Vec4 {
        self.permute4(3, 3, 2, 0)
    }
    fn wwzy(&self) -> Self::Vec4 {
        self.permute4(3, 3, 2, 1)
    }
    fn wwzz(&self) -> Self::Vec4 {
        self.permute4(3, 3, 2, 2)
    }
    fn wwzw(&self) -> Self::Vec4 {
        self.permute4(3, 3, 2, 3)
    }
    fn wwwx(&self) -> Self::Vec4 {
        self.permute4(3, 3, 3, 0)
    }
    fn wwwy(&self) -> Self::Vec4 {
        self.permute4(3, 3, 3, 1)
    }
    fn wwwz(&self) -> Self::Vec4 {
        self.permute4(3, 3, 3, 2)
    }
    fn wwww(&self) -> Self::Vec4 {
        self.permute4(3, 3, 3, 3)
    }
}
