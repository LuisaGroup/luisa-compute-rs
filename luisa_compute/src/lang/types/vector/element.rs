use super::*;

macro_rules! element {
    ($T:ty [ 2 ]: $A: ident) => {
        impl VectorAlign<2> for $T {
            type A = $A;
            type VectorExpr = VectorExprProxy2<$T>;
            type VectorVar = VectorVarProxy2<$T>;
            type VectorAtomicRef = VectorAtomicRefProxy2<$T>;
            type VectorExprData = VectorExprData<$T, 2>;
            type VectorVarData = VectorVarData<$T, 2>;
          
        }
    };
    ($T:ty [ 3 ]: $A: ident) => {
        impl VectorAlign<3> for $T {
            type A = $A;
            type VectorExpr = VectorExprProxy3<$T>;
            type VectorVar = VectorVarProxy3<$T>;
            type VectorAtomicRef = VectorAtomicRefProxy3<$T>;
            type VectorExprData = DoubledProxyData<VectorExprData<$T, 3>>;
            type VectorVarData = DoubledProxyData<VectorVarData<$T, 3>>;
        }
    };
    ($T:ty [ 4 ]: $A: ident) => {
        impl VectorAlign<4> for $T {
            type A = $A;
            type VectorExpr = VectorExprProxy4<$T>;
            type VectorVar = VectorVarProxy4<$T>;
            type VectorAtomicRef = VectorAtomicRefProxy4<$T>;
            type VectorExprData = DoubledProxyData<VectorExprData<$T, 4>>;
            type VectorVarData = DoubledProxyData<VectorVarData<$T, 4>>;
        }
    };
}

element!(bool[2]: Align2);
element!(bool[3]: Align4);
element!(bool[4]: Align4);
element!(u8[2]: Align2);
element!(u8[3]: Align4);
element!(u8[4]: Align4);
element!(i8[2]: Align2);
element!(i8[3]: Align4);
element!(i8[4]: Align4);

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

element!(f64[2]: Align16);
element!(f64[3]: Align32);
element!(f64[4]: Align32);
element!(u64[2]: Align16);
element!(u64[3]: Align32);
element!(u64[4]: Align32);
element!(i64[2]: Align16);
element!(i64[3]: Align32);
element!(i64[4]: Align32);
