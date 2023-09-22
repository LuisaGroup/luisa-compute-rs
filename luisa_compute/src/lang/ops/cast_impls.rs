
#[rustfmt::skip]mod impl_{
use crate::prelude::*;
use super::super::*;

impl Expr<f32> {
    pub fn as_i32(self) -> Expr<i32> { self.as_::<i32>() }
    pub fn as_u32(self) -> Expr<u32> { self.as_::<u32>() }
    pub fn as_f64(self) -> Expr<f64> { self.as_::<f64>() }
    pub fn as_i64(self) -> Expr<i64> { self.as_::<i64>() }
    pub fn as_u64(self) -> Expr<u64> { self.as_::<u64>() }
    pub fn as_f16(self) -> Expr<f16> { self.as_::<f16>() }
    pub fn as_i16(self) -> Expr<i16> { self.as_::<i16>() }
    pub fn as_u16(self) -> Expr<u16> { self.as_::<u16>() }
    pub fn as_i8(self) -> Expr<i8> { self.as_::<i8>() }
    pub fn as_u8(self) -> Expr<u8> { self.as_::<u8>() }
}
impl Expr<Float2> {
    pub fn as_int2(self) -> Expr<Int2> { self.as_::<Int2>() }
    pub fn as_uint2(self) -> Expr<Uint2> { self.as_::<Uint2>() }
    pub fn as_double2(self) -> Expr<Double2> { self.as_::<Double2>() }
    pub fn as_long2(self) -> Expr<Long2> { self.as_::<Long2>() }
    pub fn as_ulong2(self) -> Expr<Ulong2> { self.as_::<Ulong2>() }
    pub fn as_half2(self) -> Expr<Half2> { self.as_::<Half2>() }
    pub fn as_short2(self) -> Expr<Short2> { self.as_::<Short2>() }
    pub fn as_ushort2(self) -> Expr<Ushort2> { self.as_::<Ushort2>() }
    pub fn as_byte2(self) -> Expr<Byte2> { self.as_::<Byte2>() }
    pub fn as_ubyte2(self) -> Expr<Ubyte2> { self.as_::<Ubyte2>() }
}
impl Expr<Float3> {
    pub fn as_int3(self) -> Expr<Int3> { self.as_::<Int3>() }
    pub fn as_uint3(self) -> Expr<Uint3> { self.as_::<Uint3>() }
    pub fn as_double3(self) -> Expr<Double3> { self.as_::<Double3>() }
    pub fn as_long3(self) -> Expr<Long3> { self.as_::<Long3>() }
    pub fn as_ulong3(self) -> Expr<Ulong3> { self.as_::<Ulong3>() }
    pub fn as_half3(self) -> Expr<Half3> { self.as_::<Half3>() }
    pub fn as_short3(self) -> Expr<Short3> { self.as_::<Short3>() }
    pub fn as_ushort3(self) -> Expr<Ushort3> { self.as_::<Ushort3>() }
    pub fn as_byte3(self) -> Expr<Byte3> { self.as_::<Byte3>() }
    pub fn as_ubyte3(self) -> Expr<Ubyte3> { self.as_::<Ubyte3>() }
}
impl Expr<Float4> {
    pub fn as_int4(self) -> Expr<Int4> { self.as_::<Int4>() }
    pub fn as_uint4(self) -> Expr<Uint4> { self.as_::<Uint4>() }
    pub fn as_double4(self) -> Expr<Double4> { self.as_::<Double4>() }
    pub fn as_long4(self) -> Expr<Long4> { self.as_::<Long4>() }
    pub fn as_ulong4(self) -> Expr<Ulong4> { self.as_::<Ulong4>() }
    pub fn as_half4(self) -> Expr<Half4> { self.as_::<Half4>() }
    pub fn as_short4(self) -> Expr<Short4> { self.as_::<Short4>() }
    pub fn as_ushort4(self) -> Expr<Ushort4> { self.as_::<Ushort4>() }
    pub fn as_byte4(self) -> Expr<Byte4> { self.as_::<Byte4>() }
    pub fn as_ubyte4(self) -> Expr<Ubyte4> { self.as_::<Ubyte4>() }
}
impl Expr<i32> {
    pub fn as_f32(self) -> Expr<f32> { self.as_::<f32>() }
    pub fn as_u32(self) -> Expr<u32> { self.as_::<u32>() }
    pub fn as_f64(self) -> Expr<f64> { self.as_::<f64>() }
    pub fn as_i64(self) -> Expr<i64> { self.as_::<i64>() }
    pub fn as_u64(self) -> Expr<u64> { self.as_::<u64>() }
    pub fn as_f16(self) -> Expr<f16> { self.as_::<f16>() }
    pub fn as_i16(self) -> Expr<i16> { self.as_::<i16>() }
    pub fn as_u16(self) -> Expr<u16> { self.as_::<u16>() }
    pub fn as_i8(self) -> Expr<i8> { self.as_::<i8>() }
    pub fn as_u8(self) -> Expr<u8> { self.as_::<u8>() }
}
impl Expr<Int2> {
    pub fn as_float2(self) -> Expr<Float2> { self.as_::<Float2>() }
    pub fn as_uint2(self) -> Expr<Uint2> { self.as_::<Uint2>() }
    pub fn as_double2(self) -> Expr<Double2> { self.as_::<Double2>() }
    pub fn as_long2(self) -> Expr<Long2> { self.as_::<Long2>() }
    pub fn as_ulong2(self) -> Expr<Ulong2> { self.as_::<Ulong2>() }
    pub fn as_half2(self) -> Expr<Half2> { self.as_::<Half2>() }
    pub fn as_short2(self) -> Expr<Short2> { self.as_::<Short2>() }
    pub fn as_ushort2(self) -> Expr<Ushort2> { self.as_::<Ushort2>() }
    pub fn as_byte2(self) -> Expr<Byte2> { self.as_::<Byte2>() }
    pub fn as_ubyte2(self) -> Expr<Ubyte2> { self.as_::<Ubyte2>() }
}
impl Expr<Int3> {
    pub fn as_float3(self) -> Expr<Float3> { self.as_::<Float3>() }
    pub fn as_uint3(self) -> Expr<Uint3> { self.as_::<Uint3>() }
    pub fn as_double3(self) -> Expr<Double3> { self.as_::<Double3>() }
    pub fn as_long3(self) -> Expr<Long3> { self.as_::<Long3>() }
    pub fn as_ulong3(self) -> Expr<Ulong3> { self.as_::<Ulong3>() }
    pub fn as_half3(self) -> Expr<Half3> { self.as_::<Half3>() }
    pub fn as_short3(self) -> Expr<Short3> { self.as_::<Short3>() }
    pub fn as_ushort3(self) -> Expr<Ushort3> { self.as_::<Ushort3>() }
    pub fn as_byte3(self) -> Expr<Byte3> { self.as_::<Byte3>() }
    pub fn as_ubyte3(self) -> Expr<Ubyte3> { self.as_::<Ubyte3>() }
}
impl Expr<Int4> {
    pub fn as_float4(self) -> Expr<Float4> { self.as_::<Float4>() }
    pub fn as_uint4(self) -> Expr<Uint4> { self.as_::<Uint4>() }
    pub fn as_double4(self) -> Expr<Double4> { self.as_::<Double4>() }
    pub fn as_long4(self) -> Expr<Long4> { self.as_::<Long4>() }
    pub fn as_ulong4(self) -> Expr<Ulong4> { self.as_::<Ulong4>() }
    pub fn as_half4(self) -> Expr<Half4> { self.as_::<Half4>() }
    pub fn as_short4(self) -> Expr<Short4> { self.as_::<Short4>() }
    pub fn as_ushort4(self) -> Expr<Ushort4> { self.as_::<Ushort4>() }
    pub fn as_byte4(self) -> Expr<Byte4> { self.as_::<Byte4>() }
    pub fn as_ubyte4(self) -> Expr<Ubyte4> { self.as_::<Ubyte4>() }
}
impl Expr<u32> {
    pub fn as_f32(self) -> Expr<f32> { self.as_::<f32>() }
    pub fn as_i32(self) -> Expr<i32> { self.as_::<i32>() }
    pub fn as_f64(self) -> Expr<f64> { self.as_::<f64>() }
    pub fn as_i64(self) -> Expr<i64> { self.as_::<i64>() }
    pub fn as_u64(self) -> Expr<u64> { self.as_::<u64>() }
    pub fn as_f16(self) -> Expr<f16> { self.as_::<f16>() }
    pub fn as_i16(self) -> Expr<i16> { self.as_::<i16>() }
    pub fn as_u16(self) -> Expr<u16> { self.as_::<u16>() }
    pub fn as_i8(self) -> Expr<i8> { self.as_::<i8>() }
    pub fn as_u8(self) -> Expr<u8> { self.as_::<u8>() }
}
impl Expr<Uint2> {
    pub fn as_float2(self) -> Expr<Float2> { self.as_::<Float2>() }
    pub fn as_int2(self) -> Expr<Int2> { self.as_::<Int2>() }
    pub fn as_double2(self) -> Expr<Double2> { self.as_::<Double2>() }
    pub fn as_long2(self) -> Expr<Long2> { self.as_::<Long2>() }
    pub fn as_ulong2(self) -> Expr<Ulong2> { self.as_::<Ulong2>() }
    pub fn as_half2(self) -> Expr<Half2> { self.as_::<Half2>() }
    pub fn as_short2(self) -> Expr<Short2> { self.as_::<Short2>() }
    pub fn as_ushort2(self) -> Expr<Ushort2> { self.as_::<Ushort2>() }
    pub fn as_byte2(self) -> Expr<Byte2> { self.as_::<Byte2>() }
    pub fn as_ubyte2(self) -> Expr<Ubyte2> { self.as_::<Ubyte2>() }
}
impl Expr<Uint3> {
    pub fn as_float3(self) -> Expr<Float3> { self.as_::<Float3>() }
    pub fn as_int3(self) -> Expr<Int3> { self.as_::<Int3>() }
    pub fn as_double3(self) -> Expr<Double3> { self.as_::<Double3>() }
    pub fn as_long3(self) -> Expr<Long3> { self.as_::<Long3>() }
    pub fn as_ulong3(self) -> Expr<Ulong3> { self.as_::<Ulong3>() }
    pub fn as_half3(self) -> Expr<Half3> { self.as_::<Half3>() }
    pub fn as_short3(self) -> Expr<Short3> { self.as_::<Short3>() }
    pub fn as_ushort3(self) -> Expr<Ushort3> { self.as_::<Ushort3>() }
    pub fn as_byte3(self) -> Expr<Byte3> { self.as_::<Byte3>() }
    pub fn as_ubyte3(self) -> Expr<Ubyte3> { self.as_::<Ubyte3>() }
}
impl Expr<Uint4> {
    pub fn as_float4(self) -> Expr<Float4> { self.as_::<Float4>() }
    pub fn as_int4(self) -> Expr<Int4> { self.as_::<Int4>() }
    pub fn as_double4(self) -> Expr<Double4> { self.as_::<Double4>() }
    pub fn as_long4(self) -> Expr<Long4> { self.as_::<Long4>() }
    pub fn as_ulong4(self) -> Expr<Ulong4> { self.as_::<Ulong4>() }
    pub fn as_half4(self) -> Expr<Half4> { self.as_::<Half4>() }
    pub fn as_short4(self) -> Expr<Short4> { self.as_::<Short4>() }
    pub fn as_ushort4(self) -> Expr<Ushort4> { self.as_::<Ushort4>() }
    pub fn as_byte4(self) -> Expr<Byte4> { self.as_::<Byte4>() }
    pub fn as_ubyte4(self) -> Expr<Ubyte4> { self.as_::<Ubyte4>() }
}
impl Expr<f64> {
    pub fn as_f32(self) -> Expr<f32> { self.as_::<f32>() }
    pub fn as_i32(self) -> Expr<i32> { self.as_::<i32>() }
    pub fn as_u32(self) -> Expr<u32> { self.as_::<u32>() }
    pub fn as_i64(self) -> Expr<i64> { self.as_::<i64>() }
    pub fn as_u64(self) -> Expr<u64> { self.as_::<u64>() }
    pub fn as_f16(self) -> Expr<f16> { self.as_::<f16>() }
    pub fn as_i16(self) -> Expr<i16> { self.as_::<i16>() }
    pub fn as_u16(self) -> Expr<u16> { self.as_::<u16>() }
    pub fn as_i8(self) -> Expr<i8> { self.as_::<i8>() }
    pub fn as_u8(self) -> Expr<u8> { self.as_::<u8>() }
}
impl Expr<Double2> {
    pub fn as_float2(self) -> Expr<Float2> { self.as_::<Float2>() }
    pub fn as_int2(self) -> Expr<Int2> { self.as_::<Int2>() }
    pub fn as_uint2(self) -> Expr<Uint2> { self.as_::<Uint2>() }
    pub fn as_long2(self) -> Expr<Long2> { self.as_::<Long2>() }
    pub fn as_ulong2(self) -> Expr<Ulong2> { self.as_::<Ulong2>() }
    pub fn as_half2(self) -> Expr<Half2> { self.as_::<Half2>() }
    pub fn as_short2(self) -> Expr<Short2> { self.as_::<Short2>() }
    pub fn as_ushort2(self) -> Expr<Ushort2> { self.as_::<Ushort2>() }
    pub fn as_byte2(self) -> Expr<Byte2> { self.as_::<Byte2>() }
    pub fn as_ubyte2(self) -> Expr<Ubyte2> { self.as_::<Ubyte2>() }
}
impl Expr<Double3> {
    pub fn as_float3(self) -> Expr<Float3> { self.as_::<Float3>() }
    pub fn as_int3(self) -> Expr<Int3> { self.as_::<Int3>() }
    pub fn as_uint3(self) -> Expr<Uint3> { self.as_::<Uint3>() }
    pub fn as_long3(self) -> Expr<Long3> { self.as_::<Long3>() }
    pub fn as_ulong3(self) -> Expr<Ulong3> { self.as_::<Ulong3>() }
    pub fn as_half3(self) -> Expr<Half3> { self.as_::<Half3>() }
    pub fn as_short3(self) -> Expr<Short3> { self.as_::<Short3>() }
    pub fn as_ushort3(self) -> Expr<Ushort3> { self.as_::<Ushort3>() }
    pub fn as_byte3(self) -> Expr<Byte3> { self.as_::<Byte3>() }
    pub fn as_ubyte3(self) -> Expr<Ubyte3> { self.as_::<Ubyte3>() }
}
impl Expr<Double4> {
    pub fn as_float4(self) -> Expr<Float4> { self.as_::<Float4>() }
    pub fn as_int4(self) -> Expr<Int4> { self.as_::<Int4>() }
    pub fn as_uint4(self) -> Expr<Uint4> { self.as_::<Uint4>() }
    pub fn as_long4(self) -> Expr<Long4> { self.as_::<Long4>() }
    pub fn as_ulong4(self) -> Expr<Ulong4> { self.as_::<Ulong4>() }
    pub fn as_half4(self) -> Expr<Half4> { self.as_::<Half4>() }
    pub fn as_short4(self) -> Expr<Short4> { self.as_::<Short4>() }
    pub fn as_ushort4(self) -> Expr<Ushort4> { self.as_::<Ushort4>() }
    pub fn as_byte4(self) -> Expr<Byte4> { self.as_::<Byte4>() }
    pub fn as_ubyte4(self) -> Expr<Ubyte4> { self.as_::<Ubyte4>() }
}
impl Expr<i64> {
    pub fn as_f32(self) -> Expr<f32> { self.as_::<f32>() }
    pub fn as_i32(self) -> Expr<i32> { self.as_::<i32>() }
    pub fn as_u32(self) -> Expr<u32> { self.as_::<u32>() }
    pub fn as_f64(self) -> Expr<f64> { self.as_::<f64>() }
    pub fn as_u64(self) -> Expr<u64> { self.as_::<u64>() }
    pub fn as_f16(self) -> Expr<f16> { self.as_::<f16>() }
    pub fn as_i16(self) -> Expr<i16> { self.as_::<i16>() }
    pub fn as_u16(self) -> Expr<u16> { self.as_::<u16>() }
    pub fn as_i8(self) -> Expr<i8> { self.as_::<i8>() }
    pub fn as_u8(self) -> Expr<u8> { self.as_::<u8>() }
}
impl Expr<Long2> {
    pub fn as_float2(self) -> Expr<Float2> { self.as_::<Float2>() }
    pub fn as_int2(self) -> Expr<Int2> { self.as_::<Int2>() }
    pub fn as_uint2(self) -> Expr<Uint2> { self.as_::<Uint2>() }
    pub fn as_double2(self) -> Expr<Double2> { self.as_::<Double2>() }
    pub fn as_ulong2(self) -> Expr<Ulong2> { self.as_::<Ulong2>() }
    pub fn as_half2(self) -> Expr<Half2> { self.as_::<Half2>() }
    pub fn as_short2(self) -> Expr<Short2> { self.as_::<Short2>() }
    pub fn as_ushort2(self) -> Expr<Ushort2> { self.as_::<Ushort2>() }
    pub fn as_byte2(self) -> Expr<Byte2> { self.as_::<Byte2>() }
    pub fn as_ubyte2(self) -> Expr<Ubyte2> { self.as_::<Ubyte2>() }
}
impl Expr<Long3> {
    pub fn as_float3(self) -> Expr<Float3> { self.as_::<Float3>() }
    pub fn as_int3(self) -> Expr<Int3> { self.as_::<Int3>() }
    pub fn as_uint3(self) -> Expr<Uint3> { self.as_::<Uint3>() }
    pub fn as_double3(self) -> Expr<Double3> { self.as_::<Double3>() }
    pub fn as_ulong3(self) -> Expr<Ulong3> { self.as_::<Ulong3>() }
    pub fn as_half3(self) -> Expr<Half3> { self.as_::<Half3>() }
    pub fn as_short3(self) -> Expr<Short3> { self.as_::<Short3>() }
    pub fn as_ushort3(self) -> Expr<Ushort3> { self.as_::<Ushort3>() }
    pub fn as_byte3(self) -> Expr<Byte3> { self.as_::<Byte3>() }
    pub fn as_ubyte3(self) -> Expr<Ubyte3> { self.as_::<Ubyte3>() }
}
impl Expr<Long4> {
    pub fn as_float4(self) -> Expr<Float4> { self.as_::<Float4>() }
    pub fn as_int4(self) -> Expr<Int4> { self.as_::<Int4>() }
    pub fn as_uint4(self) -> Expr<Uint4> { self.as_::<Uint4>() }
    pub fn as_double4(self) -> Expr<Double4> { self.as_::<Double4>() }
    pub fn as_ulong4(self) -> Expr<Ulong4> { self.as_::<Ulong4>() }
    pub fn as_half4(self) -> Expr<Half4> { self.as_::<Half4>() }
    pub fn as_short4(self) -> Expr<Short4> { self.as_::<Short4>() }
    pub fn as_ushort4(self) -> Expr<Ushort4> { self.as_::<Ushort4>() }
    pub fn as_byte4(self) -> Expr<Byte4> { self.as_::<Byte4>() }
    pub fn as_ubyte4(self) -> Expr<Ubyte4> { self.as_::<Ubyte4>() }
}
impl Expr<u64> {
    pub fn as_f32(self) -> Expr<f32> { self.as_::<f32>() }
    pub fn as_i32(self) -> Expr<i32> { self.as_::<i32>() }
    pub fn as_u32(self) -> Expr<u32> { self.as_::<u32>() }
    pub fn as_f64(self) -> Expr<f64> { self.as_::<f64>() }
    pub fn as_i64(self) -> Expr<i64> { self.as_::<i64>() }
    pub fn as_f16(self) -> Expr<f16> { self.as_::<f16>() }
    pub fn as_i16(self) -> Expr<i16> { self.as_::<i16>() }
    pub fn as_u16(self) -> Expr<u16> { self.as_::<u16>() }
    pub fn as_i8(self) -> Expr<i8> { self.as_::<i8>() }
    pub fn as_u8(self) -> Expr<u8> { self.as_::<u8>() }
}
impl Expr<Ulong2> {
    pub fn as_float2(self) -> Expr<Float2> { self.as_::<Float2>() }
    pub fn as_int2(self) -> Expr<Int2> { self.as_::<Int2>() }
    pub fn as_uint2(self) -> Expr<Uint2> { self.as_::<Uint2>() }
    pub fn as_double2(self) -> Expr<Double2> { self.as_::<Double2>() }
    pub fn as_long2(self) -> Expr<Long2> { self.as_::<Long2>() }
    pub fn as_half2(self) -> Expr<Half2> { self.as_::<Half2>() }
    pub fn as_short2(self) -> Expr<Short2> { self.as_::<Short2>() }
    pub fn as_ushort2(self) -> Expr<Ushort2> { self.as_::<Ushort2>() }
    pub fn as_byte2(self) -> Expr<Byte2> { self.as_::<Byte2>() }
    pub fn as_ubyte2(self) -> Expr<Ubyte2> { self.as_::<Ubyte2>() }
}
impl Expr<Ulong3> {
    pub fn as_float3(self) -> Expr<Float3> { self.as_::<Float3>() }
    pub fn as_int3(self) -> Expr<Int3> { self.as_::<Int3>() }
    pub fn as_uint3(self) -> Expr<Uint3> { self.as_::<Uint3>() }
    pub fn as_double3(self) -> Expr<Double3> { self.as_::<Double3>() }
    pub fn as_long3(self) -> Expr<Long3> { self.as_::<Long3>() }
    pub fn as_half3(self) -> Expr<Half3> { self.as_::<Half3>() }
    pub fn as_short3(self) -> Expr<Short3> { self.as_::<Short3>() }
    pub fn as_ushort3(self) -> Expr<Ushort3> { self.as_::<Ushort3>() }
    pub fn as_byte3(self) -> Expr<Byte3> { self.as_::<Byte3>() }
    pub fn as_ubyte3(self) -> Expr<Ubyte3> { self.as_::<Ubyte3>() }
}
impl Expr<Ulong4> {
    pub fn as_float4(self) -> Expr<Float4> { self.as_::<Float4>() }
    pub fn as_int4(self) -> Expr<Int4> { self.as_::<Int4>() }
    pub fn as_uint4(self) -> Expr<Uint4> { self.as_::<Uint4>() }
    pub fn as_double4(self) -> Expr<Double4> { self.as_::<Double4>() }
    pub fn as_long4(self) -> Expr<Long4> { self.as_::<Long4>() }
    pub fn as_half4(self) -> Expr<Half4> { self.as_::<Half4>() }
    pub fn as_short4(self) -> Expr<Short4> { self.as_::<Short4>() }
    pub fn as_ushort4(self) -> Expr<Ushort4> { self.as_::<Ushort4>() }
    pub fn as_byte4(self) -> Expr<Byte4> { self.as_::<Byte4>() }
    pub fn as_ubyte4(self) -> Expr<Ubyte4> { self.as_::<Ubyte4>() }
}
impl Expr<f16> {
    pub fn as_f32(self) -> Expr<f32> { self.as_::<f32>() }
    pub fn as_i32(self) -> Expr<i32> { self.as_::<i32>() }
    pub fn as_u32(self) -> Expr<u32> { self.as_::<u32>() }
    pub fn as_f64(self) -> Expr<f64> { self.as_::<f64>() }
    pub fn as_i64(self) -> Expr<i64> { self.as_::<i64>() }
    pub fn as_u64(self) -> Expr<u64> { self.as_::<u64>() }
    pub fn as_i16(self) -> Expr<i16> { self.as_::<i16>() }
    pub fn as_u16(self) -> Expr<u16> { self.as_::<u16>() }
    pub fn as_i8(self) -> Expr<i8> { self.as_::<i8>() }
    pub fn as_u8(self) -> Expr<u8> { self.as_::<u8>() }
}
impl Expr<Half2> {
    pub fn as_float2(self) -> Expr<Float2> { self.as_::<Float2>() }
    pub fn as_int2(self) -> Expr<Int2> { self.as_::<Int2>() }
    pub fn as_uint2(self) -> Expr<Uint2> { self.as_::<Uint2>() }
    pub fn as_double2(self) -> Expr<Double2> { self.as_::<Double2>() }
    pub fn as_long2(self) -> Expr<Long2> { self.as_::<Long2>() }
    pub fn as_ulong2(self) -> Expr<Ulong2> { self.as_::<Ulong2>() }
    pub fn as_short2(self) -> Expr<Short2> { self.as_::<Short2>() }
    pub fn as_ushort2(self) -> Expr<Ushort2> { self.as_::<Ushort2>() }
    pub fn as_byte2(self) -> Expr<Byte2> { self.as_::<Byte2>() }
    pub fn as_ubyte2(self) -> Expr<Ubyte2> { self.as_::<Ubyte2>() }
}
impl Expr<Half3> {
    pub fn as_float3(self) -> Expr<Float3> { self.as_::<Float3>() }
    pub fn as_int3(self) -> Expr<Int3> { self.as_::<Int3>() }
    pub fn as_uint3(self) -> Expr<Uint3> { self.as_::<Uint3>() }
    pub fn as_double3(self) -> Expr<Double3> { self.as_::<Double3>() }
    pub fn as_long3(self) -> Expr<Long3> { self.as_::<Long3>() }
    pub fn as_ulong3(self) -> Expr<Ulong3> { self.as_::<Ulong3>() }
    pub fn as_short3(self) -> Expr<Short3> { self.as_::<Short3>() }
    pub fn as_ushort3(self) -> Expr<Ushort3> { self.as_::<Ushort3>() }
    pub fn as_byte3(self) -> Expr<Byte3> { self.as_::<Byte3>() }
    pub fn as_ubyte3(self) -> Expr<Ubyte3> { self.as_::<Ubyte3>() }
}
impl Expr<Half4> {
    pub fn as_float4(self) -> Expr<Float4> { self.as_::<Float4>() }
    pub fn as_int4(self) -> Expr<Int4> { self.as_::<Int4>() }
    pub fn as_uint4(self) -> Expr<Uint4> { self.as_::<Uint4>() }
    pub fn as_double4(self) -> Expr<Double4> { self.as_::<Double4>() }
    pub fn as_long4(self) -> Expr<Long4> { self.as_::<Long4>() }
    pub fn as_ulong4(self) -> Expr<Ulong4> { self.as_::<Ulong4>() }
    pub fn as_short4(self) -> Expr<Short4> { self.as_::<Short4>() }
    pub fn as_ushort4(self) -> Expr<Ushort4> { self.as_::<Ushort4>() }
    pub fn as_byte4(self) -> Expr<Byte4> { self.as_::<Byte4>() }
    pub fn as_ubyte4(self) -> Expr<Ubyte4> { self.as_::<Ubyte4>() }
}
impl Expr<i16> {
    pub fn as_f32(self) -> Expr<f32> { self.as_::<f32>() }
    pub fn as_i32(self) -> Expr<i32> { self.as_::<i32>() }
    pub fn as_u32(self) -> Expr<u32> { self.as_::<u32>() }
    pub fn as_f64(self) -> Expr<f64> { self.as_::<f64>() }
    pub fn as_i64(self) -> Expr<i64> { self.as_::<i64>() }
    pub fn as_u64(self) -> Expr<u64> { self.as_::<u64>() }
    pub fn as_f16(self) -> Expr<f16> { self.as_::<f16>() }
    pub fn as_u16(self) -> Expr<u16> { self.as_::<u16>() }
    pub fn as_i8(self) -> Expr<i8> { self.as_::<i8>() }
    pub fn as_u8(self) -> Expr<u8> { self.as_::<u8>() }
}
impl Expr<Short2> {
    pub fn as_float2(self) -> Expr<Float2> { self.as_::<Float2>() }
    pub fn as_int2(self) -> Expr<Int2> { self.as_::<Int2>() }
    pub fn as_uint2(self) -> Expr<Uint2> { self.as_::<Uint2>() }
    pub fn as_double2(self) -> Expr<Double2> { self.as_::<Double2>() }
    pub fn as_long2(self) -> Expr<Long2> { self.as_::<Long2>() }
    pub fn as_ulong2(self) -> Expr<Ulong2> { self.as_::<Ulong2>() }
    pub fn as_half2(self) -> Expr<Half2> { self.as_::<Half2>() }
    pub fn as_ushort2(self) -> Expr<Ushort2> { self.as_::<Ushort2>() }
    pub fn as_byte2(self) -> Expr<Byte2> { self.as_::<Byte2>() }
    pub fn as_ubyte2(self) -> Expr<Ubyte2> { self.as_::<Ubyte2>() }
}
impl Expr<Short3> {
    pub fn as_float3(self) -> Expr<Float3> { self.as_::<Float3>() }
    pub fn as_int3(self) -> Expr<Int3> { self.as_::<Int3>() }
    pub fn as_uint3(self) -> Expr<Uint3> { self.as_::<Uint3>() }
    pub fn as_double3(self) -> Expr<Double3> { self.as_::<Double3>() }
    pub fn as_long3(self) -> Expr<Long3> { self.as_::<Long3>() }
    pub fn as_ulong3(self) -> Expr<Ulong3> { self.as_::<Ulong3>() }
    pub fn as_half3(self) -> Expr<Half3> { self.as_::<Half3>() }
    pub fn as_ushort3(self) -> Expr<Ushort3> { self.as_::<Ushort3>() }
    pub fn as_byte3(self) -> Expr<Byte3> { self.as_::<Byte3>() }
    pub fn as_ubyte3(self) -> Expr<Ubyte3> { self.as_::<Ubyte3>() }
}
impl Expr<Short4> {
    pub fn as_float4(self) -> Expr<Float4> { self.as_::<Float4>() }
    pub fn as_int4(self) -> Expr<Int4> { self.as_::<Int4>() }
    pub fn as_uint4(self) -> Expr<Uint4> { self.as_::<Uint4>() }
    pub fn as_double4(self) -> Expr<Double4> { self.as_::<Double4>() }
    pub fn as_long4(self) -> Expr<Long4> { self.as_::<Long4>() }
    pub fn as_ulong4(self) -> Expr<Ulong4> { self.as_::<Ulong4>() }
    pub fn as_half4(self) -> Expr<Half4> { self.as_::<Half4>() }
    pub fn as_ushort4(self) -> Expr<Ushort4> { self.as_::<Ushort4>() }
    pub fn as_byte4(self) -> Expr<Byte4> { self.as_::<Byte4>() }
    pub fn as_ubyte4(self) -> Expr<Ubyte4> { self.as_::<Ubyte4>() }
}
impl Expr<u16> {
    pub fn as_f32(self) -> Expr<f32> { self.as_::<f32>() }
    pub fn as_i32(self) -> Expr<i32> { self.as_::<i32>() }
    pub fn as_u32(self) -> Expr<u32> { self.as_::<u32>() }
    pub fn as_f64(self) -> Expr<f64> { self.as_::<f64>() }
    pub fn as_i64(self) -> Expr<i64> { self.as_::<i64>() }
    pub fn as_u64(self) -> Expr<u64> { self.as_::<u64>() }
    pub fn as_f16(self) -> Expr<f16> { self.as_::<f16>() }
    pub fn as_i16(self) -> Expr<i16> { self.as_::<i16>() }
    pub fn as_i8(self) -> Expr<i8> { self.as_::<i8>() }
    pub fn as_u8(self) -> Expr<u8> { self.as_::<u8>() }
}
impl Expr<Ushort2> {
    pub fn as_float2(self) -> Expr<Float2> { self.as_::<Float2>() }
    pub fn as_int2(self) -> Expr<Int2> { self.as_::<Int2>() }
    pub fn as_uint2(self) -> Expr<Uint2> { self.as_::<Uint2>() }
    pub fn as_double2(self) -> Expr<Double2> { self.as_::<Double2>() }
    pub fn as_long2(self) -> Expr<Long2> { self.as_::<Long2>() }
    pub fn as_ulong2(self) -> Expr<Ulong2> { self.as_::<Ulong2>() }
    pub fn as_half2(self) -> Expr<Half2> { self.as_::<Half2>() }
    pub fn as_short2(self) -> Expr<Short2> { self.as_::<Short2>() }
    pub fn as_byte2(self) -> Expr<Byte2> { self.as_::<Byte2>() }
    pub fn as_ubyte2(self) -> Expr<Ubyte2> { self.as_::<Ubyte2>() }
}
impl Expr<Ushort3> {
    pub fn as_float3(self) -> Expr<Float3> { self.as_::<Float3>() }
    pub fn as_int3(self) -> Expr<Int3> { self.as_::<Int3>() }
    pub fn as_uint3(self) -> Expr<Uint3> { self.as_::<Uint3>() }
    pub fn as_double3(self) -> Expr<Double3> { self.as_::<Double3>() }
    pub fn as_long3(self) -> Expr<Long3> { self.as_::<Long3>() }
    pub fn as_ulong3(self) -> Expr<Ulong3> { self.as_::<Ulong3>() }
    pub fn as_half3(self) -> Expr<Half3> { self.as_::<Half3>() }
    pub fn as_short3(self) -> Expr<Short3> { self.as_::<Short3>() }
    pub fn as_byte3(self) -> Expr<Byte3> { self.as_::<Byte3>() }
    pub fn as_ubyte3(self) -> Expr<Ubyte3> { self.as_::<Ubyte3>() }
}
impl Expr<Ushort4> {
    pub fn as_float4(self) -> Expr<Float4> { self.as_::<Float4>() }
    pub fn as_int4(self) -> Expr<Int4> { self.as_::<Int4>() }
    pub fn as_uint4(self) -> Expr<Uint4> { self.as_::<Uint4>() }
    pub fn as_double4(self) -> Expr<Double4> { self.as_::<Double4>() }
    pub fn as_long4(self) -> Expr<Long4> { self.as_::<Long4>() }
    pub fn as_ulong4(self) -> Expr<Ulong4> { self.as_::<Ulong4>() }
    pub fn as_half4(self) -> Expr<Half4> { self.as_::<Half4>() }
    pub fn as_short4(self) -> Expr<Short4> { self.as_::<Short4>() }
    pub fn as_byte4(self) -> Expr<Byte4> { self.as_::<Byte4>() }
    pub fn as_ubyte4(self) -> Expr<Ubyte4> { self.as_::<Ubyte4>() }
}
impl Expr<i8> {
    pub fn as_f32(self) -> Expr<f32> { self.as_::<f32>() }
    pub fn as_i32(self) -> Expr<i32> { self.as_::<i32>() }
    pub fn as_u32(self) -> Expr<u32> { self.as_::<u32>() }
    pub fn as_f64(self) -> Expr<f64> { self.as_::<f64>() }
    pub fn as_i64(self) -> Expr<i64> { self.as_::<i64>() }
    pub fn as_u64(self) -> Expr<u64> { self.as_::<u64>() }
    pub fn as_f16(self) -> Expr<f16> { self.as_::<f16>() }
    pub fn as_i16(self) -> Expr<i16> { self.as_::<i16>() }
    pub fn as_u16(self) -> Expr<u16> { self.as_::<u16>() }
    pub fn as_u8(self) -> Expr<u8> { self.as_::<u8>() }
}
impl Expr<Byte2> {
    pub fn as_float2(self) -> Expr<Float2> { self.as_::<Float2>() }
    pub fn as_int2(self) -> Expr<Int2> { self.as_::<Int2>() }
    pub fn as_uint2(self) -> Expr<Uint2> { self.as_::<Uint2>() }
    pub fn as_double2(self) -> Expr<Double2> { self.as_::<Double2>() }
    pub fn as_long2(self) -> Expr<Long2> { self.as_::<Long2>() }
    pub fn as_ulong2(self) -> Expr<Ulong2> { self.as_::<Ulong2>() }
    pub fn as_half2(self) -> Expr<Half2> { self.as_::<Half2>() }
    pub fn as_short2(self) -> Expr<Short2> { self.as_::<Short2>() }
    pub fn as_ushort2(self) -> Expr<Ushort2> { self.as_::<Ushort2>() }
    pub fn as_ubyte2(self) -> Expr<Ubyte2> { self.as_::<Ubyte2>() }
}
impl Expr<Byte3> {
    pub fn as_float3(self) -> Expr<Float3> { self.as_::<Float3>() }
    pub fn as_int3(self) -> Expr<Int3> { self.as_::<Int3>() }
    pub fn as_uint3(self) -> Expr<Uint3> { self.as_::<Uint3>() }
    pub fn as_double3(self) -> Expr<Double3> { self.as_::<Double3>() }
    pub fn as_long3(self) -> Expr<Long3> { self.as_::<Long3>() }
    pub fn as_ulong3(self) -> Expr<Ulong3> { self.as_::<Ulong3>() }
    pub fn as_half3(self) -> Expr<Half3> { self.as_::<Half3>() }
    pub fn as_short3(self) -> Expr<Short3> { self.as_::<Short3>() }
    pub fn as_ushort3(self) -> Expr<Ushort3> { self.as_::<Ushort3>() }
    pub fn as_ubyte3(self) -> Expr<Ubyte3> { self.as_::<Ubyte3>() }
}
impl Expr<Byte4> {
    pub fn as_float4(self) -> Expr<Float4> { self.as_::<Float4>() }
    pub fn as_int4(self) -> Expr<Int4> { self.as_::<Int4>() }
    pub fn as_uint4(self) -> Expr<Uint4> { self.as_::<Uint4>() }
    pub fn as_double4(self) -> Expr<Double4> { self.as_::<Double4>() }
    pub fn as_long4(self) -> Expr<Long4> { self.as_::<Long4>() }
    pub fn as_ulong4(self) -> Expr<Ulong4> { self.as_::<Ulong4>() }
    pub fn as_half4(self) -> Expr<Half4> { self.as_::<Half4>() }
    pub fn as_short4(self) -> Expr<Short4> { self.as_::<Short4>() }
    pub fn as_ushort4(self) -> Expr<Ushort4> { self.as_::<Ushort4>() }
    pub fn as_ubyte4(self) -> Expr<Ubyte4> { self.as_::<Ubyte4>() }
}
impl Expr<u8> {
    pub fn as_f32(self) -> Expr<f32> { self.as_::<f32>() }
    pub fn as_i32(self) -> Expr<i32> { self.as_::<i32>() }
    pub fn as_u32(self) -> Expr<u32> { self.as_::<u32>() }
    pub fn as_f64(self) -> Expr<f64> { self.as_::<f64>() }
    pub fn as_i64(self) -> Expr<i64> { self.as_::<i64>() }
    pub fn as_u64(self) -> Expr<u64> { self.as_::<u64>() }
    pub fn as_f16(self) -> Expr<f16> { self.as_::<f16>() }
    pub fn as_i16(self) -> Expr<i16> { self.as_::<i16>() }
    pub fn as_u16(self) -> Expr<u16> { self.as_::<u16>() }
    pub fn as_i8(self) -> Expr<i8> { self.as_::<i8>() }
}
impl Expr<Ubyte2> {
    pub fn as_float2(self) -> Expr<Float2> { self.as_::<Float2>() }
    pub fn as_int2(self) -> Expr<Int2> { self.as_::<Int2>() }
    pub fn as_uint2(self) -> Expr<Uint2> { self.as_::<Uint2>() }
    pub fn as_double2(self) -> Expr<Double2> { self.as_::<Double2>() }
    pub fn as_long2(self) -> Expr<Long2> { self.as_::<Long2>() }
    pub fn as_ulong2(self) -> Expr<Ulong2> { self.as_::<Ulong2>() }
    pub fn as_half2(self) -> Expr<Half2> { self.as_::<Half2>() }
    pub fn as_short2(self) -> Expr<Short2> { self.as_::<Short2>() }
    pub fn as_ushort2(self) -> Expr<Ushort2> { self.as_::<Ushort2>() }
    pub fn as_byte2(self) -> Expr<Byte2> { self.as_::<Byte2>() }
}
impl Expr<Ubyte3> {
    pub fn as_float3(self) -> Expr<Float3> { self.as_::<Float3>() }
    pub fn as_int3(self) -> Expr<Int3> { self.as_::<Int3>() }
    pub fn as_uint3(self) -> Expr<Uint3> { self.as_::<Uint3>() }
    pub fn as_double3(self) -> Expr<Double3> { self.as_::<Double3>() }
    pub fn as_long3(self) -> Expr<Long3> { self.as_::<Long3>() }
    pub fn as_ulong3(self) -> Expr<Ulong3> { self.as_::<Ulong3>() }
    pub fn as_half3(self) -> Expr<Half3> { self.as_::<Half3>() }
    pub fn as_short3(self) -> Expr<Short3> { self.as_::<Short3>() }
    pub fn as_ushort3(self) -> Expr<Ushort3> { self.as_::<Ushort3>() }
    pub fn as_byte3(self) -> Expr<Byte3> { self.as_::<Byte3>() }
}
impl Expr<Ubyte4> {
    pub fn as_float4(self) -> Expr<Float4> { self.as_::<Float4>() }
    pub fn as_int4(self) -> Expr<Int4> { self.as_::<Int4>() }
    pub fn as_uint4(self) -> Expr<Uint4> { self.as_::<Uint4>() }
    pub fn as_double4(self) -> Expr<Double4> { self.as_::<Double4>() }
    pub fn as_long4(self) -> Expr<Long4> { self.as_::<Long4>() }
    pub fn as_ulong4(self) -> Expr<Ulong4> { self.as_::<Ulong4>() }
    pub fn as_half4(self) -> Expr<Half4> { self.as_::<Half4>() }
    pub fn as_short4(self) -> Expr<Short4> { self.as_::<Short4>() }
    pub fn as_ushort4(self) -> Expr<Ushort4> { self.as_::<Ushort4>() }
    pub fn as_byte4(self) -> Expr<Byte4> { self.as_::<Byte4>() }
}
}
