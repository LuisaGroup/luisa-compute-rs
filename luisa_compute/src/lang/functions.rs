use crate::{internal_prelude::*, lang::with_recorder};

use super::types::core::{Integral, Numeric};

pub fn thread_id() -> Expr<Uint3> {
    Expr::<Uint3>::from_node(
        __current_scope(|b| b.call(Func::ThreadId, &[], Uint3::type_())).into(),
    )
}

pub fn block_id() -> Expr<Uint3> {
    Expr::<Uint3>::from_node(__current_scope(|b| b.call(Func::BlockId, &[], Uint3::type_())).into())
}

pub fn dispatch_id() -> Expr<Uint3> {
    Expr::<Uint3>::from_node(
        __current_scope(|b| b.call(Func::DispatchId, &[], Uint3::type_())).into(),
    )
}

pub fn dispatch_size() -> Expr<Uint3> {
    Expr::<Uint3>::from_node(
        __current_scope(|b| b.call(Func::DispatchSize, &[], Uint3::type_())).into(),
    )
}

fn check_block_size_for_cpu() {
    with_recorder(|r| {
        assert!(
            r.block_size.is_some(),
            "CPU backend only support block operations on block size 1"
        );
        let size = r.block_size.unwrap();
        assert_eq!(
            size,
            [1, 1, 1],
            "CPU backend only support block operations on block size 1"
        );
    });
}
pub fn sync_block() {
    if is_cpu_backend() {
        check_block_size_for_cpu();
        return;
    }
    __current_scope(|b| {
        b.call(Func::SynchronizeBlock, &[], Type::void());
    })
}

pub fn warp_is_first_active_lane() -> Expr<bool> {
    Expr::<bool>::from_node(
        __current_scope(|b| b.call(Func::WarpIsFirstActiveLane, &[], bool::type_())).into(),
    )
}
pub fn warp_active_all_equal(v: Expr<impl Linear>) -> Expr<bool> {
    let v = v.node().get();
    Expr::<bool>::from_node(
        __current_scope(|b| b.call(Func::WarpActiveAllEqual, &[v], <bool as TypeOf>::type_()))
            .into(),
    )
}
pub fn warp_active_bit_and<T: Linear>(v: Expr<T>) -> Expr<T>
where
    T::Scalar: Integral + Numeric,
{
    let v = v.node().get();
    Expr::<T>::from_node(
        __current_scope(|b| b.call(Func::WarpActiveBitAnd, &[v], <bool as TypeOf>::type_())).into(),
    )
}

pub fn warp_active_bit_or<T: Linear>(v: Expr<T>) -> Expr<T>
where
    T::Scalar: Integral + Numeric,
{
    let v = v.node().get();
    Expr::<T>::from_node(
        __current_scope(|b| b.call(Func::WarpActiveBitOr, &[v], <bool as TypeOf>::type_())).into(),
    )
}

pub fn warp_active_bit_xor<T: Linear>(v: Expr<T>) -> Expr<T>
where
    T::Scalar: Integral + Numeric,
{
    let v = v.node().get();
    Expr::<T>::from_node(
        __current_scope(|b| b.call(Func::WarpActiveBitXor, &[v], <bool as TypeOf>::type_())).into(),
    )
}

pub fn warp_active_count_bits(v: impl AsExpr<Value = bool>) -> Expr<u32> {
    let v = v.as_expr().node().get();
    Expr::<u32>::from_node(
        __current_scope(|b| b.call(Func::WarpActiveCountBits, &[v], <u32 as TypeOf>::type_()))
            .into(),
    )
}
pub fn warp_active_max<T: Linear>(v: Expr<T>) -> Expr<T::Scalar> {
    let v = v.node().get();
    Expr::<T::Scalar>::from_node(
        __current_scope(|b| b.call(Func::WarpActiveMax, &[v], <T::Scalar>::type_())).into(),
    )
}
pub fn warp_active_min<T: Linear>(v: Expr<T>) -> Expr<T::Scalar> {
    let v = v.node().get();
    Expr::<T::Scalar>::from_node(
        __current_scope(|b| b.call(Func::WarpActiveMin, &[v], <T::Scalar>::type_())).into(),
    )
}
pub fn warp_active_product<T: Linear>(v: Expr<T>) -> Expr<T::Scalar> {
    let v = v.node().get();
    Expr::<T::Scalar>::from_node(
        __current_scope(|b| b.call(Func::WarpActiveProduct, &[v], <T::Scalar>::type_())).into(),
    )
}
pub fn warp_active_sum<T: Linear>(v: Expr<T>) -> Expr<T::Scalar> {
    let v = v.node().get();
    Expr::<T::Scalar>::from_node(
        __current_scope(|b| b.call(Func::WarpActiveSum, &[v], <T::Scalar>::type_())).into(),
    )
}
pub fn warp_active_all(v: Expr<bool>) -> Expr<bool> {
    let v = v.node().get();
    Expr::<bool>::from_node(
        __current_scope(|b| b.call(Func::WarpActiveAll, &[v], <bool as TypeOf>::type_())).into(),
    )
}
pub fn warp_active_any(v: Expr<bool>) -> Expr<bool> {
    let v = v.node().get();
    Expr::<bool>::from_node(
        __current_scope(|b| b.call(Func::WarpActiveAny, &[v], <bool as TypeOf>::type_())).into(),
    )
}
pub fn warp_active_bit_mask() -> Expr<Uint4> {
    Expr::<Uint4>::from_node(
        __current_scope(|b| b.call(Func::WarpActiveBitMask, &[], <Uint4 as TypeOf>::type_()))
            .into(),
    )
}
pub fn warp_prefix_count_bits(v: Expr<bool>) -> Expr<u32> {
    let v = v.node().get();
    Expr::<u32>::from_node(
        __current_scope(|b| b.call(Func::WarpPrefixCountBits, &[v], <u32 as TypeOf>::type_()))
            .into(),
    )
}
pub fn warp_prefix_sum_exclusive<T: Linear>(v: Expr<T>) -> Expr<T> {
    let v = v.node().get();
    Expr::<T>::from_node(
        __current_scope(|b| b.call(Func::WarpPrefixSum, &[v], v.type_().clone())).into(),
    )
}
pub fn warp_prefix_product_exclusive<T: Linear>(v: Expr<T>) -> Expr<T> {
    let v = v.node().get();
    Expr::<T>::from_node(
        __current_scope(|b| b.call(Func::WarpPrefixProduct, &[v], v.type_().clone())).into(),
    )
}
// TODO: Difference between `Linear` and BuiltinVarTrait?
pub fn warp_read_lane_at<T: Linear>(v: Expr<T>, index: impl AsExpr<Value = u32>) -> Expr<T> {
    let index = index.as_expr().node().get();
    let v = v.node().get();
    Expr::<T>::from_node(
        __current_scope(|b| b.call(Func::WarpReadLaneAt, &[v, index], v.type_().clone())).into(),
    )
}
pub fn warp_read_first_active_lane<T: Linear>(v: Expr<T>) -> Expr<T> {
    let v = v.node().get();
    Expr::<T>::from_node(
        __current_scope(|b| b.call(Func::WarpReadFirstLane, &[v], v.type_().clone())).into(),
    )
}
pub fn set_block_size(size: [u32; 3]) {
    with_recorder(|r| {
        assert!(
            r.building_kernel,
            "set_block_size cannot be called in callable!"
        );
        assert!(r.block_size.is_none(), "Block size already set");

        r.block_size = Some(size);
    });
}

pub fn block_size() -> Expr<Uint3> {
    with_recorder(|r| {
        let s = r.block_size.unwrap_or_else(|| panic!("Block size not set"));
        Uint3::new(s[0], s[1], s[2]).expr()
    })
}

pub unsafe fn bitcast<From: Value, To: Value>(expr: Expr<From>) -> Expr<To> {
    assert_eq!(std::mem::size_of::<From>(), std::mem::size_of::<To>());
    let expr = expr.node().get();
    Expr::<To>::from_node(
        __current_scope(|b| b.call(Func::Bitcast, &[expr], <To as TypeOf>::type_())).into(),
    )
}
