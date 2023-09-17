use crate::internal_prelude::*;

pub fn thread_id() -> Expr<Uint3> {
    Expr::<Uint3>::from_node(__current_scope(|b| {
        b.call(Func::ThreadId, &[], Uint3::type_())
    }))
}

pub fn block_id() -> Expr<Uint3> {
    Expr::<Uint3>::from_node(__current_scope(|b| {
        b.call(Func::BlockId, &[], Uint3::type_())
    }))
}

pub fn dispatch_id() -> Expr<Uint3> {
    Expr::<Uint3>::from_node(__current_scope(|b| {
        b.call(Func::DispatchId, &[], Uint3::type_())
    }))
}

pub fn dispatch_size() -> Expr<Uint3> {
    Expr::<Uint3>::from_node(__current_scope(|b| {
        b.call(Func::DispatchSize, &[], Uint3::type_())
    }))
}

fn check_block_size_for_cpu() {
    RECORDER.with(|r| {
        let r = r.borrow();
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
    Expr::<bool>::from_node(__current_scope(|b| {
        b.call(Func::WarpIsFirstActiveLane, &[], Expr::<bool>::type_())
    }))
}
pub fn warp_active_all_equal(v: impl ScalarOrVector) -> Expr<bool> {
    Expr::<bool>::from_node(__current_scope(|b| {
        b.call(
            Func::WarpActiveAllEqual,
            &[v.node()],
            <bool as TypeOf>::type_(),
        )
    }))
}
pub fn warp_active_bit_and<T: ScalarOrVector<Element = E>, E: IntVarTrait>(v: T) -> T {
    T::from_node(__current_scope(|b| {
        b.call(
            Func::WarpActiveBitAnd,
            &[v.node()],
            <bool as TypeOf>::type_(),
        )
    }))
}

pub fn warp_active_bit_or<T: ScalarOrVector<Element = E>, E: IntVarTrait>(v: T) -> T {
    T::from_node(__current_scope(|b| {
        b.call(
            Func::WarpActiveBitOr,
            &[v.node()],
            <bool as TypeOf>::type_(),
        )
    }))
}

pub fn warp_active_bit_xor<T: ScalarOrVector<Element = E>, E: IntVarTrait>(v: T) -> T {
    T::from_node(__current_scope(|b| {
        b.call(
            Func::WarpActiveBitXor,
            &[v.node()],
            <bool as TypeOf>::type_(),
        )
    }))
}

pub fn warp_active_count_bits(v: impl Into<Expr<bool>>) -> Expr<u32> {
    Expr::<u32>::from_node(__current_scope(|b| {
        b.call(
            Func::WarpActiveCountBits,
            &[v.into().node()],
            <u32 as TypeOf>::type_(),
        )
    }))
}
pub fn warp_active_max<T: ScalarOrVector>(v: T) -> T::Element {
    <T::Element>::from_node(__current_scope(|b| {
        b.call(Func::WarpActiveMax, &[v.node()], <T::ElementHost>::type_())
    }))
}
pub fn warp_active_min<T: ScalarOrVector>(v: T) -> T::Element {
    <T::Element>::from_node(__current_scope(|b| {
        b.call(Func::WarpActiveMin, &[v.node()], <T::ElementHost>::type_())
    }))
}
pub fn warp_active_product<T: ScalarOrVector>(v: T) -> T::Element {
    <T::Element>::from_node(__current_scope(|b| {
        b.call(
            Func::WarpActiveProduct,
            &[v.node()],
            <T::ElementHost>::type_(),
        )
    }))
}
pub fn warp_active_sum<T: ScalarOrVector>(v: T) -> T::Element {
    <T::Element>::from_node(__current_scope(|b| {
        b.call(Func::WarpActiveSum, &[v.node()], <T::ElementHost>::type_())
    }))
}
pub fn warp_active_all(v: Expr<bool>) -> Expr<bool> {
    Expr::<bool>::from_node(__current_scope(|b| {
        b.call(Func::WarpActiveAll, &[v.node()], <bool as TypeOf>::type_())
    }))
}
pub fn warp_active_any(v: Expr<bool>) -> Expr<bool> {
    Expr::<bool>::from_node(__current_scope(|b| {
        b.call(Func::WarpActiveAny, &[v.node()], <bool as TypeOf>::type_())
    }))
}
pub fn warp_active_bit_mask() -> Expr<Uint4> {
    Expr::<Uint4>::from_node(__current_scope(|b| {
        b.call(Func::WarpActiveBitMask, &[], <Uint4 as TypeOf>::type_())
    }))
}
pub fn warp_prefix_count_bits(v: Expr<bool>) -> Expr<u32> {
    Expr::<u32>::from_node(__current_scope(|b| {
        b.call(
            Func::WarpPrefixCountBits,
            &[v.node()],
            <u32 as TypeOf>::type_(),
        )
    }))
}
pub fn warp_prefix_sum_exclusive<T: ScalarOrVector>(v: T) -> T {
    T::from_node(__current_scope(|b| {
        b.call(Func::WarpPrefixSum, &[v.node()], v.node().type_().clone())
    }))
}
pub fn warp_prefix_product_exclusive<T: ScalarOrVector>(v: T) -> T {
    T::from_node(__current_scope(|b| {
        b.call(
            Func::WarpPrefixProduct,
            &[v.node()],
            v.node().type_().clone(),
        )
    }))
}
pub fn warp_read_lane_at<T: BuiltinVarTrait>(v: T, index: impl Into<Expr<u32>>) -> T {
    let index = index.into();
    T::from_node(__current_scope(|b| {
        b.call(
            Func::WarpReadLaneAt,
            &[v.node(), index.node()],
            v.node().type_().clone(),
        )
    }))
}
pub fn warp_read_first_active_lane<T: BuiltinVarTrait>(v: T) -> T {
    T::from_node(__current_scope(|b| {
        b.call(
            Func::WarpReadFirstLane,
            &[v.node()],
            v.node().type_().clone(),
        )
    }))
}
pub fn set_block_size(size: [u32; 3]) {
    RECORDER.with(|r| {
        let mut r = r.borrow_mut();
        assert!(
            r.building_kernel,
            "set_block_size cannot be called in callable!"
        );
        assert!(r.block_size.is_none(), "Block size already set");

        r.block_size = Some(size);
    });
}

pub fn block_size() -> Expr<Uint3> {
    RECORDER.with(|r| {
        let r = r.borrow();
        let s = r.block_size.unwrap_or_else(|| panic!("Block size not set"));
        Uint3::new(s[0], s[1], s[2]).expr()
    })
}

pub unsafe fn bitcast<From: Value, To: Value>(expr: Expr<From>) -> Expr<To> {
    assert_eq!(std::mem::size_of::<From>(), std::mem::size_of::<To>());
    Expr::<To>::from_node(__current_scope(|b| {
        b.call(Func::Bitcast, &[expr.node()], <To as TypeOf>::type_())
    }))
}
