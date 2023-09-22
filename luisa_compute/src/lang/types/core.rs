use super::*;

mod private {
    use super::*;
    pub trait Sealed {}
    impl Sealed for bool {}
    impl Sealed for f16 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
}

pub trait Primitive: private::Sealed + Copy + TypeOf + 'static {
    fn const_(&self) -> Const;
    fn primitive() -> ir::Primitive;
}
impl<T: Primitive> Value for T {
    type Expr = PrimitiveExpr<T>;
    type Var = PrimitiveVar<T>;
    type AtomicRef = PrimitiveAtomicRef<T>;

    fn expr(self) -> Expr<Self> {
        let node = __current_scope(|s| -> NodeRef { s.const_(self.const_()) });
        Expr::<Self>::from_node(node)
    }
}

impl_simple_expr_proxy!([T: Primitive] PrimitiveExpr[T] for T);
impl_simple_var_proxy!([T: Primitive] PrimitiveVar[T] for T);
impl_simple_atomic_ref_proxy!([T: Primitive] PrimitiveAtomicRef[T] for T);

macro_rules! impl_atomic {
    ($t:ty) => {
        impl AtomicRef<$t> {
            pub fn compare_exchange(
                &self,
                expected: impl AsExpr<Value = $t>,
                desired: impl AsExpr<Value = $t>,
            ) -> Expr<$t> {
                lower_atomic_ref(
                    self.node(),
                    Func::AtomicCompareExchange,
                    &[expected.as_expr().node(), desired.as_expr().node()],
                )
            }
            pub fn exchange(&self, operand: impl AsExpr<Value = $t>) -> Expr<$t> {
                lower_atomic_ref(
                    self.node(),
                    Func::AtomicExchange,
                    &[operand.as_expr().node()],
                )
            }
            pub fn fetch_add(&self, operand: impl AsExpr<Value = $t>) -> Expr<$t> {
                lower_atomic_ref(
                    self.node(),
                    Func::AtomicFetchAdd,
                    &[operand.as_expr().node()],
                )
            }
            pub fn fetch_sub(&self, operand: impl AsExpr<Value = $t>) -> Expr<$t> {
                lower_atomic_ref(
                    self.node(),
                    Func::AtomicFetchSub,
                    &[operand.as_expr().node()],
                )
            }
            pub fn fetch_min(&self, operand: impl AsExpr<Value = $t>) -> Expr<$t> {
                lower_atomic_ref(
                    self.node(),
                    Func::AtomicFetchMin,
                    &[operand.as_expr().node()],
                )
            }
            pub fn fetch_max(&self, operand: impl AsExpr<Value = $t>) -> Expr<$t> {
                lower_atomic_ref(
                    self.node(),
                    Func::AtomicFetchMax,
                    &[operand.as_expr().node()],
                )
            }
        }
    };
}
macro_rules! impl_atomic_bit {
    ($t:ty) => {
        impl AtomicRef<$t> {
            pub fn fetch_and(&self, operand: impl AsExpr<Value = $t>) -> Expr<$t> {
                lower_atomic_ref(
                    self.node(),
                    Func::AtomicFetchAnd,
                    &[operand.as_expr().node()],
                )
            }
            pub fn fetch_or(&self, operand: impl AsExpr<Value = $t>) -> Expr<$t> {
                lower_atomic_ref(
                    self.node(),
                    Func::AtomicFetchOr,
                    &[operand.as_expr().node()],
                )
            }
            pub fn fetch_xor(&self, operand: impl AsExpr<Value = $t>) -> Expr<$t> {
                lower_atomic_ref(
                    self.node(),
                    Func::AtomicFetchXor,
                    &[operand.as_expr().node()],
                )
            }
        }
    };
}
impl_atomic!(i32);
impl_atomic!(u32);
impl_atomic!(i64);
impl_atomic!(u64);
impl_atomic!(f32);
impl_atomic_bit!(u32);
impl_atomic_bit!(u64);
impl_atomic_bit!(i32);
impl_atomic_bit!(i64);
fn lower_atomic_ref<T: Value>(node: NodeRef, op: Func, args: &[NodeRef]) -> Expr<T> {
    let inst = node.get().instruction.as_ref();
    match inst {
        Instruction::Call(f, buffer_and_indices) => match f {
            Func::AtomicRef => {
                let new_args = buffer_and_indices
                    .iter()
                    .chain(args.iter())
                    .map(|n| *n)
                    .collect::<Vec<_>>();
                Expr::<T>::from_node(__current_scope(|b| {
                    b.call(op, &new_args, <T as TypeOf>::type_())
                }))
            }
            _ => unreachable!("{:?}", inst),
        },
        _ => unreachable!(),
    }
}

impl Primitive for bool {
    fn const_(&self) -> Const {
        Const::Bool(*self)
    }
    fn primitive() -> ir::Primitive {
        ir::Primitive::Bool
    }
}

impl Primitive for f16 {
    fn const_(&self) -> Const {
        Const::Float16(*self)
    }
    fn primitive() -> ir::Primitive {
        ir::Primitive::Float16
    }
}
impl Primitive for f32 {
    fn const_(&self) -> Const {
        Const::Float32(*self)
    }
    fn primitive() -> ir::Primitive {
        ir::Primitive::Float32
    }
}
impl Primitive for f64 {
    fn const_(&self) -> Const {
        Const::Float64(*self)
    }
    fn primitive() -> ir::Primitive {
        ir::Primitive::Float64
    }
}

impl Primitive for i8 {
    fn const_(&self) -> Const {
        Const::Int8(*self)
    }
    fn primitive() -> ir::Primitive {
        ir::Primitive::Int8
    }
}
impl Primitive for i16 {
    fn const_(&self) -> Const {
        Const::Int16(*self)
    }
    fn primitive() -> ir::Primitive {
        ir::Primitive::Int16
    }
}
impl Primitive for i32 {
    fn const_(&self) -> Const {
        Const::Int32(*self)
    }
    fn primitive() -> ir::Primitive {
        ir::Primitive::Int32
    }
}
impl Primitive for i64 {
    fn const_(&self) -> Const {
        Const::Int64(*self)
    }
    fn primitive() -> ir::Primitive {
        ir::Primitive::Int64
    }
}

impl Primitive for u8 {
    fn const_(&self) -> Const {
        Const::Uint8(*self)
    }
    fn primitive() -> ir::Primitive {
        ir::Primitive::Uint8
    }
}
impl Primitive for u16 {
    fn const_(&self) -> Const {
        Const::Uint16(*self)
    }
    fn primitive() -> ir::Primitive {
        ir::Primitive::Uint16
    }
}
impl Primitive for u32 {
    fn const_(&self) -> Const {
        Const::Uint32(*self)
    }
    fn primitive() -> ir::Primitive {
        ir::Primitive::Uint32
    }
}
impl Primitive for u64 {
    fn const_(&self) -> Const {
        Const::Uint64(*self)
    }
    fn primitive() -> ir::Primitive {
        ir::Primitive::Uint64
    }
}

macro_rules! impls {
    ($T:ident for $($t:ty),*) => {
        $(impl $T for $t {})*
    };
}

pub trait Integral: Primitive {}
impls!(Integral for bool, i8, i16, i32, i64, u8, u16, u32, u64);

pub trait Numeric: Primitive {}
impls!(Numeric for f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);

pub trait Floating: Numeric {}
impls!(Floating for f16, f32, f64);

pub trait Signed: Numeric {}
impls!(Signed for f16, f32, f64, i8, i16, i32, i64);
