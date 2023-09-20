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
    fn primitive(&self) -> ir::Primitive;
}
impl<T: Primitive> Value for T {
    type Expr = PrimitiveExpr<T>;
    type Var = PrimitiveVar<T>;
    type ExprData = ();
    type VarData = ();

    fn expr(&self) -> Expr<Self> {
        let node = __current_scope(|s| -> NodeRef { s.const_(self.const_()) });
        Expr::<Self>::from_node(node)
    }
}

impl_simple_expr_proxy!([T: Primitive] PrimitiveExpr[T] for T);
impl_simple_var_proxy!([T: Primitive] PrimitiveVar[T] for T);

impl Primitive for bool {
    fn const_(&self) -> Const {
        Const::Bool(*self)
    }
    fn primitive(&self) -> ir::Primitive {
        ir::Primitive::Bool
    }
}

impl Primitive for f16 {
    fn const_(&self) -> Const {
        Const::F16(*self)
    }
    fn primitive(&self) -> ir::Primitive {
        ir::Primitive::F16
    }
}
impl Primitive for f32 {
    fn const_(&self) -> Const {
        Const::F32(*self)
    }
    fn primitive(&self) -> ir::Primitive {
        ir::Primitive::F32
    }
}
impl Primitive for f64 {
    fn const_(&self) -> Const {
        Const::F64(*self)
    }
    fn primitive(&self) -> ir::Primitive {
        ir::Primitive::F64
    }
}

impl Primitive for i8 {
    fn const_(&self) -> Const {
        todo!() // Const::I8(*self)
    }
}
impl Primitive for i16 {
    fn const_(&self) -> Const {
        Const::I16(*self)
    }
    fn primitive(&self) -> ir::Primitive {
        ir::Primitive::Int16
    }
}
impl Primitive for i32 {
    fn const_(&self) -> Const {
        Const::I32(*self)
    }
    fn primitive(&self) -> ir::Primitive {
        ir::Primitive::Int32
    }
}
impl Primitive for i64 {
    fn const_(&self) -> Const {
        Const::I64(*self)
    }
    fn primitive(&self) -> ir::Primitive {
        ir::Primitive::Int64
    }
}

impl Primitive for u8 {
    fn const_(&self) -> Const {
        todo!() // Const::U8(*self)
    }
}
impl Primitive for u16 {
    fn const_(&self) -> Const {
        Const::U16(*self)
    }
    fn primitive(&self) -> ir::Primitive {
        ir::Primitive::UInt16
    }
}
impl Primitive for u32 {
    fn const_(&self) -> Const {
        Const::U32(*self)
    }
    fn primitive(&self) -> ir::Primitive {
        ir::Primitive::UInt32
    }
}
impl Primitive for u64 {
    fn const_(&self) -> Const {
        Const::U64(*self)
    }
    fn primitive(&self) -> ir::Primitive {
        ir::Primitive::UInt64
    }
}

pub trait Integral: Primitive {}
impl Integral for bool {}
impl Integral for i8 {}
impl Integral for i16 {}
impl Integral for i32 {}
impl Integral for i64 {}
impl Integral for u8 {}
impl Integral for u16 {}
impl Integral for u32 {}
impl Integral for u64 {}

pub trait Numeric: Primitive {}
impl Numeric for f16 {}
impl Numeric for f32 {}
impl Numeric for f64 {}
impl Numeric for i8 {}
impl Numeric for i16 {}
impl Numeric for i32 {}
impl Numeric for i64 {}
impl Numeric for u8 {}
impl Numeric for u16 {}
impl Numeric for u32 {}
impl Numeric for u64 {}

pub trait Floating: Numeric {}
impl Floating for f16 {}
impl Floating for f32 {}
impl Floating for f64 {}

pub trait Signed: Numeric {}
impl Signed for f16 {}
impl Signed for f32 {}
impl Signed for f64 {}
impl Signed for i8 {}
impl Signed for i16 {}
impl Signed for i32 {}
impl Signed for i64 {}

#[deprecated]
pub type Bool = Expr<bool>;
#[deprecated]
pub type F16 = Expr<f16>;
#[deprecated]
pub type F32 = Expr<f32>;
#[deprecated]
pub type F64 = Expr<f64>;
#[deprecated]
pub type I16 = Expr<i16>;
#[deprecated]
pub type I32 = Expr<i32>;
#[deprecated]
pub type I64 = Expr<i64>;
#[deprecated]
pub type U16 = Expr<u16>;
#[deprecated]
pub type U32 = Expr<u32>;
#[deprecated]
pub type U64 = Expr<u64>;

#[deprecated]
pub type F16Var = Var<f16>;
#[deprecated]
pub type F32Var = Var<f32>;
#[deprecated]
pub type F64Var = Var<f64>;
#[deprecated]
pub type I16Var = Var<i16>;
#[deprecated]
pub type I32Var = Var<i32>;
#[deprecated]
pub type I64Var = Var<i64>;
#[deprecated]
pub type U16Var = Var<u16>;
#[deprecated]
pub type U32Var = Var<u32>;
#[deprecated]
pub type U64Var = Var<u64>;

#[deprecated]
pub type Half = Expr<f16>;
#[deprecated]
pub type Float = Expr<f32>;
#[deprecated]
pub type Double = Expr<f64>;
#[deprecated]
pub type Int = Expr<i32>;
#[deprecated]
pub type Long = Expr<i64>;
#[deprecated]
pub type Uint = Expr<u32>;
#[deprecated]
pub type Ulong = Expr<u64>;
#[deprecated]
pub type Short = Expr<i16>;
#[deprecated]
pub type Ushort = Expr<u16>;

#[deprecated]
pub type BoolVar = Var<bool>;
#[deprecated]
pub type HalfVar = Var<f16>;
#[deprecated]
pub type FloatVar = Var<f32>;
#[deprecated]
pub type DoubleVar = Var<f64>;
#[deprecated]
pub type IntVar = Var<i32>;
#[deprecated]
pub type LongVar = Var<i64>;
#[deprecated]
pub type UintVar = Var<u32>;
#[deprecated]
pub type UlongVar = Var<u64>;
#[deprecated]
pub type ShortVar = Var<i16>;
#[deprecated]
pub type UshortVar = Var<u16>;
