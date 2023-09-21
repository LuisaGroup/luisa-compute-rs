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
    type ExprData = ();
    type VarData = ();

    fn expr(self) -> Expr<Self> {
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

mod legacy {
    use super::*;

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
}
