//! The purpose of this module is to provide traits to represent things that may
//! either be an expression or a normal value. This is necessary for making the
//! trace macro work for both types of value.

use super::{vec::*, Aggregate, Bool, Expr, Var};
use crate::{VarCmp, VarCmpEq};
use half::f16;

pub trait BoolIfElseMaybeExpr<R> {
    fn if_then_else(self, then: impl FnOnce() -> R, else_: impl FnOnce() -> R) -> R;
}
impl<R> BoolIfElseMaybeExpr<R> for bool {
    fn if_then_else(self, then: impl FnOnce() -> R, else_: impl FnOnce() -> R) -> R {
        if self {
            then()
        } else {
            else_()
        }
    }
}
impl<R: Aggregate> BoolIfElseMaybeExpr<R> for Bool {
    fn if_then_else(self, then: impl FnOnce() -> R, else_: impl FnOnce() -> R) -> R {
        super::if_then_else(self, then, else_)
    }
}

pub trait BoolIfMaybeExpr {
    fn if_then(self, then: impl FnOnce());
}
impl BoolIfMaybeExpr for bool {
    fn if_then(self, then: impl FnOnce()) {
        if self {
            then()
        }
    }
}
impl BoolIfMaybeExpr for Bool {
    fn if_then(self, then: impl FnOnce()) {
        super::if_then_else(self, then, || {})
    }
}

pub trait BoolWhileMaybeExpr {
    fn while_loop(this: impl FnMut() -> Self, body: impl FnMut());
}
impl BoolWhileMaybeExpr for bool {
    fn while_loop(mut this: impl FnMut() -> Self, mut body: impl FnMut()) {
        while this() {
            body()
        }
    }
}
impl BoolWhileMaybeExpr for Bool {
    fn while_loop(this: impl FnMut() -> Self, body: impl FnMut()) {
        super::generic_loop(this, body, || {});
    }
}

// TODO: Support lazy expressions if that isn't done already?
pub trait BoolLazyOpsMaybeExpr<R> {
    type Ret;
    fn and(self, other: impl FnOnce() -> R) -> Self::Ret;
    fn or(self, other: impl FnOnce() -> R) -> Self::Ret;
}
impl BoolLazyOpsMaybeExpr<bool> for bool {
    type Ret = bool;
    fn and(self, other: impl FnOnce() -> bool) -> Self::Ret {
        self && other()
    }
    fn or(self, other: impl FnOnce() -> bool) -> Self::Ret {
        self || other()
    }
}
impl BoolLazyOpsMaybeExpr<Bool> for bool {
    type Ret = Bool;
    fn and(self, other: impl FnOnce() -> Bool) -> Self::Ret {
        self & other()
    }
    fn or(self, other: impl FnOnce() -> Bool) -> Self::Ret {
        self | other()
    }
}
impl BoolLazyOpsMaybeExpr<bool> for Bool {
    type Ret = Bool;
    fn and(self, other: impl FnOnce() -> bool) -> Self::Ret {
        self & other()
    }
    fn or(self, other: impl FnOnce() -> bool) -> Self::Ret {
        self | other()
    }
}
impl BoolLazyOpsMaybeExpr<Bool> for Bool {
    type Ret = Bool;
    fn and(self, other: impl FnOnce() -> Bool) -> Self::Ret {
        self & other()
    }
    fn or(self, other: impl FnOnce() -> Bool) -> Self::Ret {
        self | other()
    }
}

pub trait EqMaybeExpr<X> {
    type Bool;
    fn eq(self, other: X) -> Self::Bool;
    fn ne(self, other: X) -> Self::Bool;
}
impl<R, A: PartialEq<R>> EqMaybeExpr<R> for A {
    type Bool = bool;
    fn eq(self, other: R) -> Self::Bool {
        self == other
    }
    fn ne(self, other: R) -> Self::Bool {
        self != other
    }
}
macro_rules! impl_eme {
    ($t: ty, $s: ty) => {
        impl EqMaybeExpr<$s> for $t {
            type Bool = <$t as $crate::VarTrait>::Bool;
            fn eq(self, other: $s) -> Self::Bool {
                self.cmpeq(other)
            }
            fn ne(self, other: $s) -> Self::Bool {
                self.cmpne(other)
            }
        }
    };
}
macro_rules! impl_mem {
    ($t: ty, $s: ty) => {
        impl EqMaybeExpr<$s> for $t {
            type Bool = <$s as $crate::VarTrait>::Bool;
            fn eq(self, other: $s) -> Self::Bool {
                other.cmpeq(self)
            }
            fn ne(self, other: $s) -> Self::Bool {
                other.cmpne(self)
            }
        }
    };
}
macro_rules! emes {
    ($x: ty $(, $y: ty)*) => {
        impl_eme!(Expr<$x>, Expr<$x>);
        impl_eme!(Expr<$x>, $x);
        impl_mem!($x, Expr<$x>);
        $(impl_eme!(Expr<$x>, $y);
        impl_mem!($y, Expr<$x>);)*
    };
}
emes!(bool);
emes!(Bool2);
emes!(Bool3);
emes!(Bool4);

pub trait PartialOrdMaybeExpr<R> {
    type Bool;
    fn lt(self, other: R) -> Self::Bool;
    fn le(self, other: R) -> Self::Bool;
    fn ge(self, other: R) -> Self::Bool;
    fn gt(self, other: R) -> Self::Bool;
}
impl<R, A: PartialOrd<R>> PartialOrdMaybeExpr<R> for A {
    type Bool = bool;
    fn lt(self, other: R) -> Self::Bool {
        self < other
    }
    fn le(self, other: R) -> Self::Bool {
        self <= other
    }
    fn ge(self, other: R) -> Self::Bool {
        self >= other
    }
    fn gt(self, other: R) -> Self::Bool {
        self > other
    }
}
macro_rules! impl_pome {
    ($t: ty, $s: ty) => {
        impl_eme!($t, $s);
        impl PartialOrdMaybeExpr<$s> for $t {
            type Bool = <$t as $crate::VarTrait>::Bool;
            fn lt(self, other: $s) -> Self::Bool {
                self.cmplt(other)
            }
            fn le(self, other: $s) -> Self::Bool {
                self.cmple(other)
            }
            fn ge(self, other: $s) -> Self::Bool {
                self.cmpge(other)
            }
            fn gt(self, other: $s) -> Self::Bool {
                self.cmpgt(other)
            }
        }
    };
}
macro_rules! impl_emop {
    ($t: ty, $s: ty) => {
        impl_mem!($t, $s);
        impl PartialOrdMaybeExpr<$s> for $t {
            type Bool = <$s as $crate::VarTrait>::Bool;
            fn lt(self, other: $s) -> Self::Bool {
                other.cmpgt(self)
            }
            fn le(self, other: $s) -> Self::Bool {
                other.cmpge(self)
            }
            fn ge(self, other: $s) -> Self::Bool {
                other.cmplt(self)
            }
            fn gt(self, other: $s) -> Self::Bool {
                other.cmplt(self)
            }
        }
    };
}
macro_rules! pomes {
    ($x: ty $(, $y:ty)*) => {
        impl_pome!(Expr<$x>, Expr<$x>);
        impl_pome!(Expr<$x>, $x);
        impl_emop!($x, Expr<$x>);
        impl_pome!(Expr<$x>, Var<$x>);
        impl_emop!(Var<$x>, Expr<$x>);
        $(impl_pome!(Expr<$x>, $y);
        impl_emop!($y, Expr<$x>);)*
    };
}
pomes!(f16);
pomes!(f32);
pomes!(f64);
pomes!(i16);
pomes!(i32);
pomes!(i64);
pomes!(u16);
pomes!(u32);
pomes!(u64);

pomes!(Float2, Expr<PackedFloat2>, f32);
pomes!(Float3, Expr<PackedFloat3>, f32);
pomes!(Float4, Expr<PackedFloat4>, f32);
pomes!(Double2);
pomes!(Double3);
pomes!(Double4);
pomes!(Int2, Expr<PackedInt2>);
pomes!(Int3, Expr<PackedInt3>);
pomes!(Int4, Expr<PackedInt4>);
pomes!(Uint2, Expr<PackedUint2>);
pomes!(Uint3, Expr<PackedUint3>);
pomes!(Uint4, Expr<PackedUint4>);

#[allow(dead_code)]
fn tests() {
    <_ as BoolWhileMaybeExpr>::while_loop(|| true, || {});
    <_ as BoolWhileMaybeExpr>::while_loop(|| Bool::from(true), || {});
}
