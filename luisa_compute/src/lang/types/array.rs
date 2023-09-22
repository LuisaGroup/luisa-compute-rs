use std::ops::Index;

use super::*;
use crate::lang::index::IntoIndex;
use ir::ArrayType;

impl<T: Value, const N: usize> Value for [T; N] {
    type Expr = ArrayExpr<T, N>;
    type Var = ArrayVar<T, N>;
}
impl<T: Value, const N: usize> ArrayNewExpr<T, N> for [T; N] {
    fn expr_from_elements(elems: [Expr<T>; N]) -> Expr<Self> {
        let node =
            __current_scope(|b| b.call(Func::Array, &elems.map(|e| e.node()), <[T; N]>::type_()));
        Expr::<Self>::from_node(node)
    }
}

impl_simple_expr_proxy!([T: Value, const N: usize] ArrayExpr[T, N] for [T; N]);
impl_simple_var_proxy!([T: Value, const N: usize] ArrayVar[T, N] for [T; N]);

impl<T: Value, const N: usize> ArrayExpr<T, N> {
    pub fn len(&self) -> Expr<u32> {
        (N as u32).expr()
    }
}

impl<T: Value, const N: usize, X: IntoIndex> Index<X> for ArrayExpr<T, N> {
    type Output = Expr<T>;
    fn index(&self, i: X) -> &Self::Output {
        let i = i.to_u64();

        // TODO: Add need_runtime_check()?
        if need_runtime_check() {
            lc_assert!(i.lt((N as u64).expr()));
        }

        Expr::<T>::from_node(__current_scope(|b| {
            b.call(Func::ExtractElement, &[self.0.node, i.node()], T::type_())
        }))
        ._ref()
    }
}
macro_rules! impl_array_vec_conversion{
    ($N:literal,$($xs:expr,)*)=>{
        impl<T: Value> From<Expr<[T; $N]>> for Expr<Vector<T, $N>>
            where T: vector::VectorAlign<$N>
        {
            fn from(array: Expr<[T; $N]>) -> Self {
                Vector::<T, $N>::expr($(array[$xs]),*)
            }
        }
        impl<T:Value> From<Expr<Vector<T,$N>>> for Expr<[T;$N]>
            where T: vector::VectorAlign<$N>,
        {
            fn from(vec:Expr<Vector<T,$N>>)->Self{
                let elems = (0..$N).map(|i| __extract::<T>(vec.node(), i)).collect::<Vec<_>>();
                let node = __current_scope(|b| b.call(Func::Array, &elems, <[T;$N]>::type_()));
                Self::from_node(node)
            }
        }
    }
}
impl_array_vec_conversion!(2, 0, 1,);
impl_array_vec_conversion!(3, 0, 1, 2,);
impl_array_vec_conversion!(4, 0, 1, 2, 3,);
#[derive(Clone, Copy, Debug)]
pub struct VLArrayExpr<T: Value> {
    marker: PhantomData<T>,
    pub(super) node: NodeRef,
}

impl<T: Value> FromNode for VLArrayExpr<T> {
    fn from_node(node: NodeRef) -> Self {
        Self {
            marker: PhantomData,
            node,
        }
    }
}

impl<T: Value> ToNode for VLArrayExpr<T> {
    fn node(&self) -> NodeRef {
        self.node
    }
}

impl<T: Value> Aggregate for VLArrayExpr<T> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        nodes.push(self.node);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        Self::from_node(iter.next().unwrap())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VLArrayVar<T: Value> {
    marker: PhantomData<T>,
    node: NodeRef,
}

impl<T: Value> FromNode for VLArrayVar<T> {
    fn from_node(node: NodeRef) -> Self {
        Self {
            marker: PhantomData,
            node,
        }
    }
}

impl<T: Value> ToNode for VLArrayVar<T> {
    fn node(&self) -> NodeRef {
        self.node
    }
}

impl<T: Value> Aggregate for VLArrayVar<T> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        nodes.push(self.node);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        Self::from_node(iter.next().unwrap())
    }
}

impl<T: Value> VLArrayVar<T> {
    pub fn read<I: Into<Expr<u32>>>(&self, i: I) -> Expr<T> {
        let i = i.into();
        if need_runtime_check() {
            lc_assert!(i.lt(self.len()), "VLArrayVar::read out of bounds");
        }

        Expr::<T>::from_node(__current_scope(|b| {
            let gep = b.call(Func::GetElementPtr, &[self.node, i.node()], T::type_());
            b.call(Func::Load, &[gep], T::type_())
        }))
    }
    pub fn len(&self) -> Expr<u32> {
        match self.node.type_().as_ref() {
            Type::Array(ArrayType { element: _, length }) => (*length as u32).expr(),
            _ => unreachable!(),
        }
    }
    pub fn static_len(&self) -> usize {
        match self.node.type_().as_ref() {
            Type::Array(ArrayType { element: _, length }) => *length,
            _ => unreachable!(),
        }
    }
    pub fn write<I: Into<Expr<u32>>, V: Into<Expr<T>>>(&self, i: I, value: V) {
        let i = i.into();
        let value = value.into();

        if need_runtime_check() {
            lc_assert!(i.lt(self.len()), "VLArrayVar::read out of bounds");
        }

        __current_scope(|b| {
            let gep = b.call(Func::GetElementPtr, &[self.node, i.node()], T::type_());
            b.update(gep, value.node());
        });
    }
    pub fn load(&self) -> VLArrayExpr<T> {
        VLArrayExpr::from_node(__current_scope(|b| {
            b.call(Func::Load, &[self.node], self.node.type_().clone())
        }))
    }
    pub fn store(&self, value: VLArrayExpr<T>) {
        __current_scope(|b| {
            b.update(self.node, value.node);
        });
    }
    pub fn zero(length: usize) -> Self {
        FromNode::from_node(__current_scope(|b| {
            b.local_zero_init(ir::context::register_type(Type::Array(ArrayType {
                element: T::type_(),
                length,
            })))
        }))
    }
}

impl<T: Value> VLArrayExpr<T> {
    pub fn zero(length: usize) -> Self {
        let node = __current_scope(|b| {
            b.call(
                Func::ZeroInitializer,
                &[],
                ir::context::register_type(Type::Array(ArrayType {
                    element: T::type_(),
                    length,
                })),
            )
        });
        Self::from_node(node)
    }
    pub fn static_len(&self) -> usize {
        match self.node.type_().as_ref() {
            Type::Array(ArrayType { element: _, length }) => *length,
            _ => unreachable!(),
        }
    }
    pub fn read<I: IntoIndex>(&self, i: I) -> Expr<T> {
        let i = i.to_u64();
        if need_runtime_check() {
            lc_assert!(i.lt(self.len()));
        }

        Expr::<T>::from_node(__current_scope(|b| {
            b.call(Func::ExtractElement, &[self.node, i.node()], T::type_())
        }))
    }
    pub fn len(&self) -> Expr<u64> {
        match self.node.type_().as_ref() {
            Type::Array(ArrayType { element: _, length }) => (*length as u64).expr(),
            _ => unreachable!(),
        }
    }
}
