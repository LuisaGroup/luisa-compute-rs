use std::ops::Index;

use super::*;
use crate::lang::index::IntoIndex;
use ir::ArrayType;

impl<T: Value, const N: usize> Value for [T; N] {
    type Expr = ArrayExpr<T, N>;
    type Var = ArrayVar<T, N>;
    type AtomicRef = ArrayAtomicRef<T, N>;
}
impl<T: Value, const N: usize> ArrayNewExpr<T, N> for [T; N] {
    fn from_elems_expr(elems: [Expr<T>; N]) -> Expr<Self> {
        let elems = elems.map(|e| e.node().get());
        let node = __current_scope(|b| b.call(Func::Array, &elems, <[T; N]>::type_()));
        Expr::<Self>::from_node(node.into())
    }
}

impl_simple_expr_proxy!([T: Value, const N: usize] ArrayExpr[T, N] for [T; N]);
impl_simple_var_proxy!([T: Value, const N: usize] ArrayVar[T, N] for [T; N]);
impl_simple_atomic_ref_proxy!([T: Value, const N: usize] ArrayAtomicRef[T, N] for [T; N]);
#[derive(Clone)]
pub struct ArraySoa<T: SoaValue, const N: usize> {
    pub(crate) elems: Vec<T::SoaBuffer>,
    _marker: PhantomData<[T; N]>,
}
impl<T: SoaValue, const N: usize> SoaValue for [T; N] {
    type SoaBuffer = ArraySoa<T, N>;
}
impl<T: SoaValue, const N: usize> SoaBufferProxy for ArraySoa<T, N> {
    type Value = [T; N];
    fn from_soa_storage(
        storage: ByteBufferVar,
        meta: Expr<SoaMetadata>,
        global_offset: usize,
    ) -> Self {
        let elems = (0..N)
            .map(|i| {
                T::SoaBuffer::from_soa_storage(
                    storage.clone(),
                    meta,
                    global_offset + i * T::SoaBuffer::num_buffers(),
                )
            })
            .collect::<Vec<_>>();
        Self {
            elems,
            _marker: PhantomData,
        }
    }
    fn num_buffers() -> usize {
        T::SoaBuffer::num_buffers() * N
    }
}
impl<T: SoaValue, const N: usize> IndexRead for ArraySoa<T, N> {
    type Element = [T; N];
    fn read<I: IntoIndex>(&self, i: I) -> Expr<Self::Element> {
        let i = i.to_u64();
        let elems = (0..N).map(|j| self.elems[j].read(i)).collect::<Vec<_>>();
        <[T; N]>::from_elems_expr(elems.try_into().unwrap_or_else(|_| unreachable!()))
    }
}
impl<T: SoaValue, const N: usize> IndexWrite for ArraySoa<T, N> {
    fn write<I: IntoIndex, V: AsExpr<Value = Self::Element>>(&self, i: I, value: V) {
        let i = i.to_u64();
        let value = value.as_expr();
        for j in 0..N {
            self.elems[j].write(i, value.read(j as u64));
        }
    }
}
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
            check_index_lt_usize(i, N);
        }
        let i = i.node().get();
        let self_node = self.0.node().get();
        Expr::<T>::from_node(
            __current_scope(|b| b.call(Func::ExtractElement, &[self_node, i], T::type_())).into(),
        )
        ._ref()
    }
}
impl<T: Value, const N: usize, X: IntoIndex> Index<X> for ArrayAtomicRef<T, N> {
    type Output = AtomicRef<T>;
    fn index(&self, i: X) -> &Self::Output {
        let i = i.to_u64();

        // TODO: Add need_runtime_check()?
        if need_runtime_check() {
            check_index_lt_usize(i, N);
        }
        let node = self.0.node.get();
        let inst = node.get().instruction.as_ref();
        let mut args = match inst {
            Instruction::Call(f, args) => match f {
                Func::AtomicRef => args.to_vec(),
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };
        args.push(i.node().get());
        AtomicRef::<T>::from_node(
            __current_scope(|b| b.call_no_append(Func::AtomicRef, &args, T::type_())).into(),
        )
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
                let elems = (0..$N).map(|i| __extract::<T>(vec.node().get(), i)).collect::<Vec<_>>();
                let node = __current_scope(|b| b.call(Func::Array, &elems, <[T;$N]>::type_()));
                Self::from_node(node.into())
            }
        }
    }
}
impl<T: Value, const N: usize> IndexRead for Expr<[T; N]> {
    type Element = T;
    fn read<I: IntoIndex>(&self, i: I) -> Expr<Self::Element> {
        let i = i.to_u64();
        if need_runtime_check() {
            check_index_lt_usize(i, N);
        }
        let self_node = self.node().get();
        let i = i.node().get();
        Expr::<T>::from_node(
            __current_scope(|b| b.call(Func::ExtractElement, &[self_node, i], T::type_())).into(),
        )
    }
}
impl<T: Value, const N: usize> IndexRead for Var<[T; N]> {
    type Element = T;
    fn read<I: IntoIndex>(&self, i: I) -> Expr<Self::Element> {
        let i = i.to_u64();
        if need_runtime_check() {
            check_index_lt_usize(i, N);
        }
        let self_node = self.node().get();
        let i = i.node().get();
        Expr::<T>::from_node(
            __current_scope(|b| {
                let gep = b.call(Func::GetElementPtr, &[self_node, i], T::type_());
                b.load(gep)
            })
            .into(),
        )
    }
}
impl<T: Value, const N: usize> IndexWrite for Var<[T; N]> {
    fn write<I: IntoIndex, V: AsExpr<Value = Self::Element>>(&self, i: I, value: V) {
        let i = i.to_u64();
        let value = value.as_expr();
        if need_runtime_check() {
            check_index_lt_usize(i, N);
        }
        let self_node = self.node().get();
        let i = i.node().get();
        let value = value.node().get();
        __current_scope(|b| {
            let gep = b.call(Func::GetElementPtr, &[self_node, i], T::type_());
            b.update(gep, value);
        });
    }
}
impl_array_vec_conversion!(2, 0, 1,);
impl_array_vec_conversion!(3, 0, 1, 2,);
impl_array_vec_conversion!(4, 0, 1, 2, 3,);
#[derive(Clone, Copy, Debug)]
pub struct VLArrayExpr<T: Value> {
    marker: PhantomData<T>,
    pub(super) node: SafeNodeRef,
}

impl<T: Value> FromNode for VLArrayExpr<T> {
    fn from_node(node: SafeNodeRef) -> Self {
        Self {
            marker: PhantomData,
            node,
        }
    }
}

impl<T: Value> ToNode for VLArrayExpr<T> {
    fn node(&self) -> SafeNodeRef {
        self.node
    }
}

impl<T: Value> Aggregate for VLArrayExpr<T> {
    fn to_nodes(&self, nodes: &mut Vec<SafeNodeRef>) {
        nodes.push(self.node);
    }
    fn from_nodes<I: Iterator<Item = SafeNodeRef>>(iter: &mut I) -> Self {
        Self::from_node(iter.next().unwrap())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VLArrayVar<T: Value> {
    marker: PhantomData<T>,
    node: SafeNodeRef,
}

impl<T: Value> FromNode for VLArrayVar<T> {
    fn from_node(node: SafeNodeRef) -> Self {
        Self {
            marker: PhantomData,
            node,
        }
    }
}

impl<T: Value> ToNode for VLArrayVar<T> {
    fn node(&self) -> SafeNodeRef {
        self.node
    }
}

impl<T: Value> Aggregate for VLArrayVar<T> {
    fn to_nodes(&self, nodes: &mut Vec<SafeNodeRef>) {
        nodes.push(self.node);
    }
    fn from_nodes<I: Iterator<Item = SafeNodeRef>>(iter: &mut I) -> Self {
        Self::from_node(iter.next().unwrap())
    }
}
impl<T: Value> IndexRead for VLArrayVar<T> {
    type Element = T;
    fn read<I: IntoIndex>(&self, i: I) -> Expr<Self::Element> {
        let i = i.to_u64();
        if need_runtime_check() {
            check_index_lt_usize(i, self.len());
        }
        let self_node = self.node.get();
        let i = i.node().get();
        Expr::<T>::from_node(
            __current_scope(|b| {
                let gep = b.call(Func::GetElementPtr, &[self_node, i], T::type_());
                b.call(Func::Load, &[gep], T::type_())
            })
            .into(),
        )
    }
}
impl<T: Value> IndexWrite for VLArrayVar<T> {
    fn write<I: IntoIndex, V: AsExpr<Value = Self::Element>>(&self, i: I, value: V) {
        let i = i.to_u64();
        let value = value.as_expr();

        if need_runtime_check() {
            check_index_lt_usize(i, self.len());
        }
        let self_node = self.node.get();
        let i = i.node().get();
        let value = value.node().get();
        __current_scope(|b| {
            let gep = b.call(Func::GetElementPtr, &[self_node, i], T::type_());
            b.update(gep, value);
        });
    }
}
impl<T: Value> VLArrayVar<T> {
    pub fn len_expr(&self) -> Expr<u64> {
        match self.node.get().type_().as_ref() {
            Type::Array(ArrayType { element: _, length }) => (*length as u64).expr(),
            _ => unreachable!(),
        }
    }
    pub fn len(&self) -> usize {
        match self.node.get().type_().as_ref() {
            Type::Array(ArrayType { element: _, length }) => *length,
            _ => unreachable!(),
        }
    }
    pub fn load(&self) -> VLArrayExpr<T> {
        let node = self.node.get();
        VLArrayExpr::from_node(
            __current_scope(|b| b.call(Func::Load, &[node], node.type_().clone())).into(),
        )
    }
    pub fn store(&self, value: VLArrayExpr<T>) {
        let node = self.node.get();
        let value = value.node.get();
        __current_scope(|b| {
            b.update(node, value);
        });
    }
    pub fn zero(length: usize) -> Self {
        FromNode::from_node(
            __current_scope(|b| {
                b.local_zero_init(ir::context::register_type(Type::Array(ArrayType {
                    element: T::type_(),
                    length,
                })))
            })
            .into(),
        )
    }
}
impl<T: Value> IndexRead for VLArrayExpr<T> {
    type Element = T;
    fn read<I: IntoIndex>(&self, i: I) -> Expr<Self::Element> {
        let i = i.to_u64();
        if need_runtime_check() {
            check_index_lt_usize(i, self.len());
        }
        let node = self.node.get();
        let i = i.node().get();
        Expr::<T>::from_node(
            __current_scope(|b| b.call(Func::ExtractElement, &[node, i], T::type_())).into(),
        )
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
        Self::from_node(node.into())
    }
    pub fn len(&self) -> usize {
        match self.node.get().type_().as_ref() {
            Type::Array(ArrayType { element: _, length }) => *length,
            _ => unreachable!(),
        }
    }
    pub fn len_expr(&self) -> Expr<u64> {
        match self.node.get().type_().as_ref() {
            Type::Array(ArrayType { element: _, length }) => (*length as u64).expr(),
            _ => unreachable!(),
        }
    }
}
