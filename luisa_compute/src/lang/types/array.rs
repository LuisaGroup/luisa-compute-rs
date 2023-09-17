use super::*;

#[derive(Clone, Copy, Debug)]
pub struct ArrayExpr<T: Value, const N: usize> {
    marker: std::marker::PhantomData<T>,
    node: NodeRef,
}

#[derive(Clone, Copy, Debug)]
pub struct ArrayVar<T: Value, const N: usize> {
    marker: std::marker::PhantomData<T>,
    node: NodeRef,
}

impl<T: Value, const N: usize> FromNode for ArrayExpr<T, N> {
    fn from_node(node: NodeRef) -> Self {
        Self {
            marker: std::marker::PhantomData,
            node,
        }
    }
}

impl<T: Value, const N: usize> ToNode for ArrayExpr<T, N> {
    fn node(&self) -> NodeRef {
        self.node
    }
}

impl<T: Value, const N: usize> Aggregate for ArrayExpr<T, N> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        nodes.push(self.node);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        Self::from_node(iter.next().unwrap())
    }
}

impl<T: Value, const N: usize> FromNode for ArrayVar<T, N> {
    fn from_node(node: NodeRef) -> Self {
        Self {
            marker: std::marker::PhantomData,
            node,
        }
    }
}

impl<T: Value, const N: usize> ToNode for ArrayVar<T, N> {
    fn node(&self) -> NodeRef {
        self.node
    }
}

impl<T: Value, const N: usize> Aggregate for ArrayVar<T, N> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        nodes.push(self.node);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        Self::from_node(iter.next().unwrap())
    }
}

impl<T: Value, const N: usize> ExprProxy for ArrayExpr<T, N> {
    type Value = [T; N];
}

impl<T: Value, const N: usize> VarProxy for ArrayVar<T, N> {
    type Value = [T; N];
}

impl<T: Value, const N: usize> ArrayVar<T, N> {
    pub fn len(&self) -> Expr<u32> {
        const_(N as u32)
    }
}

impl<T: Value, const N: usize> ArrayExpr<T, N> {
    pub fn zero() -> Self {
        let node = __current_scope(|b| b.call(Func::ZeroInitializer, &[], <[T; N]>::type_()));
        Self::from_node(node)
    }
    pub fn len(&self) -> Expr<u32> {
        const_(N as u32)
    }
}

impl<T: Value, const N: usize> IndexRead for ArrayExpr<T, N> {
    type Element = T;
    fn read<I: IntoIndex>(&self, i: I) -> Expr<T> {
        let i = i.to_u64();

        lc_assert!(i.cmplt(const_(N as u64)));

        Expr::<T>::from_node(__current_scope(|b| {
            b.call(Func::ExtractElement, &[self.node, i.node()], T::type_())
        }))
    }
}

impl<T: Value, const N: usize> IndexRead for ArrayVar<T, N> {
    type Element = T;
    fn read<I: IntoIndex>(&self, i: I) -> Expr<T> {
        let i = i.to_u64();
        if need_runtime_check() {
            lc_assert!(i.cmplt(const_(N as u64)));
        }

        Expr::<T>::from_node(__current_scope(|b| {
            let gep = b.call(Func::GetElementPtr, &[self.node, i.node()], T::type_());
            b.call(Func::Load, &[gep], T::type_())
        }))
    }
}

impl<T: Value, const N: usize> IndexWrite for ArrayVar<T, N> {
    fn write<I: IntoIndex, V: Into<Expr<T>>>(&self, i: I, value: V) {
        let i = i.to_u64();
        let value = value.into();

        if need_runtime_check() {
            lc_assert!(i.cmplt(const_(N as u64)));
        }

        __current_scope(|b| {
            let gep = b.call(Func::GetElementPtr, &[self.node, i.node()], T::type_());
            b.update(gep, value.node());
        });
    }
}

impl<T: Value + TypeOf, const N: usize> Value for [T; N] {
    type Expr = ArrayExpr<T, N>;
    type Var = ArrayVar<T, N>;
    fn fields() -> Vec<String> {
        todo!("why this method exists?")
    }
}

// TODO: What's the point of the two separate arrays?
#[derive(Clone, Copy, Debug)]
pub struct VLArrayExpr<T: Value> {
    marker: std::marker::PhantomData<T>,
    node: NodeRef,
}

impl<T: Value> FromNode for VLArrayExpr<T> {
    fn from_node(node: NodeRef) -> Self {
        Self {
            marker: std::marker::PhantomData,
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
    marker: std::marker::PhantomData<T>,
    node: NodeRef,
}

impl<T: Value> FromNode for VLArrayVar<T> {
    fn from_node(node: NodeRef) -> Self {
        Self {
            marker: std::marker::PhantomData,
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
            lc_assert!(i.cmplt(self.len()), "VLArrayVar::read out of bounds");
        }

        Expr::<T>::from_node(__current_scope(|b| {
            let gep = b.call(Func::GetElementPtr, &[self.node, i.node()], T::type_());
            b.call(Func::Load, &[gep], T::type_())
        }))
    }
    pub fn len(&self) -> Expr<u32> {
        match self.node.type_().as_ref() {
            Type::Array(ArrayType { element: _, length }) => const_(*length as u32),
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
            lc_assert!(i.cmplt(self.len()), "VLArrayVar::read out of bounds");
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
            lc_assert!(i.cmplt(self.len()));
        }

        Expr::<T>::from_node(__current_scope(|b| {
            b.call(Func::ExtractElement, &[self.node, i.node()], T::type_())
        }))
    }
    pub fn len(&self) -> Expr<u64> {
        match self.node.type_().as_ref() {
            Type::Array(ArrayType { element: _, length }) => const_(*length as u64),
            _ => unreachable!(),
        }
    }
}
