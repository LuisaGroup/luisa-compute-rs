use super::array::VLArrayExpr;
use super::*;
use crate::lang::index::IntoIndex;
use ir::ArrayType;

pub struct Shared<T: Value> {
    marker: PhantomData<T>,
    node: SafeNodeRef,
}
impl<T: Value> Shared<T> {
    pub fn new(length: usize) -> Self {
        let shared = with_recorder(|r| {
            let shared = new_node(
                &r.pools,
                Node::new(
                    CArc::new(Instruction::Shared),
                    ir::context::register_type(Type::Array(ArrayType {
                        element: T::type_(),
                        length,
                    })),
                ),
            )
            .into();

            r.shared.push(shared);
            shared
        });
        Self {
            marker: PhantomData,
            node: shared.into(),
        }
    }
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
        let self_node = self.node.get();
        VLArrayExpr::from_node(
            __current_scope(|b| b.call(Func::Load, &[self_node], self_node.type_().clone())).into(),
        )
    }
    pub fn store(&self, value: VLArrayExpr<T>) {
        let self_node = self.node.get();
        let value = value.node().get();
        __current_scope(|b| {
            b.update(self_node, value);
        });
    }
}
impl<T: Value> IndexRead for Shared<T> {
    type Element = T;
    fn read<I: IntoIndex>(&self, i: I) -> Expr<Self::Element> {
        let i = i.to_u64();

        if need_runtime_check() {
            check_index_lt_usize(i, self.len());
        }
        let i = i.node().get();
        let self_node = self.node.get();
        Expr::from_node(
            __current_scope(|b| {
                let gep = b.call(Func::GetElementPtr, &[self_node, i], T::type_());
                b.load(gep)
            })
            .into(),
        )
    }
}

impl<T: Value> IndexWrite for Shared<T> {
    fn write<I: IntoIndex, V: AsExpr<Value = Self::Element>>(&self, i: I, value: V) {
        let i = i.to_u64();
        let value = value.as_expr();

        if need_runtime_check() {
            check_index_lt_usize(i, self.len());
        }
        let i = i.node().get();
        let value = value.node().get();
        let self_node = self.node.get();
        __current_scope(|b| {
            let gep = b.call(Func::GetElementPtr, &[self_node, i], T::type_());
            b.update(gep, value);
        });
    }
}
