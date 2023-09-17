use super::array::VLArrayExpr;
use super::*;
use crate::lang::index::IntoIndex;
use ir::ArrayType;

pub struct Shared<T: Value> {
    marker: PhantomData<T>,
    node: NodeRef,
}
impl<T: Value> Shared<T> {
    pub fn new(length: usize) -> Self {
        Self {
            marker: PhantomData,
            node: __current_scope(|b| {
                let shared = new_node(
                    b.pools(),
                    Node::new(
                        CArc::new(Instruction::Shared),
                        ir::context::register_type(Type::Array(ArrayType {
                            element: T::type_(),
                            length,
                        })),
                    ),
                );
                RECORDER.with(|r| {
                    let mut r = r.borrow_mut();
                    r.shared.push(shared);
                });
                shared
            }),
        }
    }
    pub fn len(&self) -> Expr<u64> {
        match self.node.type_().as_ref() {
            Type::Array(ArrayType { element: _, length }) => (*length as u64).expr(),
            _ => unreachable!(),
        }
    }
    pub fn static_len(&self) -> usize {
        match self.node.type_().as_ref() {
            Type::Array(ArrayType { element: _, length }) => *length,
            _ => unreachable!(),
        }
    }
    pub fn write<I: IntoIndex, V: Into<Expr<T>>>(&self, i: I, value: V) {
        let i = i.to_u64();
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
}
