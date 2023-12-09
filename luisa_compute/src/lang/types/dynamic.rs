use super::array::{VLArrayExpr, VLArrayVar};
use super::*;
use ir::ArrayType;
use std::any::Any;
use std::rc::Rc;

#[derive(Clone, Copy)]
pub struct DynExpr {
    /// we need the actuall node to be resolved here
    /// so it must be a [`NodeRef`]
    node: NodeRef,
}

impl<V: Value> From<Expr<V>> for DynExpr {
    fn from(value: Expr<V>) -> Self {
        Self {
            node: value.node().get(),
        }
    }
}

impl<V: Value> From<Var<V>> for DynVar {
    fn from(value: Var<V>) -> Self {
        Self {
            node: value.node().get(),
        }
    }
}

impl DynExpr {
    pub fn downcast<T: Value>(&self) -> Option<Expr<T>> {
        if ir::context::is_type_equal(self.node.type_(), &T::type_()) {
            Some(Expr::<T>::from_node(self.node.into()))
        } else {
            None
        }
    }
    pub fn get<T: Value>(&self) -> Expr<T> {
        self.downcast::<T>().unwrap_or_else(|| {
            panic!(
                "DynExpr::get: type mismatch: expected {}, got {}",
                std::any::type_name::<T>(),
                self.node.type_().to_string()
            )
        })
    }
    pub fn downcast_array<T: Value>(&self, len: usize) -> Option<VLArrayExpr<T>> {
        let array_type = ir::context::register_type(Type::Array(ArrayType {
            element: T::type_(),
            length: len,
        }));
        if ir::context::is_type_equal(self.node.type_(), &array_type) {
            Some(VLArrayExpr::<T>::from_node(self.node.into()))
        } else {
            None
        }
    }
    pub fn get_array<T: Value>(&self, len: usize) -> VLArrayExpr<T> {
        let array_type = ir::context::register_type(Type::Array(ArrayType {
            element: T::type_(),
            length: len,
        }));
        self.downcast_array::<T>(len).unwrap_or_else(|| {
            panic!(
                "DynExpr::get: type mismatch: expected {}, got {}",
                array_type,
                self.node.type_().to_string()
            )
        })
    }
    pub fn new<V: Value>(expr: Expr<V>) -> Self {
        Self {
            node: expr.node().get(),
        }
    }
}

impl CallableParameter for DynExpr {
    fn def_param(arg: Option<Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self {
        let arg = arg.unwrap_or_else(|| panic!("DynExpr should be used in DynCallable only!"));
        let arg = arg.downcast_ref::<Self>().unwrap();
        let node = builder.arg(arg.node.type_().clone(), true).into();
        Self { node }
    }
    fn encode(&self, encoder: &mut CallableArgEncoder) {
        encoder.args.push(self.node)
    }
}

impl Aggregate for DynExpr {
    fn to_nodes(&self, nodes: &mut Vec<SafeNodeRef>) {
        nodes.push(self.node.into())
    }
    fn from_nodes<I: Iterator<Item = SafeNodeRef>>(iter: &mut I) -> Self {
        Self {
            node: iter.next().unwrap().get(),
        }
    }
}

impl FromNode for DynExpr {
    fn from_node(node: SafeNodeRef) -> Self {
        Self { node: node.get() }
    }
}

impl ToNode for DynExpr {
    fn node(&self) -> SafeNodeRef {
        self.node.into()
    }
}

unsafe impl CallableRet for DynExpr {
    fn _return(&self) -> CArc<Type> {
        let node = self.node;
        __current_scope(|b| {
            b.return_(node);
        });
        self.node.type_().clone()
    }
    fn _from_return(node: NodeRef) -> Self {
        Self::from_node(node.into())
    }
    fn _return_type() -> CArc<Type> {
        panic!("should not be called")
    }
}

impl Aggregate for DynVar {
    fn to_nodes(&self, nodes: &mut Vec<SafeNodeRef>) {
        nodes.push(self.node.into())
    }
    fn from_nodes<I: Iterator<Item = SafeNodeRef>>(iter: &mut I) -> Self {
        Self {
            node: iter.next().unwrap().get(),
        }
    }
}

impl FromNode for DynVar {
    fn from_node(node: SafeNodeRef) -> Self {
        Self { node: node.get() }
    }
}

impl ToNode for DynVar {
    fn node(&self) -> SafeNodeRef {
        self.node.into()
    }
}

#[derive(Clone, Copy)]
pub struct DynVar {
    node: NodeRef,
}

impl CallableParameter for DynVar {
    fn def_param(arg: Option<Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self {
        let arg = arg.unwrap_or_else(|| panic!("DynVar should be used in DynCallable only!"));
        let arg = arg.downcast_ref::<Self>().unwrap();
        let node = builder.arg(arg.node.type_().clone(), false).into();
        Self { node }
    }
    fn encode(&self, encoder: &mut CallableArgEncoder) {
        encoder.args.push(self.node)
    }
}

impl DynVar {
    pub fn downcast<T: Value>(&self) -> Option<Var<T>> {
        if ir::context::is_type_equal(self.node.type_(), &T::type_()) {
            Some(Var::<T>::from_node(self.node.into()))
        } else {
            None
        }
    }
    pub fn get<T: Value>(&self) -> Var<T> {
        self.downcast::<T>().unwrap_or_else(|| {
            panic!(
                "DynVar::get: type mismatch: expected {}, got {}",
                std::any::type_name::<T>(),
                self.node.type_().to_string()
            )
        })
    }
    pub fn downcast_array<T: Value>(&self, len: usize) -> Option<VLArrayVar<T>> {
        let array_type = ir::context::register_type(Type::Array(ArrayType {
            element: T::type_(),
            length: len,
        }));
        if ir::context::is_type_equal(self.node.type_(), &array_type) {
            Some(VLArrayVar::<T>::from_node(self.node.into()))
        } else {
            None
        }
    }
    pub fn get_array<T: Value>(&self, len: usize) -> VLArrayVar<T> {
        let array_type = ir::context::register_type(Type::Array(ArrayType {
            element: T::type_(),
            length: len,
        }));
        self.downcast_array::<T>(len).unwrap_or_else(|| {
            panic!(
                "DynExpr::get: type mismatch: expected {}, got {}",
                array_type,
                self.node.type_().to_string()
            )
        })
    }
    pub fn load(&self) -> DynExpr {
        let self_node = self.node;
        DynExpr {
            node: __current_scope(|b| b.call(Func::Load, &[self_node], self_node.type_().clone()))
                .into(),
        }
    }
    pub fn store(&self, value: &DynExpr) {
        let self_node = self.node;
        let value = value.node;
        __current_scope(|b| b.update(self_node, value));
    }
    pub fn zero<T: Value>() -> Self {
        let v = Var::<T>::zeroed();
        Self {
            node: v.node().get(),
        }
    }
}
