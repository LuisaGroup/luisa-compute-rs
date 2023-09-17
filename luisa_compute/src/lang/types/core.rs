use super::*;
use std::ops::Deref;

// This is a hack in order to get rust-analyzer to display type hints as Expr<f32>
// instead of Expr<f32>, which is rather redundant and generally clutters things up.
pub(crate) mod prim {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    pub struct Expr<T> {
        pub(crate) node: NodeRef,
        pub(crate) _phantom: PhantomData<T>,
    }

    #[derive(Clone, Copy, Debug)]
    pub struct Var<T> {
        pub(crate) node: NodeRef,
        pub(crate) _phantom: PhantomData<T>,
    }
}

impl<T> Aggregate for prim::Expr<T> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        nodes.push(self.node);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        Self {
            node: iter.next().unwrap(),
            _phantom: PhantomData,
        }
    }
}

impl<T> Aggregate for prim::Var<T> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        nodes.push(self.node);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        Self {
            node: iter.next().unwrap(),
            _phantom: PhantomData,
        }
    }
}

impl<T: Copy + 'static + Value> FromNode for prim::Expr<T> {
    fn from_node(node: NodeRef) -> Self {
        Self {
            node,
            _phantom: PhantomData,
        }
    }
}
impl<T: Copy + 'static + Value> ToNode for prim::Expr<T> {
    fn node(&self) -> NodeRef {
        self.node
    }
}

impl<T: Value> Deref for prim::Var<T>
where
    prim::Var<T>: VarProxy<Value = T>,
{
    type Target = T::Expr;
    fn deref(&self) -> &Self::Target {
        self._deref()
    }
}

macro_rules! impl_prim {
    ($t:ty) => {
        impl From<$t> for prim::Expr<$t> {
            fn from(v: $t) -> Self {
                (v).expr()
            }
        }
        impl From<Var<$t>> for prim::Expr<$t> {
            fn from(v: Var<$t>) -> Self {
                v.load()
            }
        }
        impl FromNode for prim::Var<$t> {
            fn from_node(node: NodeRef) -> Self {
                Self {
                    node,
                    _phantom: PhantomData,
                }
            }
        }
        impl ToNode for prim::Var<$t> {
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        impl ExprProxy for prim::Expr<$t> {
            type Value = $t;
        }
        impl VarProxy for prim::Var<$t> {
            type Value = $t;
        }
        impl Value for $t {
            type Expr = prim::Expr<$t>;
            type Var = prim::Var<$t>;
            fn fields() -> Vec<String> {
                vec![]
            }
        }
        impl_callable_param!($t, prim::Expr<$t>, prim::Var<$t>);
    };
}

impl_prim!(bool);
impl_prim!(u32);
impl_prim!(u64);
impl_prim!(i32);
impl_prim!(i64);
impl_prim!(i16);
impl_prim!(u16);
impl_prim!(f16);
impl_prim!(f32);
impl_prim!(f64);
