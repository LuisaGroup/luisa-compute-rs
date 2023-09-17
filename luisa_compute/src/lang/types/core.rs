use super::*;

// This is a hack in order to get rust-analyzer to display type hints as Expr<f32>
// instead of Expr<f32>, which is rather redundant and generally clutters things up.
mod _prim {
    use super::NodeRef;

    #[derive(Clone, Copy, Debug)]
    pub struct Expr<T> {
        pub(crate) node: NodeRef,
        pub(crate) _phantom: std::marker::PhantomData<T>,
    }

    #[derive(Clone, Copy, Debug)]
    pub struct Var<T> {
        pub(crate) node: NodeRef,
        pub(crate) _phantom: std::marker::PhantomData<T>,
    }
}

impl<T> Aggregate for _prim::Expr<T> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        nodes.push(self.node);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        Self {
            node: iter.next().unwrap(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> Aggregate for _prim::Var<T> {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        nodes.push(self.node);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        Self {
            node: iter.next().unwrap(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Copy + 'static + Value> FromNode for _prim::Expr<T> {
    fn from_node(node: NodeRef) -> Self {
        Self {
            node,
            _phantom: std::marker::PhantomData,
        }
    }
}
impl<T: Copy + 'static + Value> ToNode for _prim::Expr<T> {
    fn node(&self) -> NodeRef {
        self.node
    }
}

impl<T: Value> Deref for _prim::Var<T>
where
    _prim::Var<T>: VarProxy<Value = T>,
{
    type Target = T::Expr;
    fn deref(&self) -> &Self::Target {
        self._deref()
    }
}

macro_rules! impl_prim {
    ($t:ty) => {
        impl From<$t> for _prim::Expr<$t> {
            fn from(v: $t) -> Self {
                const_(v)
            }
        }
        impl From<Var<$t>> for _prim::Expr<$t> {
            fn from(v: Var<$t>) -> Self {
                v.load()
            }
        }
        impl FromNode for _prim::Var<$t> {
            fn from_node(node: NodeRef) -> Self {
                Self {
                    node,
                    _phantom: std::marker::PhantomData,
                }
            }
        }
        impl ToNode for _prim::Var<$t> {
            fn node(&self) -> NodeRef {
                self.node
            }
        }
        impl ExprProxy for _prim::Expr<$t> {
            type Value = $t;
        }
        impl VarProxy for _prim::Var<$t> {
            type Value = $t;
        }
        impl Value for $t {
            type Expr = _prim::Expr<$t>;
            type Var = _prim::Var<$t>;
            fn fields() -> Vec<String> {
                vec![]
            }
        }
        impl_callable_param!($t, _prim::Expr<$t>, _prim::Var<$t>);
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
