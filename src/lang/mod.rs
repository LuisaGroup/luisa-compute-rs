use std::ops::Deref;

use crate::*;
pub use ir::ir::NodeRef;
use luisa_compute_ir as ir;

pub trait Value: Copy + ir::TypeOf {
    type Proxy;
    type SoN: StructOfNodes;

    fn destruct(v: Var<Self>) -> Self::SoN;
    fn construct(s: Self::SoN) -> Var<Self>;
}

/* Struct of Nodes
*  Similar to how enoki, Dr.Jit handles structs
*  Makes every field of the struct a node
*  Most concise way to manipulate structs
*/
pub trait StructOfNodes {
    fn to_vec_nodes(&self) -> Vec<NodeRef> {
        let mut nodes = vec![];
        Self::to_nodes(&self, &mut nodes);
        nodes
    }
    fn from_vec_nodes(nodes: Vec<NodeRef>) -> Self {
        let mut iter = nodes.into_iter();
        let ret = Self::from_nodes(&mut iter);
        assert!(iter.next().is_none());
        ret
    }
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>);
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self;
}
pub trait Selectable {
    fn select(mask: Mask, lhs: Self, rhs: Self) -> Self;
}
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Var<T: Value> {
    pub(crate) proxy: T::Proxy,
}
pub type Mask = Var<bool>;

impl<T: Value> Var<T> {
    pub fn set(&self, value: Self) {
        unimplemented!()
    }
    pub fn expand(&self) -> T::SoN {
        unimplemented!()
    }
    pub fn collect(value: T::SoN) -> Self {
        unimplemented!()
    }
}

impl<T: Value> Deref for Var<T> {
    type Target = T::Proxy;
    fn deref(&self) -> &Self::Target {
        &self.proxy
    }
}
