use crate::lang::*;
impl<T0: StructOfNodes> StructOfNodes for (T0,) {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        self.0.to_nodes(nodes);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        let v0 = T0::from_nodes(iter);
        (v0,)
    }
}
impl<T0: StructOfNodes, T1: StructOfNodes> StructOfNodes for (T0, T1) {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        self.0.to_nodes(nodes);
        self.1.to_nodes(nodes);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        let v0 = T0::from_nodes(iter);
        let v1 = T1::from_nodes(iter);
        (v0, v1)
    }
}
impl<T0: StructOfNodes, T1: StructOfNodes, T2: StructOfNodes> StructOfNodes for (T0, T1, T2) {
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        self.0.to_nodes(nodes);
        self.1.to_nodes(nodes);
        self.2.to_nodes(nodes);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        let v0 = T0::from_nodes(iter);
        let v1 = T1::from_nodes(iter);
        let v2 = T2::from_nodes(iter);
        (v0, v1, v2)
    }
}
impl<T0: StructOfNodes, T1: StructOfNodes, T2: StructOfNodes, T3: StructOfNodes> StructOfNodes
    for (T0, T1, T2, T3)
{
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        self.0.to_nodes(nodes);
        self.1.to_nodes(nodes);
        self.2.to_nodes(nodes);
        self.3.to_nodes(nodes);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        let v0 = T0::from_nodes(iter);
        let v1 = T1::from_nodes(iter);
        let v2 = T2::from_nodes(iter);
        let v3 = T3::from_nodes(iter);
        (v0, v1, v2, v3)
    }
}
impl<
        T0: StructOfNodes,
        T1: StructOfNodes,
        T2: StructOfNodes,
        T3: StructOfNodes,
        T4: StructOfNodes,
    > StructOfNodes for (T0, T1, T2, T3, T4)
{
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        self.0.to_nodes(nodes);
        self.1.to_nodes(nodes);
        self.2.to_nodes(nodes);
        self.3.to_nodes(nodes);
        self.4.to_nodes(nodes);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        let v0 = T0::from_nodes(iter);
        let v1 = T1::from_nodes(iter);
        let v2 = T2::from_nodes(iter);
        let v3 = T3::from_nodes(iter);
        let v4 = T4::from_nodes(iter);
        (v0, v1, v2, v3, v4)
    }
}
impl<
        T0: StructOfNodes,
        T1: StructOfNodes,
        T2: StructOfNodes,
        T3: StructOfNodes,
        T4: StructOfNodes,
        T5: StructOfNodes,
    > StructOfNodes for (T0, T1, T2, T3, T4, T5)
{
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        self.0.to_nodes(nodes);
        self.1.to_nodes(nodes);
        self.2.to_nodes(nodes);
        self.3.to_nodes(nodes);
        self.4.to_nodes(nodes);
        self.5.to_nodes(nodes);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        let v0 = T0::from_nodes(iter);
        let v1 = T1::from_nodes(iter);
        let v2 = T2::from_nodes(iter);
        let v3 = T3::from_nodes(iter);
        let v4 = T4::from_nodes(iter);
        let v5 = T5::from_nodes(iter);
        (v0, v1, v2, v3, v4, v5)
    }
}
impl<
        T0: StructOfNodes,
        T1: StructOfNodes,
        T2: StructOfNodes,
        T3: StructOfNodes,
        T4: StructOfNodes,
        T5: StructOfNodes,
        T6: StructOfNodes,
    > StructOfNodes for (T0, T1, T2, T3, T4, T5, T6)
{
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        self.0.to_nodes(nodes);
        self.1.to_nodes(nodes);
        self.2.to_nodes(nodes);
        self.3.to_nodes(nodes);
        self.4.to_nodes(nodes);
        self.5.to_nodes(nodes);
        self.6.to_nodes(nodes);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        let v0 = T0::from_nodes(iter);
        let v1 = T1::from_nodes(iter);
        let v2 = T2::from_nodes(iter);
        let v3 = T3::from_nodes(iter);
        let v4 = T4::from_nodes(iter);
        let v5 = T5::from_nodes(iter);
        let v6 = T6::from_nodes(iter);
        (v0, v1, v2, v3, v4, v5, v6)
    }
}
impl<
        T0: StructOfNodes,
        T1: StructOfNodes,
        T2: StructOfNodes,
        T3: StructOfNodes,
        T4: StructOfNodes,
        T5: StructOfNodes,
        T6: StructOfNodes,
        T7: StructOfNodes,
    > StructOfNodes for (T0, T1, T2, T3, T4, T5, T6, T7)
{
    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {
        self.0.to_nodes(nodes);
        self.1.to_nodes(nodes);
        self.2.to_nodes(nodes);
        self.3.to_nodes(nodes);
        self.4.to_nodes(nodes);
        self.5.to_nodes(nodes);
        self.6.to_nodes(nodes);
        self.7.to_nodes(nodes);
    }
    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {
        let v0 = T0::from_nodes(iter);
        let v1 = T1::from_nodes(iter);
        let v2 = T2::from_nodes(iter);
        let v3 = T3::from_nodes(iter);
        let v4 = T4::from_nodes(iter);
        let v5 = T5::from_nodes(iter);
        let v6 = T6::from_nodes(iter);
        let v7 = T7::from_nodes(iter);
        (v0, v1, v2, v3, v4, v5, v6, v7)
    }
}
