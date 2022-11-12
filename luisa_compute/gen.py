s = 'use crate::lang::*;\n'

for i in range(8):
    types = ['T'+str(k) for k in range(i+1)]
    constrained = ['{}: Aggregate'.format(t) for t in types]
    s += 'impl<{}> Aggregate for ({},) {{\n'.format(', '.join(constrained),
                                                  ',' .join(types))
    s += '    fn to_nodes(&self, nodes: &mut Vec<NodeRef>) {\n'
    for k in range(i+1):
        s += '        self.{}.to_nodes(nodes);\n'.format(k)
    s += '    }\n'
    s += '    fn from_nodes<I: Iterator<Item = NodeRef>>(iter: &mut I) -> Self {\n'
    for k in range(i+1):
        s += '        let v{} = {}::from_nodes(iter);\n'.format(k, types[k])
    s += '        ({},)\n'.format(', '.join(['v'+str(k) for k in range(i+1)]))
    s += '    }\n'
    s += '}\n'
    

with open('src/lang/traits_impl.rs', 'w') as f:
    f.write(s)

# 