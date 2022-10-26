s = 'use crate::lang::*;\n'

for i in range(8):
    types = ['T'+str(k) for k in range(i+1)]
    constrained = ['{}: StructOfNodes'.format(t) for t in types]
    s += 'impl<{}> StructOfNodes for ({},) {{\n'.format(', '.join(constrained),
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

s = 'use crate::prelude::*;\n'
types = ['float','int','bool','uint']
rust_type = ['f32','i32','bool','u32']
comps = 'xyzw'
for t in types:
    for l in [2,3,4]:
        s += '#[function]\n'
        s += 'pub fn make_{}{}({}) -> {}{} {{\n'.format(t, l,', '.join(['{}: {}'.format(comps[i],rust_type[i]) for i in range(l)]), t, l)
        s += '    {}{} {{ {} }}\n'.format(t, l, ', '.join(comps[:l]),)
        s += '}\n'
with open('src/lang/math_impl.rs', 'w') as f:
    f.write(s)