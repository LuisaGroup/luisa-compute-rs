from typing import List
from itertools import permutations, product

prims = ['f32', 'i32', 'u32', 'f64', 'i64', 'u64', 'f16', 'i16', 'u16', 'i8', 'u8']
file = open('cast_impls.rs', 'w')
print('\n#[rustfmt::skip]mod impl_{\nuse crate::prelude::*;\nuse super::super::*;\n', file=file)
v_name = {
    'f32':'float',
    'i32':'int',
    'u32':'uint',
    'f64':'double',
    'i64':'long',
    'u64':'ulong',
    'f16':'half',
    'i16':'short',
    'u16':'ushort',
    'i8':'byte',
    'u8':'ubyte'
}
def make_typename(t):
    t = list(t)
    t[0] = t[0].upper() 
    return ''.join(t)
for p in prims:
    print('impl Expr<{}> {{'.format(p), file=file)
    for q in prims:
        if p != q:
            print('    pub fn as_{0}(self) -> Expr<{0}> {{ self.as_::<{0}>() }}'.format(q), file=file)
    print('}', file=file)
    for n in [2,3,4]:
        print('impl Expr<{}{}> {{'.format(make_typename(v_name[p]),n), file=file)
        for q in prims:
            if p != q:
                print('    pub fn as_{2}{1}(self) -> Expr<{0}{1}> {{ self.as_::<{0}{1}>() }}'.format(make_typename(v_name[q]),n, v_name[q]), file=file)
        print('}', file=file)
print('}', file=file)