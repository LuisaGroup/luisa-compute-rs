from typing import List
from itertools import permutations, product
s = ''
def swizzle_name(perm: List[int]):
    return ''.join('xyzw'[i] for i in perm)

swizzles2 = list(product(range(4), repeat=2))
swizzles3 = list(product(range(4), repeat=3))
swizzles4 = list(product(range(4), repeat=4))
all_swizzles = dict()
all_swizzles[2] = swizzles2
all_swizzles[3] = swizzles3
all_swizzles[4] = swizzles4
# Vec<m> -> Vec<n>
sw_m_to_n = {}
for m in range(2, 5):
    for n in range(2, 5):
        comps = 'xyzw'[:m]
        sw_m_to_n[(m, n)] = [sw for sw in all_swizzles[n] if len(sw) == n and all([s < m for s in sw])]
for n in range(2,5):
    s += 'pub trait Vec{}Swizzle {{\n'.format(n)
    s += '    type Vec2;\n'
    s += '    type Vec3;\n'
    s += '    type Vec4;\n'
    s += '    fn permute2(&self, x: u32, y: u32) -> Self::Vec2;\n'
    s += '    fn permute3(&self, x: u32, y: u32, z: u32) -> Self::Vec3;\n'
    s += '    fn permute4(&self, x: u32, y: u32, z: u32, w: u32) -> Self::Vec4;\n'
    for sw in sw_m_to_n[(n, 2)]:
        s += '    fn {}(&self) -> Self::Vec2 {{\n'.format(swizzle_name(sw))
        s += '        self.permute2({}, {})\n'.format(sw[0], sw[1])
        s += '    }\n'
    for sw in sw_m_to_n[(n, 3)]:
        s += '    fn {}(&self) -> Self::Vec3 {{\n'.format(swizzle_name(sw))
        s += '        self.permute3({}, {}, {})\n'.format(sw[0], sw[1], sw[2])
        s += '    }\n'
    for sw in sw_m_to_n[(n, 4)]:
        s += '    fn {}(&self) -> Self::Vec4 {{\n'.format(swizzle_name(sw))
        s += '        self.permute4({}, {}, {}, {})\n'.format(sw[0], sw[1], sw[2], sw[3])
        s += '    }\n'
    s += '}\n'

with open('swizzle.rs', 'w') as f:
    f.write(s)
