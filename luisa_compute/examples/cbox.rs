use std::env::current_exe;

#[allow(unused_imports)]
use luisa::prelude::*;
use luisa::Value;
use luisa::{Expr, Float3};
use luisa_compute as luisa;
#[derive(Value, Clone, Copy)]
#[repr(C)]
pub struct Onb {
    tangent: Float3,
    binormal: Float3,
    normal: Float3,
}

impl OnbExpr {
    fn to_world(&self, v: Expr<Float3>) -> Expr<Float3> {
        self.tangent() * v.x() + self.binormal() * v.y() + self.normal() * v.z()
    }
}

#[derive(Value, Clone, Copy)]
#[repr(C)]
pub struct Vertex {
    x: f32,
    y: f32,
    z: f32,
}

const CBOX_OBJ: &'static str = "
# The original Cornell Box in OBJ format.
# Note that the real box is not a perfect cube, so
# the faces are imperfect in this data set.
#
# Created by Guedis Cardenas and Morgan McGuire at Williams College, 2011
# Released into the Public Domain.
#
# http://graphics.cs.williams.edu/data
# http://www.graphics.cornell.edu/online/box/data.html
#

mtllib CornellBox-Original.mtl

g floor
v  -1.01  0.00   0.99
v   1.00  0.00   0.99
v   1.00  0.00  -1.04
v  -0.99  0.00  -1.04
f -4 -3 -2 -1

g ceiling
v  -1.02  1.99   0.99
v  -1.02  1.99  -1.04
v   1.00  1.99  -1.04
v   1.00  1.99   0.99
f -4 -3 -2 -1

g backWall
v  -0.99  0.00  -1.04
v   1.00  0.00  -1.04
v   1.00  1.99  -1.04
v  -1.02  1.99  -1.04
f -4 -3 -2 -1

g rightWall
v	1.00  0.00  -1.04
v	1.00  0.00   0.99
v	1.00  1.99   0.99
v	1.00  1.99  -1.04
f -4 -3 -2 -1

g leftWall
v  -1.01  0.00   0.99
v  -0.99  0.00  -1.04
v  -1.02  1.99  -1.04
v  -1.02  1.99   0.99
f -4 -3 -2 -1

g shortBox

# Top Face
v	0.53  0.60   0.75
v	0.70  0.60   0.17
v	0.13  0.60   0.00
v  -0.05  0.60   0.57
f -4 -3 -2 -1

# Left Face
v  -0.05  0.00   0.57
v  -0.05  0.60   0.57
v   0.13  0.60   0.00
v   0.13  0.00   0.00
f -4 -3 -2 -1

# Front Face
v	0.53  0.00   0.75
v	0.53  0.60   0.75
v  -0.05  0.60   0.57
v  -0.05  0.00   0.57
f -4 -3 -2 -1

# Right Face
v	0.70  0.00   0.17
v	0.70  0.60   0.17
v	0.53  0.60   0.75
v	0.53  0.00   0.75
f -4 -3 -2 -1

# Back Face
v	0.13  0.00   0.00
v	0.13  0.60   0.00
v	0.70  0.60   0.17
v	0.70  0.00   0.17
f -4 -3 -2 -1

# Bottom Face
v	0.53  0.00   0.75
v	0.70  0.00   0.17
v	0.13  0.00   0.00
v  -0.05  0.00   0.57
f -4 -3 -2 -1

g tallBox

# Top Face
v	-0.53  1.20   0.09
v	 0.04  1.20  -0.09
v	-0.14  1.20  -0.67
v	-0.71  1.20  -0.49
f -4 -3 -2 -1

# Left Face
v	-0.53  0.00   0.09
v	-0.53  1.20   0.09
v	-0.71  1.20  -0.49
v	-0.71  0.00  -0.49
f -4 -3 -2 -1

# Back Face
v	-0.71  0.00  -0.49
v	-0.71  1.20  -0.49
v	-0.14  1.20  -0.67
v	-0.14  0.00  -0.67
f -4 -3 -2 -1

# Right Face
v	-0.14  0.00  -0.67
v	-0.14  1.20  -0.67
v	 0.04  1.20  -0.09
v	 0.04  0.00  -0.09
f -4 -3 -2 -1

# Front Face
v	 0.04  0.00  -0.09
v	 0.04  1.20  -0.09
v	-0.53  1.20   0.09
v	-0.53  0.00   0.09
f -4 -3 -2 -1

# Bottom Face
v	-0.53  0.00   0.09
v	 0.04  0.00  -0.09
v	-0.14  0.00  -0.67
v	-0.71  0.00  -0.49
f -4 -3 -2 -1

g light
v	-0.24  1.98   0.16
v	-0.24  1.98  -0.22
v	 0.23  1.98  -0.22
v	 0.23  1.98   0.16
f -4 -3 -2 -1";

fn main() {
    use luisa::*;
    init_logger();

    std::env::set_var("WINIT_UNIX_BACKEND", "x11");
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() <= 2,
        "Usage: {} <backend>. <backend>: cpu, cuda, dx, metal, remote",
        args[0]
    );

    let ctx = Context::new(current_exe().unwrap());
    let device = ctx
        .create_device(if args.len() == 2 {
            args[1].as_str()
        } else {
            "cpu"
        })
        .unwrap();
    let mut buf = std::io::BufReader::new(CBOX_OBJ.as_bytes());
    let (models, _) = tobj::load_obj_buf(
        &mut buf,
        &tobj::LoadOptions {
            triangulate: true,
            ..Default::default()
        },
        |p| tobj::load_mtl(p),
    )
    .unwrap();

    let vertex_heap = device.create_bindless_array(65536).unwrap();
    let index_heap = device.create_bindless_array(65536).unwrap();
    let mut vertex_buffers: Vec<Buffer<Vertex>> = vec![];
    let mut index_buffers: Vec<Buffer<RtxIndex>> = vec![];
    let accel = device.create_accel(AccelOption::default()).unwrap();

    for (index, model) in models.iter().enumerate() {
        let vertex_buffer = device
            .create_buffer_from_slice(unsafe {
                let vertex_ptr = &model.mesh.positions as *const _ as *const f32;
                std::slice::from_raw_parts(
                    vertex_ptr as *const Vertex,
                    model.mesh.positions.len() / 3,
                )
            })
            .unwrap();
        let index_buffer = device
            .create_buffer_from_slice(unsafe {
                let index_ptr = &model.mesh.indices as *const _ as *const u32;
                std::slice::from_raw_parts(
                    index_ptr as *const RtxIndex,
                    model.mesh.indices.len() / 3,
                )
            })
            .unwrap();
        let mesh = device
            .create_mesh(
                vertex_buffer.view(..),
                index_buffer.view(..),
                AccelOption::default(),
            )
            .unwrap();
        vertex_buffers.push(vertex_buffer);
        index_buffers.push(index_buffer);
        vertex_heap.emplace_buffer(index, vertex_buffers.last().unwrap());
        index_heap.emplace_buffer(index, index_buffers.last().unwrap());
        mesh.build(AccelBuildRequest::ForceBuild);
        accel.push_mesh(&mesh, glam::Mat4::IDENTITY.into(), 255, true)
    }
    accel.build(AccelBuildRequest::ForceBuild);
}
