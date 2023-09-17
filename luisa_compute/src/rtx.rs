use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::internal_prelude::*;

use crate::runtime::*;
use crate::{ResourceTracker, *};
use luisa_compute_ir::ir::{
    new_node, AccelBinding, Binding, Func, Instruction, IrBuilder, Node, NodeRef, Type, StructType
};
use parking_lot::RwLock;
use std::ops::Deref;

use luisa_compute_api_types as api;
pub use api::{
    AccelBuildModificationFlags, AccelBuildRequest, AccelOption, AccelUsageHint, MeshType,
    PixelFormat, PixelStorage,
};


pub(crate) struct AccelHandle {
    pub(crate) device: Device,
    pub(crate) handle: api::Accel,
    pub(crate) native_handle: *mut std::ffi::c_void,
}
impl Drop for AccelHandle {
    fn drop(&mut self) {
        self.device.inner.destroy_accel(self.handle);
    }
}
unsafe impl Send for AccelHandle {}
unsafe impl Sync for AccelHandle {}
#[derive(Clone)]
pub(crate) enum InstanceHandle {
    Mesh(Arc<MeshHandle>),
    Procedural(Arc<ProceduralPrimitiveHandle>),
}
impl InstanceHandle {
    pub(crate) fn handle(&self) -> u64 {
        match self {
            InstanceHandle::Mesh(h) => h.handle.0,
            InstanceHandle::Procedural(h) => h.handle.0,
        }
    }
}
pub struct Accel {
    pub(crate) handle: Arc<AccelHandle>,
    pub(crate) instance_handles: RwLock<Vec<Option<InstanceHandle>>>,
    pub(crate) modifications: RwLock<HashMap<usize, api::AccelBuildModification>>,
}
pub(crate) struct ProceduralPrimitiveHandle {
    pub(crate) device: Device,
    pub(crate) handle: api::ProceduralPrimitive,
    pub(crate) native_handle: *mut std::ffi::c_void,
    #[allow(dead_code)]
    pub(crate) aabb_buffer: Arc<BufferHandle>,
}
impl Drop for ProceduralPrimitiveHandle {
    fn drop(&mut self) {
        self.device.inner.destroy_procedural_primitive(self.handle);
    }
}
unsafe impl Send for ProceduralPrimitiveHandle {}
unsafe impl Sync for ProceduralPrimitiveHandle {}
pub struct ProceduralPrimitive {
    pub(crate) handle: Arc<ProceduralPrimitiveHandle>,
    pub(crate) aabb_buffer: api::Buffer,
    pub(crate) aabb_buffer_offset: usize,
    pub(crate) aabb_buffer_count: usize,
}

impl ProceduralPrimitive {
    pub fn native_handle(&self) -> *mut std::ffi::c_void {
        self.handle.native_handle
    }
    pub fn build_async<'a>(&self, request: AccelBuildRequest) -> Command<'a> {
        let mut rt = ResourceTracker::new();
        rt.add(self.handle.clone());
        Command {
            inner: api::Command::ProceduralPrimitiveBuild(api::ProceduralPrimitiveBuildCommand {
                handle: self.handle.handle,
                request,
                aabb_buffer: self.aabb_buffer,
                aabb_buffer_offset: self.aabb_buffer_offset,
                aabb_count: self.aabb_buffer_count,
            }),
            marker: PhantomData,
            resource_tracker: rt,
            callback: None,
        }
    }
    pub fn build(&self, request: AccelBuildRequest) {
        submit_default_stream_and_sync(&self.handle.device, [self.build_async(request)]);
    }
}
pub(crate) struct MeshHandle {
    pub(crate) device: Device,
    pub(crate) handle: api::Mesh,
    pub(crate) native_handle: *mut std::ffi::c_void,
    #[allow(dead_code)]
    pub(crate) vbuffer: Arc<BufferHandle>,
    #[allow(dead_code)]
    pub(crate) ibuffer: Arc<BufferHandle>,
}
impl Drop for MeshHandle {
    fn drop(&mut self) {
        self.device.inner.destroy_mesh(self.handle);
    }
}
unsafe impl Send for MeshHandle {}
unsafe impl Sync for MeshHandle {}
pub struct Mesh {
    pub(crate) handle: Arc<MeshHandle>,
    pub(crate) vertex_buffer: api::Buffer,
    pub(crate) vertex_buffer_offset: usize,
    pub(crate) vertex_buffer_size: usize,
    pub(crate) vertex_stride: usize,
    pub(crate) index_buffer: api::Buffer,
    pub(crate) index_buffer_offset: usize,
    pub(crate) index_buffer_size: usize,
    pub(crate) index_stride: usize,
}
impl Mesh {
    pub fn native_handle(&self) -> *mut std::ffi::c_void {
        self.handle.native_handle
    }
    pub fn build_async<'a>(&self, request: AccelBuildRequest) -> Command<'a> {
        let mut rt = ResourceTracker::new();
        rt.add(self.handle.clone());
        Command {
            inner: api::Command::MeshBuild(api::MeshBuildCommand {
                mesh: self.handle.handle,
                request,
                vertex_buffer: self.vertex_buffer,
                vertex_buffer_offset: self.vertex_buffer_offset,
                vertex_buffer_size: self.vertex_buffer_size,
                vertex_stride: self.vertex_stride,
                index_buffer: self.index_buffer,
                index_buffer_offset: self.index_buffer_offset,
                index_buffer_size: self.index_buffer_size,
                index_stride: self.index_stride,
            }),
            marker: PhantomData,
            resource_tracker: rt,
            callback: None,
        }
    }
    pub fn build(&self, request: AccelBuildRequest) {
        submit_default_stream_and_sync(&self.handle.device, [self.build_async(request)]);
    }
}

impl Accel {
    fn push_handle(
        &self,
        handle: InstanceHandle,
        transform: Mat4,
        ray_mask: u32,
        opaque: bool,
        user_id: u32,
    ) {
        let mut flags =
            api::AccelBuildModificationFlags::PRIMITIVE | AccelBuildModificationFlags::TRANSFORM;

        flags |= api::AccelBuildModificationFlags::VISIBILITY
            | api::AccelBuildModificationFlags::USER_ID;

        if opaque {
            flags |= api::AccelBuildModificationFlags::OPAQUE_ON;
        } else {
            flags |= api::AccelBuildModificationFlags::OPAQUE_OFF;
        }
        let mut modifications = self.modifications.write();
        let mut instance_handles = self.instance_handles.write();
        let index = modifications.len() as u32;
        modifications.insert(
            instance_handles.len(),
            api::AccelBuildModification {
                mesh: handle.handle(),
                affine: transform.into_affine3x4(),
                flags,
                visibility: ray_mask,
                index,
                user_id,
            },
        );

        instance_handles.push(Some(handle));
    }
    fn set_handle(
        &self,
        index: usize,
        handle: InstanceHandle,
        transform: Mat4,
        ray_mask: u32,
        opaque: bool,
        user_id: u32,
    ) {
        let mut flags = api::AccelBuildModificationFlags::PRIMITIVE;
        dbg!(flags);
        flags |= api::AccelBuildModificationFlags::VISIBILITY
            | api::AccelBuildModificationFlags::USER_ID;

        if opaque {
            flags |= api::AccelBuildModificationFlags::OPAQUE_ON;
        } else {
            flags |= api::AccelBuildModificationFlags::OPAQUE_OFF;
        }
        let mut modifications = self.modifications.write();
        modifications.insert(
            index as usize,
            api::AccelBuildModification {
                mesh: handle.handle(),
                affine: transform.into_affine3x4(),
                flags,
                visibility: ray_mask,
                index: index as u32,
                user_id,
            },
        );
        let mut instance_handles = self.instance_handles.write();
        instance_handles[index] = Some(handle);
    }
    pub fn push_mesh(&self, mesh: &Mesh, transform: Mat4, ray_mask: u32, opaque: bool) {
        self.push_handle(
            InstanceHandle::Mesh(mesh.handle.clone()),
            transform,
            ray_mask,
            opaque,
            0,
        )
    }
    pub fn push_procedural_primitive(
        &self,
        prim: &ProceduralPrimitive,
        transform: Mat4,
        ray_mask: u32,
    ) {
        self.push_handle(
            InstanceHandle::Procedural(prim.handle.clone()),
            transform,
            ray_mask,
            false,
            0,
        )
    }
    pub fn set_mesh(
        &self,
        index: usize,
        mesh: &Mesh,
        transform: Mat4,
        ray_mask: u32,
        opaque: bool,
    ) {
        self.set_handle(
            index,
            InstanceHandle::Mesh(mesh.handle.clone()),
            transform,
            ray_mask,
            opaque,
            0,
        )
    }
    pub fn set_procedural_primitive(
        &self,
        index: usize,
        prim: &ProceduralPrimitive,
        transform: Mat4,
        ray_mask: u32,
    ) {
        self.set_handle(
            index,
            InstanceHandle::Procedural(prim.handle.clone()),
            transform,
            ray_mask,
            false,
            0,
        )
    }
    pub fn pop(&self) {
        let mut modifications = self.modifications.write();
        let mut instance_handles = self.instance_handles.write();
        let n = instance_handles.len();
        modifications.remove(&n);
        instance_handles.pop().unwrap();
    }
    pub fn build(&self, request: api::AccelBuildRequest) {
        submit_default_stream_and_sync(&self.handle.device, [self.build_async(request)])
    }
    pub fn build_async<'a>(&'a self, request: api::AccelBuildRequest) -> Command<'a> {
        let mut rt = ResourceTracker::new();
        let instance_handles = self.instance_handles.read();
        rt.add(self.handle.clone());
        let mut modifications = self.modifications.write();
        let m = modifications.drain().map(|(_, v)| v).collect::<Vec<_>>();
        let m = Arc::new(m);
        rt.add(m.clone());
        Command {
            marker: PhantomData,
            inner: api::Command::AccelBuild(api::AccelBuildCommand {
                accel: self.handle.handle,
                request,
                instance_count: instance_handles.len() as u32,
                modifications: m.as_ptr(),
                modifications_count: m.len(),
                update_instance_buffer_only: false,
            }),
            resource_tracker: rt,
            callback: None,
        }
    }
    pub fn var(&self) -> AccelVar {
        AccelVar::new(self)
    }
    pub(crate) fn handle(&self) -> api::Accel {
        self.handle.handle
    }
    pub fn native_handle(&self) -> *mut std::ffi::c_void {
        self.handle.native_handle
    }
}
#[derive(Clone)]
pub struct AccelVar {
    pub(crate) node: NodeRef,
    #[allow(dead_code)]
    pub(crate) handle: Option<Arc<AccelHandle>>,
}

#[repr(C)]
#[repr(align(16))]
#[derive(Clone, Copy, Value, Debug)]
pub struct Ray {
    pub orig: PackedFloat3,
    pub tmin: f32,
    pub dir: PackedFloat3,
    pub tmax: f32,
}
#[repr(C)]
#[derive(Clone, Copy, Value, Debug)]
pub struct Aabb {
    pub min: PackedFloat3,
    pub max: PackedFloat3,
}

#[repr(C)]
#[derive(Clone, Copy, Value, Debug)]
pub struct TriangleHit {
    pub inst: u32,
    pub prim: u32,
    pub bary: Float2,
    pub committed_ray_t: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Value, Debug)]
pub struct ProceduralHit {
    pub inst: u32,
    pub prim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Value, Debug)]
pub struct CommittedHit {
    pub inst_id: u32,
    pub prim_id: u32,
    pub bary: Float2,
    pub hit_type: u32,
    pub committed_ray_t: f32,
}
impl CommittedHitExpr {
    pub fn miss(&self) -> Expr<bool> {
        self.hit_type().cmpeq(HitType::Miss as u32)
    }
    pub fn triangle_hit(&self) -> Expr<bool> {
        self.hit_type().cmpeq(HitType::Triangle as u32)
    }
    pub fn procedural_hit(&self) -> Expr<bool> {
        self.hit_type().cmpeq(HitType::Procedural as u32)
    }
}
#[derive(Clone, Copy)]
#[repr(u32)]
pub enum HitType {
    Miss = 0,
    Triangle = 1,
    Procedural = 2,
}

pub fn offset_ray_origin(p: Expr<Float3>, n: Expr<Float3>) -> Expr<Float3> {
    lazy_static! {
        static ref F: Callable<fn(Expr<Float3>, Expr<Float3>) -> Expr<Float3>> =
            create_static_callable::<fn(Expr<Float3>, Expr<Float3>) -> Expr<Float3>>(|p, n| {
                const ORIGIN: f32 = 1.0f32 / 32.0f32;
                const FLOAT_SCALE: f32 = 1.0f32 / 65536.0f32;
                const INT_SCALE: f32 = 256.0f32;
                let of_i = (INT_SCALE * n).int();
                let p_i = p.bitcast::<Int3>() + Int3Expr::select(p.cmplt(0.0f32), -of_i, of_i);
                Float3Expr::select(
                    p.abs().cmplt(ORIGIN),
                    p + FLOAT_SCALE * n,
                    p_i.bitcast::<Float3>(),
                )
            });
    }
    F.call(p, n)
}
pub type Index = PackedUint3;

#[repr(C)]
#[repr(align(8))]
#[derive(Clone, Copy, Value, Debug)]
pub struct Hit {
    pub inst_id: u32,
    pub prim_id: u32,
    pub u: f32,
    pub v: f32,
    pub t: f32,
}

#[cfg(test)]
mod test {
    #[test]
    fn rtx_layout() {
        use super::*;
        assert_eq!(std::mem::align_of::<Ray>(), 16);
        assert_eq!(std::mem::size_of::<Ray>(), 32);
        assert_eq!(std::mem::size_of::<Hit>(), 24);
        assert_eq!(std::mem::align_of::<Hit>(), 8);
        assert_eq!(std::mem::size_of::<Index>(), 12);
    }
}

impl HitExpr {
    pub fn valid(&self) -> Expr<bool> {
        self.inst_id().cmpne(u32::MAX)
    }
    pub fn miss(&self) -> Expr<bool> {
        self.inst_id().cmpeq(u32::MAX)
    }
}

#[derive(Clone, Copy)]
pub struct TriangleCandidate {
    query: NodeRef,
    hit: TriangleHitExpr,
}
#[derive(Clone, Copy)]
pub struct ProceduralCandidate {
    query: NodeRef,
    hit: ProceduralHitExpr,
}
impl TriangleCandidate {
    pub fn commit(&self) {
        __current_scope(|b| b.call(Func::RayQueryCommitTriangle, &[self.query], Type::void()));
    }
    pub fn terminate(&self) {
        __current_scope(|b| b.call(Func::RayQueryTerminate, &[self.query], Type::void()));
    }
    pub fn ray(&self) -> Expr<Ray> {
        Expr::<Ray>::from_node(__current_scope(|b| {
            b.call(
                Func::RayQueryWorldSpaceRay,
                &[self.query],
                rtx::Ray::type_(),
            )
        }))
    }
}
impl Deref for TriangleCandidate {
    type Target = TriangleHitExpr;
    fn deref(&self) -> &Self::Target {
        &self.hit
    }
}
impl ProceduralCandidate {
    pub fn commit(&self, t: Expr<f32>) {
        __current_scope(|b| {
            b.call(
                Func::RayQueryCommitProcedural,
                &[self.query, t.node()],
                Type::void(),
            )
        });
    }
    pub fn terminate(&self) {
        __current_scope(|b| b.call(Func::RayQueryTerminate, &[self.query], Type::void()));
    }
    pub fn ray(&self) -> Expr<Ray> {
        Expr::<Ray>::from_node(__current_scope(|b| {
            b.call(
                Func::RayQueryWorldSpaceRay,
                &[self.query],
                rtx::Ray::type_(),
            )
        }))
    }
}
impl Deref for ProceduralCandidate {
    type Target = ProceduralHitExpr;
    fn deref(&self) -> &Self::Target {
        &self.hit
    }
}

pub struct RayQuery<T, P> {
    pub on_triangle_hit: T,
    pub on_procedural_hit: P,
}
impl AccelVar {
    #[inline]
    pub fn instance_transform(&self, index: Expr<u32>) -> Expr<Mat4> {
        FromNode::from_node(__current_scope(|b| {
            b.call(
                Func::RayTracingInstanceTransform,
                &[self.node, index.node()],
                Mat4::type_(),
            )
        }))
    }

    #[inline]
    pub fn trace_closest_masked(
        &self,
        ray: impl Into<Expr<Ray>>,
        mask: impl Into<Expr<u32>>,
    ) -> Expr<Hit> {
        let ray = ray.into();
        let mask = mask.into();
        FromNode::from_node(__current_scope(|b| {
            b.call(
                Func::RayTracingTraceClosest,
                &[self.node, ray.node(), mask.node()],
                Hit::type_(),
            )
        }))
    }
    #[inline]
    pub fn trace_any_masked(
        &self,
        ray: impl Into<Expr<Ray>>,
        mask: impl Into<Expr<u32>>,
    ) -> Expr<bool> {
        let ray = ray.into();
        let mask = mask.into();
        FromNode::from_node(__current_scope(|b| {
            b.call(
                Func::RayTracingTraceAny,
                &[self.node, ray.node(), mask.node()],
                bool::type_(),
            )
        }))
    }
    #[inline]
    pub fn trace_closest(&self, ray: impl Into<Expr<Ray>>) -> Expr<Hit> {
        self.trace_closest_masked(ray, 0xff)
    }
    #[inline]
    pub fn trace_any(&self, ray: impl Into<Expr<Ray>>) -> Expr<bool> {
        self.trace_any_masked(ray, 0xff)
    }
    #[inline]
    pub fn query_all<T, P>(
        &self,

        ray: impl Into<Expr<Ray>>,
        mask: impl Into<Expr<u32>>,
        ray_query: RayQuery<T, P>,
    ) -> Expr<CommittedHit>
    where
        T: FnOnce(TriangleCandidate),
        P: FnOnce(ProceduralCandidate),
    {
        self._query(false, ray, mask, ray_query)
    }
    #[inline]
    pub fn query_any<T, P>(
        &self,
        ray: impl Into<Expr<Ray>>,
        mask: impl Into<Expr<u32>>,
        ray_query: RayQuery<T, P>,
    ) -> Expr<CommittedHit>
    where
        T: FnOnce(TriangleCandidate),
        P: FnOnce(ProceduralCandidate),
    {
        self._query(true, ray, mask, ray_query)
    }
    #[inline]
    fn _query<T, P>(
        &self,
        terminate_on_first: bool,
        ray: impl Into<Expr<Ray>>,
        mask: impl Into<Expr<u32>>,
        ray_query: RayQuery<T, P>,
    ) -> Expr<CommittedHit>
    where
        T: FnOnce(TriangleCandidate),
        P: FnOnce(ProceduralCandidate),
    {
        let ray = ray.into();
        let mask = mask.into();
        let query = __current_scope(|b| {
            b.call(
                if terminate_on_first {
                    Func::RayTracingQueryAny
                } else {
                    Func::RayTracingQueryAll
                },
                &[self.node, ray.node(), mask.node()],
                Type::opaque(
                    if terminate_on_first {
                        "LC_RayQueryAny"
                    } else {
                        "LC_RayQueryAll"
                    }
                    .into(),
                ),
            )
        });

        RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            let pools = r.pools.clone().unwrap();
            let s = &mut r.scopes;
            s.push(IrBuilder::new(pools));
        });
        let triangle_candidate = __current_scope(|b| TriangleCandidate {
            query,
            hit: FromNode::from_node(b.call(
                Func::RayQueryTriangleCandidateHit,
                &[query],
                TriangleHit::type_(),
            )),
        });
        (ray_query.on_triangle_hit)(triangle_candidate);
        let on_triangle_hit = __pop_scope();
        RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            let pools = r.pools.clone().unwrap();
            let s = &mut r.scopes;
            s.push(IrBuilder::new(pools));
        });
        let procedural_candidate = __current_scope(|b| ProceduralCandidate {
            query,
            hit: FromNode::from_node(b.call(
                Func::RayQueryProceduralCandidateHit,
                &[query],
                ProceduralHit::type_(),
            )),
        });
        (ray_query.on_procedural_hit)(procedural_candidate);
        let on_procedural_hit = __pop_scope();
        __current_scope(|b| {
            b.ray_query(query, on_triangle_hit, on_procedural_hit, Type::void());
            FromNode::from_node(b.call(Func::RayQueryCommittedHit, &[query], CommittedHit::type_()))
        })
    }
    pub fn new(accel: &rtx::Accel) -> Self {
        let node = RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            assert!(r.lock, "BufferVar must be created from within a kernel");
            let handle: u64 = accel.handle().0;
            let binding = Binding::Accel(AccelBinding { handle });
            if let Some((_, node, _, _)) = r.captured_buffer.get(&binding) {
                *node
            } else {
                let node = new_node(
                    r.pools.as_ref().unwrap(),
                    Node::new(CArc::new(Instruction::Accel), Type::void()),
                );
                let i = r.captured_buffer.len();
                r.captured_buffer
                    .insert(binding, (i, node, binding, accel.handle.clone()));
                node
            }
        });
        Self {
            node,
            handle: Some(accel.handle.clone()),
        }
    }
}
