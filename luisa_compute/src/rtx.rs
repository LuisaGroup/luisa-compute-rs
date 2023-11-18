use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::internal_prelude::*;

use crate::runtime::*;
use crate::{ResourceTracker, *};
use luisa_compute_ir::ir::CurveBasisSet;
use luisa_compute_ir::ir::{AccelBinding, Binding, Func, Instruction, IrBuilder, Node, Type};
use parking_lot::RwLock;
use std::ops::Deref;
pub mod curve;
pub use api::{
    AccelBuildModificationFlags, AccelBuildRequest, AccelOption, AccelUsageHint, MeshType,
    PixelFormat, PixelStorage,
};
pub use curve::*;
use luisa_compute_api_types as api;

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
    pub fn build_async(&self, request: AccelBuildRequest) -> Command<'static, 'static> {
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
    pub fn build_async(&self, request: AccelBuildRequest) -> Command<'static, 'static> {
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
    pub fn build_async(&self, request: api::AccelBuildRequest) -> Command<'static, 'static> {
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
    pub(crate) node: SafeNodeRef,
    #[allow(dead_code)]
    pub(crate) handle: Option<Arc<AccelHandle>>,
}

#[repr(C)]
#[repr(align(16))]
#[derive(Clone, Copy, Value, Debug, Soa)]
#[value_new(pub)]
pub struct Ray {
    pub orig: [f32; 3],
    pub tmin: f32,
    pub dir: [f32; 3],
    pub tmax: f32,
}
#[repr(C)]
#[derive(Clone, Copy, Value, Debug, Soa)]
#[value_new(pub)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

/// Represents a hit on a triangle or a curve
/// Use [`Expr<SurfaceHit>::is_curve`] to check if it's a curve or a triangle
///
/// Use [`Expr<SurfaceHit>::curve_parameter`] or [`Expr<SurfaceHit>::triangle_barycentric_coord`] to access the hit data
///
/// For triangles, use [`TriangleInterpolate::interpolate`] to interpolate vertex attributes
///
/// For curves, see [`CurveEvaluator`] trait and its impls [`PiecewiseLinearCurve`],
///  [`BezierCurve`], [`CubicBSplineCurve`], [`CatmullRomCurve`] for more information
#[repr(C)]
#[derive(Clone, Copy, Value, Debug, Soa)]
pub struct SurfaceHit {
    pub inst: u32,
    pub prim: u32,
    /// Don't use directly
    ///
    /// use [`SurfaceHitExpr::triangle_barycentric_coord`] and [`SurfaceHitExpr::curve_parameter`] to access
    pub bary: Float2,
    pub committed_ray_t: f32,
}

pub trait TriangleInterpolate<V: Value> {
    fn interpolate(
        &self,
        v0: impl AsExpr<Value = V>,
        v1: impl AsExpr<Value = V>,
        v2: impl AsExpr<Value = V>,
    ) -> Expr<V>;
}
impl TriangleInterpolate<f32> for Expr<Float2> {
    #[tracked]
    fn interpolate(
        &self,
        v0: impl AsExpr<Value = f32>,
        v1: impl AsExpr<Value = f32>,
        v2: impl AsExpr<Value = f32>,
    ) -> Expr<f32> {
        (1.0 - self.x - self.y) * v0.as_expr() + self.x * v1.as_expr() + self.y * v2.as_expr()
    }
}
impl TriangleInterpolate<Float2> for Expr<Float2> {
    #[tracked]
    fn interpolate(
        &self,
        v0: impl AsExpr<Value = Float2>,
        v1: impl AsExpr<Value = Float2>,
        v2: impl AsExpr<Value = Float2>,
    ) -> Expr<Float2> {
        (1.0 - self.x - self.y) * v0.as_expr() + self.x * v1.as_expr() + self.y * v2.as_expr()
    }
}
impl TriangleInterpolate<Float3> for Expr<Float2> {
    #[tracked]
    fn interpolate(
        &self,
        v0: impl AsExpr<Value = Float3>,
        v1: impl AsExpr<Value = Float3>,
        v2: impl AsExpr<Value = Float3>,
    ) -> Expr<Float3> {
        (1.0 - self.x - self.y) * v0.as_expr() + self.x * v1.as_expr() + self.y * v2.as_expr()
    }
}
impl TriangleInterpolate<Float4> for Expr<Float2> {
    #[tracked]
    fn interpolate(
        &self,
        v0: impl AsExpr<Value = Float4>,
        v1: impl AsExpr<Value = Float4>,
        v2: impl AsExpr<Value = Float4>,
    ) -> Expr<Float4> {
        (1.0 - self.x - self.y) * v0.as_expr() + self.x * v1.as_expr() + self.y * v2.as_expr()
    }
}
macro_rules! impl_triangle_interp {
    ($t:ty) => {
        impl TriangleInterpolate<f32> for Expr<$t> {
            #[tracked]
            fn interpolate(
                &self,
                v0: impl AsExpr<Value = f32>,
                v1: impl AsExpr<Value = f32>,
                v2: impl AsExpr<Value = f32>,
            ) -> Expr<f32> {
                self.bary.interpolate(v0, v1, v2)
            }
        }
        impl TriangleInterpolate<Float2> for Expr<$t> {
            #[tracked]
            fn interpolate(
                &self,
                v0: impl AsExpr<Value = Float2>,
                v1: impl AsExpr<Value = Float2>,
                v2: impl AsExpr<Value = Float2>,
            ) -> Expr<Float2> {
                self.bary.interpolate(v0, v1, v2)
            }
        }
        impl TriangleInterpolate<Float3> for Expr<$t> {
            #[tracked]
            fn interpolate(
                &self,
                v0: impl AsExpr<Value = Float3>,
                v1: impl AsExpr<Value = Float3>,
                v2: impl AsExpr<Value = Float3>,
            ) -> Expr<Float3> {
                self.bary.interpolate(v0, v1, v2)
            }
        }
        impl TriangleInterpolate<Float4> for Expr<$t> {
            #[tracked]
            fn interpolate(
                &self,
                v0: impl AsExpr<Value = Float4>,
                v1: impl AsExpr<Value = Float4>,
                v2: impl AsExpr<Value = Float4>,
            ) -> Expr<Float4> {
                self.bary.interpolate(v0, v1, v2)
            }
        }
    };
}
impl_triangle_interp!(SurfaceHit);
impl_triangle_interp!(CommittedHit);
#[repr(C)]
#[derive(Clone, Copy, Value, Debug, Soa)]
pub struct ProceduralHit {
    pub inst: u32,
    pub prim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Value, Debug, Soa)]
pub struct CommittedHit {
    pub inst: u32,
    pub prim: u32,
    pub bary: Float2,
    pub hit_type: u32,
    pub committed_ray_t: f32,
}
impl CommittedHitExpr {
    pub fn miss(&self) -> Expr<bool> {
        self.hit_type.eq(HitType::Miss as u32)
    }
    #[tracked]
    pub fn triangle_hit(&self) -> Expr<bool> {
        self.hit_type.eq(HitType::Surface as u32) & self.bary.y.ge(0.0)
    }
    #[tracked]
    pub fn procedural_hit(&self) -> Expr<bool> {
        self.hit_type.eq(HitType::Procedural as u32) & self.bary.y.lt(0.0)
    }
    pub fn curve_parameter(&self) -> Expr<f32> {
        self.bary.x
    }
    pub fn triangle_barycentric_coord(&self) -> Expr<Float2> {
        self.bary
    }
}

#[derive(Clone, Copy)]
#[repr(u32)]
pub enum HitType {
    Miss = 0,
    Surface = 1,
    Procedural = 2,
}
#[tracked]
pub fn offset_ray_origin(
    p: impl AsExpr<Value = Float3>,
    n: impl AsExpr<Value = Float3>,
) -> Expr<Float3> {
    let ret = Var::<Float3>::zeroed();
    outline(|| {
        let p: Expr<Float3> = p.as_expr();
        let n: Expr<Float3> = n.as_expr();
        let origin: f32 = 1.0f32 / 32.0f32;
        let float_scale: f32 = 1.0f32 / 65536.0f32;
        let int_scale: f32 = 256.0f32;

        let of_i = (int_scale * n).as_int3();
        let p_i = p.bitcast::<Int3>() + p.lt(0.0f32).select(-of_i, of_i);
        *ret = (p.abs() < origin).select(p + float_scale * n, p_i.bitcast::<Float3>())
    });
    ret.load()
}
pub type Index = [u32; 3];

#[repr(C)]
#[repr(align(8))]
#[derive(Clone, Copy, Value, Debug, Soa)]
#[deprecated(note = "Use `SurfaceHit` instead")]
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
        assert_eq!(std::mem::size_of::<SurfaceHit>(), 24);
        assert_eq!(std::mem::align_of::<SurfaceHit>(), 8);
        assert_eq!(std::mem::size_of::<Index>(), 12);
    }
}
#[deprecated]
impl HitExpr {
    pub fn valid(&self) -> Expr<bool> {
        self.inst_id.ne(u32::MAX)
    }
    pub fn miss(&self) -> Expr<bool> {
        self.inst_id.eq(u32::MAX)
    }
}

impl SurfaceHitExpr {
    pub fn valid(&self) -> Expr<bool> {
        self.inst.ne(u32::MAX)
    }
    pub fn miss(&self) -> Expr<bool> {
        self.prim.eq(u32::MAX)
    }
    pub fn is_curve(&self) -> Expr<bool> {
        self.bary.y.lt(0.0)
    }
    pub fn is_triangle(&self) -> Expr<bool> {
        !self.is_curve()
    }
    pub fn curve_parameter(&self) -> Expr<f32> {
        self.bary.x
    }
    pub fn triangle_barycentric_coord(&self) -> Expr<Float2> {
        self.bary
    }
}

#[derive(Clone, Copy)]
pub struct SurfaceCandidate {
    query: SafeNodeRef,
    hit: Expr<SurfaceHit>,
}
pub type TriangleHit = SurfaceHit;
#[deprecated(note = "Use `SurfaceCandidate` instead")]
pub type TriangleCandidate = SurfaceCandidate;

#[derive(Clone, Copy)]
pub struct ProceduralCandidate {
    query: SafeNodeRef,
    hit: Expr<ProceduralHit>,
}
impl SurfaceCandidate {
    pub fn commit(&self) {
        let query = self.query.get();
        __current_scope(|b| b.call(Func::RayQueryCommitTriangle, &[query], Type::void()));
    }
    pub fn terminate(&self) {
        let query = self.query.get();
        __current_scope(|b| b.call(Func::RayQueryTerminate, &[query], Type::void()));
    }
    pub fn ray(&self) -> Expr<Ray> {
        let query = self.query.get();
        Expr::<Ray>::from_node(
            __current_scope(|b| b.call(Func::RayQueryWorldSpaceRay, &[query], rtx::Ray::type_()))
                .into(),
        )
    }
}
impl Deref for SurfaceCandidate {
    type Target = SurfaceHitExpr;
    fn deref(&self) -> &Self::Target {
        &self.hit
    }
}
impl ProceduralCandidate {
    pub fn commit(&self, t: impl AsExpr<Value = f32>) {
        let t = t.as_expr().node().get();
        let query = self.query.get();
        __current_scope(|b| b.call(Func::RayQueryCommitProcedural, &[query, t], Type::void()));
    }
    pub fn terminate(&self) {
        let query = self.query.get();
        __current_scope(|b| b.call(Func::RayQueryTerminate, &[query], Type::void()));
    }
    pub fn ray(&self) -> Expr<Ray> {
        let query = self.query.get();
        Expr::<Ray>::from_node(
            __current_scope(|b| b.call(Func::RayQueryWorldSpaceRay, &[query], rtx::Ray::type_()))
                .into(),
        )
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
#[derive(Clone, Copy)]
pub struct AccelTraceOptions {
    pub curve_bases: CurveBasisSet,
    pub mask: Expr<u32>,
}
impl Default for AccelTraceOptions {
    fn default() -> Self {
        Self {
            curve_bases: CurveBasisSet::empty(),
            mask: u32::MAX.expr(),
        }
    }
}
pub struct RayQueryBase<const TERMINATE_ON_FIRST: bool> {
    query: SafeNodeRef,
    on_surface_hit: Option<Pooled<BasicBlock>>,
    on_procedural_hit: Option<Pooled<BasicBlock>>,
}
impl<const TERMINATE_ON_FIRST: bool> RayQueryBase<TERMINATE_ON_FIRST> {
    pub fn on_surface_hit(self, f: impl Fn(SurfaceCandidate)) -> Self {
        assert!(
            self.on_surface_hit.is_none(),
            "Surface hit already recorded"
        );
        with_recorder(|r| {
            let pools = r.pools.clone();
            let s = &mut r.scopes;
            s.push(IrBuilder::new(pools));
        });
        let query = self.query.get();
        let candidate = SurfaceCandidate {
            query: self.query,
            hit: FromNode::from_node(
                __current_scope(|b| {
                    b.call(
                        Func::RayQueryTriangleCandidateHit,
                        &[query],
                        TriangleHit::type_(),
                    )
                })
                .into(),
            ),
        };
        (f)(candidate);
        let on_surface_hit = __pop_scope();
        Self {
            on_surface_hit: Some(on_surface_hit),
            ..self
        }
    }
    pub fn on_procedural_hit(self, f: impl Fn(ProceduralCandidate)) -> Self {
        assert!(
            self.on_procedural_hit.is_none(),
            "Procedural hit already recorded"
        );
        with_recorder(|r| {
            let pools = r.pools.clone();
            let s = &mut r.scopes;
            s.push(IrBuilder::new(pools));
        });
        let query = self.query.get();
        let procedural_candidate = ProceduralCandidate {
            query: self.query,
            hit: FromNode::from_node(
                __current_scope(|b| {
                    b.call(
                        Func::RayQueryProceduralCandidateHit,
                        &[query],
                        ProceduralHit::type_(),
                    )
                })
                .into(),
            ),
        };
        (f)(procedural_candidate);
        let on_procedural_hit = __pop_scope();
        Self {
            on_procedural_hit: Some(on_procedural_hit),
            ..self
        }
    }
    pub fn trace(self) -> Expr<CommittedHit> {
        let query = self.query.get();
        let on_surface_hit = self
            .on_surface_hit
            .unwrap_or_else(|| IrBuilder::new(__module_pools().clone()).finish());
        let on_procedural_hit = self
            .on_procedural_hit
            .unwrap_or_else(|| IrBuilder::new(__module_pools().clone()).finish());
        FromNode::from_node(
            __current_scope(|b| {
                b.ray_query(query, on_surface_hit, on_procedural_hit, Type::void());
                b.call(Func::RayQueryCommittedHit, &[query], CommittedHit::type_())
            })
            .into(),
        )
    }
}

impl AccelVar {
    pub fn instance_transform(&self, index: Expr<u32>) -> Expr<Mat4> {
        let index = index.node().get();
        let self_node = self.node.get();
        FromNode::from_node(
            __current_scope(|b| {
                b.call(
                    Func::RayTracingInstanceTransform,
                    &[self_node, index],
                    Mat4::type_(),
                )
            })
            .into(),
        )
    }

    pub fn intersect(
        &self,
        ray: impl AsExpr<Value = Ray>,
        options: AccelTraceOptions,
    ) -> Expr<SurfaceHit> {
        let ray = ray.as_expr().node().get();
        let mask = options.mask.node().get();
        let self_node = self.node.get();
        with_recorder(|r| {
            r.add_required_curve_basis(options.curve_bases);
        });
        FromNode::from_node(
            __current_scope(|b| {
                b.call(
                    Func::RayTracingTraceClosest,
                    &[self_node, ray, mask],
                    SurfaceHit::type_(),
                )
            })
            .into(),
        )
    }
    pub fn intersect_any(
        &self,
        ray: impl AsExpr<Value = Ray>,
        options: AccelTraceOptions,
    ) -> Expr<bool> {
        let ray = ray.as_expr().node().get();
        let mask = options.mask.node().get();
        let self_node = self.node.get();
        with_recorder(|r| {
            r.add_required_curve_basis(options.curve_bases);
        });
        FromNode::from_node(
            __current_scope(|b| {
                b.call(
                    Func::RayTracingTraceAny,
                    &[self_node, ray, mask],
                    bool::type_(),
                )
            })
            .into(),
        )
    }

    #[deprecated(note = "Use `intersect` instead")]
    #[allow(deprecated)]
    pub fn trace_closest_masked(
        &self,
        ray: impl AsExpr<Value = Ray>,
        mask: impl AsExpr<Value = u32>,
    ) -> Expr<Hit> {
        let ray = ray.as_expr().node().get();
        let mask = mask.as_expr().node().get();
        let self_node = self.node.get();
        FromNode::from_node(
            __current_scope(|b| {
                b.call(
                    Func::RayTracingTraceClosest,
                    &[self_node, ray, mask],
                    Hit::type_(),
                )
            })
            .into(),
        )
    }

    #[deprecated(note = "Use `intersect_any` instead")]
    pub fn trace_any_masked(
        &self,
        ray: impl AsExpr<Value = Ray>,
        mask: impl AsExpr<Value = u32>,
    ) -> Expr<bool> {
        let ray = ray.as_expr().node().get();
        let mask = mask.as_expr().node().get();
        let self_node = self.node.get();
        FromNode::from_node(
            __current_scope(|b| {
                b.call(
                    Func::RayTracingTraceAny,
                    &[self_node, ray, mask],
                    bool::type_(),
                )
            })
            .into(),
        )
    }
    #[deprecated(note = "Use `intersect` instead")]
    #[allow(deprecated)]
    pub fn trace_closest(&self, ray: impl AsExpr<Value = Ray>) -> Expr<Hit> {
        self.trace_closest_masked(ray, u32::MAX.expr())
    }
    #[deprecated(note = "Use `intersect_any` instead")]
    #[allow(deprecated)]
    pub fn trace_any(&self, ray: impl AsExpr<Value = Ray>) -> Expr<bool> {
        self.trace_any_masked(ray, u32::MAX.expr())
    }
    fn make_rq<const TERMINATE_ON_FIRST: bool>(
        &self,
        ray: impl AsExpr<Value = Ray>,
        options: AccelTraceOptions,
    ) -> RayQueryBase<TERMINATE_ON_FIRST> {
        let ray = ray.as_expr().node().get();
        let mask = options.mask.node().get();
        let self_node = self.node.get();
        let query = __current_scope(|b| {
            b.call(
                if TERMINATE_ON_FIRST {
                    Func::RayTracingQueryAny
                } else {
                    Func::RayTracingQueryAll
                },
                &[self_node, ray, mask],
                Type::opaque(
                    if TERMINATE_ON_FIRST {
                        "LC_RayQueryAny"
                    } else {
                        "LC_RayQueryAll"
                    }
                    .into(),
                ),
            )
        });
        RayQueryBase {
            query: query.into(),
            on_procedural_hit: None,
            on_surface_hit: None,
        }
    }
    pub fn traverse(
        &self,
        ray: impl AsExpr<Value = Ray>,
        options: AccelTraceOptions,
    ) -> RayQueryBase<false> {
        self.make_rq(ray, options)
    }
    pub fn traverse_any(
        &self,
        ray: impl AsExpr<Value = Ray>,
        options: AccelTraceOptions,
    ) -> RayQueryBase<true> {
        self.make_rq(ray, options)
    }
    #[deprecated(note = "Use `traverse` instead")]
    pub fn query_all<T, P>(
        &self,
        ray: impl AsExpr<Value = Ray>,
        mask: impl AsExpr<Value = u32>,
        ray_query: RayQuery<T, P>,
    ) -> Expr<CommittedHit>
    where
        T: FnOnce(SurfaceCandidate),
        P: FnOnce(ProceduralCandidate),
    {
        self._query(false, ray, mask, ray_query)
    }

    #[deprecated(note = "Use `traverse_any` instead")]
    pub fn query_any<T, P>(
        &self,
        ray: impl AsExpr<Value = Ray>,
        mask: impl AsExpr<Value = u32>,
        ray_query: RayQuery<T, P>,
    ) -> Expr<CommittedHit>
    where
        T: FnOnce(SurfaceCandidate),
        P: FnOnce(ProceduralCandidate),
    {
        self._query(true, ray, mask, ray_query)
    }

    fn _query<T, P>(
        &self,
        terminate_on_first: bool,
        ray: impl AsExpr<Value = Ray>,
        mask: impl AsExpr<Value = u32>,
        ray_query: RayQuery<T, P>,
    ) -> Expr<CommittedHit>
    where
        T: FnOnce(SurfaceCandidate),
        P: FnOnce(ProceduralCandidate),
    {
        let ray = ray.as_expr().node().get();
        let mask = mask.as_expr().node().get();
        let self_node = self.node.get();
        let query = __current_scope(|b| {
            b.call(
                if terminate_on_first {
                    Func::RayTracingQueryAny
                } else {
                    Func::RayTracingQueryAll
                },
                &[self_node, ray, mask],
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

        with_recorder(|r| {
            let pools = r.pools.clone();
            let s = &mut r.scopes;
            s.push(IrBuilder::new(pools));
        });
        let triangle_candidate = SurfaceCandidate {
            query: query.into(),
            hit: FromNode::from_node(
                __current_scope(|b| {
                    b.call(
                        Func::RayQueryTriangleCandidateHit,
                        &[query],
                        TriangleHit::type_(),
                    )
                })
                .into(),
            ),
        };
        (ray_query.on_triangle_hit)(triangle_candidate);
        let on_triangle_hit = __pop_scope();
        with_recorder(|r| {
            let pools = r.pools.clone();
            let s = &mut r.scopes;
            s.push(IrBuilder::new(pools));
        });
        let procedural_candidate = ProceduralCandidate {
            query: query.into(),
            hit: FromNode::from_node(
                __current_scope(|b| {
                    b.call(
                        Func::RayQueryProceduralCandidateHit,
                        &[query],
                        ProceduralHit::type_(),
                    )
                })
                .into(),
            ),
        };
        (ray_query.on_procedural_hit)(procedural_candidate);
        let on_procedural_hit = __pop_scope();
        FromNode::from_node(
            __current_scope(|b| {
                b.ray_query(query, on_triangle_hit, on_procedural_hit, Type::void());
                b.call(Func::RayQueryCommittedHit, &[query], CommittedHit::type_())
            })
            .into(),
        )
    }
    pub fn new(accel: &rtx::Accel) -> Self {
        let node = with_recorder(|r| {
            let handle: u64 = accel.handle().0;
            let binding = Binding::Accel(AccelBinding { handle });
            if let Some((a, b)) = r.check_on_same_device(&accel.handle.device) {
                panic!(
                    "Accel created for a device: `{:?}` but used in `{:?}`",
                    b, a
                );
            }
            r.capture_or_get(binding, &Arc::downgrade(&accel.handle), || {
                Node::new(CArc::new(Instruction::Accel), Type::void())
            })
        })
        .into();
        Self {
            node,
            handle: Some(accel.handle.clone()),
        }
    }
}
impl_resource_deref_to_var!(Accel, AccelVar);
