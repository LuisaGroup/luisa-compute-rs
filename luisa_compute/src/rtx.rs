use std::{collections::HashMap, marker::PhantomData, sync::Arc};

use crate::{lang::AccelVar, runtime::submit_default_stream_and_sync, ResourceTracker, *};
use api::AccelBuildRequest;
use luisa_compute_api_types as api;
use parking_lot::RwLock;
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
pub struct Accel {
    pub(crate) handle: Arc<AccelHandle>,
    pub(crate) mesh_handles: RwLock<Vec<Option<Arc<MeshHandle>>>>,
    pub(crate) modifications: RwLock<HashMap<usize, api::AccelBuildModification>>,
}
pub(crate) struct MeshHandle {
    pub(crate) device: Device,
    pub(crate) handle: api::Mesh,
    pub(crate) native_handle: *mut std::ffi::c_void,
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
        }
    }
    pub fn build(&self, request: AccelBuildRequest) {
        submit_default_stream_and_sync(&self.handle.device, [self.build_async(request)]).unwrap();
    }
}

impl Accel {
    fn push_handle(&self, handle: Arc<MeshHandle>, transform: Mat4, visible: u8, opaque: bool) {
        let mut flags = api::AccelBuildModificationFlags::PRIMITIVE;

        flags |= api::AccelBuildModificationFlags::VISIBILITY;

        if opaque {
            flags |= api::AccelBuildModificationFlags::OPAQUE;
        }
        let mut modifications = self.modifications.write();
        let mut mesh_handles = self.mesh_handles.write();
        let index = modifications.len() as u32;
        modifications.insert(
            mesh_handles.len(),
            api::AccelBuildModification {
                mesh: handle.handle.0,
                affine: transform.into_affine3x4(),
                flags,
                visibility: visible,
                index,
            },
        );

        mesh_handles.push(Some(handle));
    }
    fn set_handle(
        &self,
        index: usize,
        handle: Arc<MeshHandle>,
        transform: Mat4,
        visible: u8,
        opaque: bool,
    ) {
        let mut flags = api::AccelBuildModificationFlags::PRIMITIVE;
        dbg!(flags);
        flags |= api::AccelBuildModificationFlags::VISIBILITY;

        if opaque {
            flags |= api::AccelBuildModificationFlags::OPAQUE_ON;
        }
        let mut modifications = self.modifications.write();
        modifications.insert(
            index as usize,
            api::AccelBuildModification {
                mesh: handle.handle.0,
                affine: transform.into_affine3x4(),
                flags,
                visibility: visible,
                index: index as u32,
            },
        );
        let mut mesh_handles = self.mesh_handles.write();
        mesh_handles[index] = Some(handle.clone());
    }
    pub fn push_mesh(&self, mesh: &Mesh, transform: Mat4, visible: u8, opaque: bool) {
        self.push_handle(mesh.handle.clone(), transform, visible, opaque)
    }
    pub fn set_mesh(&self, index: usize, mesh: &Mesh, transform: Mat4, visible: u8, opaque: bool) {
        self.set_handle(index, mesh.handle.clone(), transform, visible, opaque)
    }
    pub fn pop(&self) {
        let mut modifications = self.modifications.write();
        let mut mesh_handles = self.mesh_handles.write();
        let n = mesh_handles.len();
        modifications.remove(&n);
        mesh_handles.pop().unwrap();
    }
    pub fn build(&self, request: api::AccelBuildRequest) {
        submit_default_stream_and_sync(&self.handle.device, [self.build_async(request)]).unwrap()
    }
    pub fn build_async<'a>(&'a self, request: api::AccelBuildRequest) -> Command<'a> {
        let mut rt = ResourceTracker::new();
        let mesh_handles = self.mesh_handles.read();
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
                instance_count: mesh_handles.len() as u32,
                modifications: m.as_ptr(),
                modifications_count: m.len(),
                build_accel: true,
            }),
            resource_tracker: rt,
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
