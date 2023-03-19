use std::{cell::RefCell, collections::HashMap, marker::PhantomData, sync::Arc};

use crate::{
    prelude::{Command, Device, Mat4},
    runtime::submit_default_stream_and_sync,
    ResourceTracker, lang::AccelVar,
};
use api::AccelBuildModification;
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
pub struct Accel {
    pub(crate) handle: Arc<AccelHandle>,
    pub(crate) mesh_handles: RefCell<Vec<Arc<MeshHandle>>>,
    pub(crate) modifications: RefCell<HashMap<usize, api::AccelBuildModification>>,
}
pub(crate) struct MeshHandle {
    pub(crate) device: Device,
    pub(crate) handle: api::Mesh,
    pub(crate) native_handle: *mut std::ffi::c_void,
}

pub struct Mesh {
    pub(crate) handle: Arc<MeshHandle>,
}
impl Accel {
    // fn push_handle(&self, handle: Arc<MeshHandle>, transform: Mat4, visible: bool, opaque: bool) {
    //     let mut flags = api::AccelBuildModificationFlags::EMPTY;
    //     if visible {
    //         flags |= api::AccelBuildModificationFlags::VISIBILITY_ON;
    //     }
    //     if opaque {
    //         flags |= api::AccelBuildModificationFlags::OPAQUE;
    //     }
    //     let mut modifications = self.modifications.borrow_mut();
    //     let mut mesh_handles = self.mesh_handles.borrow_mut();
    //     let index = modifications.len() as u32;
    //     modifications.insert(
    //         mesh_handles.len(),
    //         api::AccelBuildModification {
    //             mesh: handle.handle.0,
    //             affine: transform.into_affine3x4(),
    //             flags,
    //             index,
    //         },
    //     );

    //     mesh_handles.push(handle);
    // }
    // pub fn set_handle(
    //     &self,
    //     index: usize,
    //     handle: api::Mesh,
    //     transform: Mat4,
    //     visible: bool,
    //     opaque: bool,
    // ) {
    //     let mut flags = api::AccelBuildModificationFlags::EMPTY;
    //     if visible {
    //         flags |= api::AccelBuildModificationFlags::VISIBILITY_ON;
    //     }
    //     if opaque {
    //         flags |= api::AccelBuildModificationFlags::OPAQUE;
    //     }
    //     let mut modifications = self.modifications.borrow_mut();
    //     let index = modifications.len() as u32;
    //     modifications.insert(
    //         index as usize,
    //         api::AccelBuildModification {
    //             mesh: handle.0,
    //             affine: transform.into_affine3x4(),
    //             flags,
    //             index,
    //         },
    //     );
    //     let mut mesh_handles = self.mesh_handles.borrow_mut();
    //     mesh_handles[index as usize] = Arc::new(MeshHandle {
    //         device: self.handle.device.clone(),
    //         handle,
    //     });
    // }
    // pub fn push_mesh(&self, mesh: &Mesh, transform: Mat4, visible: bool, opaque: bool) {
    //     self.push_handle(mesh.handle.clone(), transform, visible, opaque)
    // }
    // pub fn set_mesh(
    //     &self,
    //     index: usize,
    //     mesh: &Mesh,
    //     transform: Mat4,
    //     visible: bool,
    //     opaque: bool,
    // ) {
    //     self.set_handle(index, mesh.handle.handle, transform, visible, opaque)
    // }
    // pub fn pop(&self) {
    //     let mut modifications = self.modifications.borrow_mut();
    //     let mut mesh_handles = self.mesh_handles.borrow_mut();
    //     let n = mesh_handles.len();
    //     modifications.remove(&n);
    //     mesh_handles.pop().unwrap();
    // }
    // pub fn update(&self, build_accel: bool, request: api::AccelBuildRequest) {
    //     unsafe {
    //         submit_default_stream_and_sync(
    //             &self.handle.device,
    //             [self.update_async(build_accel, request)],
    //         )
    //         .unwrap()
    //     }
    // }
    // pub unsafe fn update_async<'a>(
    //     &'a self,
    //     build_accel: bool,
    //     request: api::AccelBuildRequest,
    // ) -> Command<'a> {
    //     let mut rt = ResourceTracker::new();
    //     let mesh_handles = self.mesh_handles.borrow();
    //     rt.add(self.handle.clone());
    //     let mut modifications = self.modifications.borrow_mut();
    //     let m = modifications.drain().map(|(_, v)| v).collect::<Vec<_>>();
    //     let m = Arc::new(m);
    //     rt.add(m.clone());
    //     Command {
    //         marker: PhantomData,
    //         inner: api::Command::AccelBuild(api::AccelBuildCommand {
    //             accel: self.handle.handle,
    //             request,
    //             instance_count: mesh_handles.len() as u32,
    //             modifications: m.as_ptr(),
    //             modifications_count: m.len(),
    //             build_accel,
    //         }),
    //         resource_tracker: rt,
    //     }
    // }
    pub fn var(&self)->AccelVar {
        AccelVar::new(self)
    }
    pub(crate) fn handle(&self) -> api::Accel {
        self.handle.handle
    }
}
