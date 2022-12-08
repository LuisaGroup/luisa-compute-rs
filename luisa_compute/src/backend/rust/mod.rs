// A Rust implementation of LuisaCompute backend.

use std::sync::Arc;

use crate::prelude::{Device, DeviceHandle};

use super::Backend;
mod resource;
pub struct RustBackend {}
impl Backend for RustBackend {
    fn create_buffer(&self, size_bytes: usize) -> super::Result<luisa_compute_api_types::Buffer> {
        todo!()
    }

    fn destroy_buffer(&self, texture: luisa_compute_api_types::Buffer) {
        todo!()
    }

    fn buffer_native_handle(&self, texture: luisa_compute_api_types::Buffer) -> *mut libc::c_void {
        todo!()
    }

    fn create_texture(
        &self,
        format: luisa_compute_api_types::PixelFormat,
        dimension: u32,
        width: u32,
        height: u32,
        depth: u32,
        mipmap_levels: u32,
    ) -> super::Result<luisa_compute_api_types::Texture> {
        todo!()
    }

    fn destroy_texture(&self, texture: luisa_compute_api_types::Texture) {
        todo!()
    }

    fn texture_native_handle(
        &self,
        texture: luisa_compute_api_types::Texture,
    ) -> *mut libc::c_void {
        todo!()
    }

    fn create_bindless_array(
        &self,
        size: usize,
    ) -> super::Result<luisa_compute_api_types::BindlessArray> {
        todo!()
    }

    fn destroy_bindless_array(&self, array: luisa_compute_api_types::BindlessArray) {
        todo!()
    }

    fn emplace_buffer_in_bindless_array(
        &self,
        array: luisa_compute_api_types::BindlessArray,
        index: usize,
        handle: luisa_compute_api_types::Buffer,
        offset_bytes: usize,
    ) {
        todo!()
    }

    fn emplace_tex2d_in_bindless_array(
        &self,
        array: luisa_compute_api_types::BindlessArray,
        index: usize,
        handle: luisa_compute_api_types::Texture,
        sampler: luisa_compute_api_types::Sampler,
    ) {
        todo!()
    }

    fn emplace_tex3d_in_bindless_array(
        &self,
        array: luisa_compute_api_types::BindlessArray,
        index: usize,
        handle: luisa_compute_api_types::Texture,
        sampler: luisa_compute_api_types::Sampler,
    ) {
        todo!()
    }

    fn remove_buffer_from_bindless_array(
        &self,
        array: luisa_compute_api_types::BindlessArray,
        index: usize,
    ) {
        todo!()
    }

    fn remove_tex2d_from_bindless_array(
        &self,
        array: luisa_compute_api_types::BindlessArray,
        index: usize,
    ) {
        todo!()
    }

    fn remove_tex3d_from_bindless_array(
        &self,
        array: luisa_compute_api_types::BindlessArray,
        index: usize,
    ) {
        todo!()
    }

    fn create_stream(&self) -> super::Result<luisa_compute_api_types::Stream> {
        todo!()
    }

    fn destroy_stream(&self, stream: luisa_compute_api_types::Stream) {
        todo!()
    }

    fn synchronize_stream(&self, stream: luisa_compute_api_types::Stream) -> super::Result<()> {
        todo!()
    }

    fn stream_native_handle(&self, stream: luisa_compute_api_types::Stream) -> *mut libc::c_void {
        todo!()
    }

    fn dispatch(
        &self,
        stream: luisa_compute_api_types::Stream,
        command_list: &[luisa_compute_api_types::Command],
    ) -> super::Result<()> {
        todo!()
    }

    fn create_shader(
        &self,
        kernel: &luisa_compute_ir::ir::KernelModule,
        meta_options: &str,
    ) -> super::Result<luisa_compute_api_types::Shader> {
        todo!()
    }

    fn destroy_shader(&self, shader: luisa_compute_api_types::Shader) {
        todo!()
    }

    fn create_event(&self) -> super::Result<luisa_compute_api_types::Event> {
        todo!()
    }

    fn destroy_event(&self, event: luisa_compute_api_types::Event) {
        todo!()
    }

    fn signal_event(&self, event: luisa_compute_api_types::Event) {
        todo!()
    }

    fn wait_event(&self, event: luisa_compute_api_types::Event) -> super::Result<()> {
        todo!()
    }

    fn synchronize_event(&self, event: luisa_compute_api_types::Event) -> super::Result<()> {
        todo!()
    }
}
impl RustBackend {
    pub fn create_device() -> super::Result<Device> {
        let backend = Arc::new(RustBackend {});
        let default_stream = backend.create_stream()?;
        Ok(Device {
            inner: Arc::new(DeviceHandle {
                backend,
                default_stream,
            }),
        })
    }
}
