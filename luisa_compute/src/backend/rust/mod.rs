// A Rust implementation of LuisaCompute backend.

use std::{ptr::null, sync::Arc};

use base64ct::Encoding;
use rayon::ThreadPool;
use sha2::{Digest, Sha256};

use crate::prelude::{Device, DeviceHandle};

use self::{resource::BufferImpl, stream::StreamImpl};

use super::Backend;
mod codegen;
mod resource;
mod shader;
mod shader_impl;
mod stream;
pub struct RustBackend {
    shared_pool: Arc<rayon::ThreadPool>,
}
impl Backend for RustBackend {
    fn create_buffer(
        &self,
        size_bytes: usize,
        align: usize,
    ) -> super::Result<luisa_compute_api_types::Buffer> {
        let buffer = Box::new(BufferImpl::new(size_bytes, align));
        let ptr = Box::into_raw(buffer);
        Ok(luisa_compute_api_types::Buffer(ptr as u64))
    }

    fn destroy_buffer(&self, buffer: luisa_compute_api_types::Buffer) {
        unsafe {
            let ptr = buffer.0 as *mut BufferImpl;
            drop(Box::from_raw(ptr));
        }
    }

    fn buffer_native_handle(&self, buffer: luisa_compute_api_types::Buffer) -> *mut libc::c_void {
        unsafe {
            let buffer = &*(buffer.0 as *mut BufferImpl);
            buffer.data as *mut libc::c_void
        }
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
        let stream = Box::into_raw(Box::new(StreamImpl::new(self.shared_pool.clone())));
        Ok(luisa_compute_api_types::Stream(stream as u64))
    }

    fn destroy_stream(&self, stream: luisa_compute_api_types::Stream) {
        unsafe {
            let stream = stream.0 as *mut StreamImpl;
            drop(Box::from_raw(stream));
        }
    }

    fn synchronize_stream(&self, stream: luisa_compute_api_types::Stream) -> super::Result<()> {
        unsafe {
            let stream = stream.0 as *mut StreamImpl;
            (*stream).synchronize();
            Ok(())
        }
    }

    fn stream_native_handle(&self, stream: luisa_compute_api_types::Stream) -> *mut libc::c_void {
        stream.0 as *mut libc::c_void
    }

    fn dispatch(
        &self,
        stream_: luisa_compute_api_types::Stream,
        command_list: &[luisa_compute_api_types::Command],
    ) -> super::Result<()> {
        unsafe {
            let stream = &*(stream_.0 as *mut StreamImpl);
            let command_list = command_list.to_vec();
            stream.enqueue(move || {
                let stream = &*(stream_.0 as *mut StreamImpl);
                stream.dispatch(&command_list)
            });
            Ok(())
        }
    }

    fn create_shader(
        &self,
        kernel: &luisa_compute_ir::ir::KernelModule,
        _meta_options: &str,
    ) -> super::Result<luisa_compute_api_types::Shader> {
        // let debug =
        //     luisa_compute_ir::ir::debug::luisa_compute_ir_dump_human_readable(&kernel.module);
        // let debug = std::ffi::CString::new(debug.as_ref()).unwrap();
        // println!("{}", debug.to_str().unwrap());
        let gened_src = codegen::CodeGen::run(&kernel);
        // println!("{}", gened_src);
        let lib_path = shader::compile(gened_src).unwrap();
        let shader = Box::new(shader::ShaderImpl::load(lib_path));
        let shader = Box::into_raw(shader);
        Ok(luisa_compute_api_types::Shader(shader as u64))
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
        let backend = Arc::new(RustBackend {
            shared_pool: Arc::new(rayon::ThreadPoolBuilder::new().build().unwrap()),
        });
        let default_stream = backend.create_stream()?;
        Ok(Device {
            inner: Arc::new(DeviceHandle {
                backend,
                default_stream,
            }),
        })
    }
}
fn sha256(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s);
    let hash = hasher.finalize();
    format!("A{}", base64ct::Base64UrlUnpadded::encode_string(&hash))
}
