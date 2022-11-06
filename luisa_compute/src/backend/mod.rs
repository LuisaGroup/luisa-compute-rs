use crate::prelude::*;
use api::PixelFormat;
use libc::c_void;
use luisa_compute_api_types as api;
use luisa_compute_ir::ir::KernelModule;
use sys::LCDeviceInterface;
pub trait Backend {
    fn create_buffer(&self, size_bytes: usize) -> api::Buffer;
    fn destroy_buffer(&self, texture: api::Buffer);
    fn buffer_native_handle(&self, texture: api::Buffer) -> *mut c_void;
    fn create_texture(
        &self,
        format: PixelFormat,
        dimension: u32,
        width: u32,
        height: u32,
        depth: u32,
        mipmap_levels: u32,
    ) -> u64;
    fn destroy_texture(&self, texture: api::Texture);
    fn texture_native_handle(&self, texture: api::Texture) -> *mut c_void;
    fn create_bindless_array(&self, size: usize) -> u64;
    fn destroy_bindless_array(&self, array: api::BindlessArray);
    fn emplace_buffer_in_bindless_array(
        &self,
        array: api::BindlessArray,
        index: usize,
        handle: api::Buffer,
        offset_bytes: usize,
    );
    fn emplace_tex2d_in_bindless_array(
        &self,
        array: api::BindlessArray,
        index: usize,
        handle: api::Texture,
        sampler: api::Sampler,
    );
    fn emplace_tex3d_in_bindless_array(
        &self,
        array: api::BindlessArray,
        index: usize,
        handle: api::Texture,
        sampler: api::Sampler,
    );
    
    fn remove_buffer_in_bindless_array(&self, array: api::BindlessArray, handle: api::Buffer);
    fn remove_tex2d_in_bindless_array(&self, array: api::BindlessArray, handle: api::Texture);
    fn remove_tex3d_in_bindless_array(&self, array: api::BindlessArray, handle: api::Texture);
    fn create_stream(&self) -> api::Stream;
    fn destroy_stream(&self, stream: api::Stream);
    fn synchronize_stream(&self, stream: api::Stream);
    fn stream_native_handle(&self, stream: api::Stream) -> *mut c_void;
    fn dispatch(&self, stream: api::Stream, list: api::CommandList);
    // fn create_swap_chain(
    //     &self,
    //     window_handle: u64,
    //     stream_handle: u64,
    //     width: u32,
    //     height: u32,
    //     allow_hdr: bool,
    //     back_buffer_size: u32,
    // ) -> u64;
    // fn destroy_swap_chain(&self, swap_chain: u64);
    // fn swap_chain_pixel_storage(&self, swap_chain: u64) -> api::PixelStorage;
    // fn present_display_in_stream(
    //     &self,
    //     stream_handle: u64,
    //     swapchain_handle: u64,
    //     image_handle: u64,
    // );
    fn create_shader_ex(&self, kernel: &KernelModule, meta_options: &str) -> u64;
    fn destroy_shader(&self, shader: u64);
    fn create_event(&self) -> u64;
    fn destroy_event(&self, event: u64);
    fn signal_event(&self, event: u64);
    fn wait_event(&self, event: u64);
    fn synchronize_event(&self, event: u64);
    // fn create_mesh(
    //     &self,
    //     v_texture: api::Buffer,
    //     v_offset: usize,
    //     v_stride: usize,
    //     v_count: usize,
    //     t_texture: api::Buffer,
    //     t_offset: usize,
    //     t_count: usize,
    //     hint: api::AccelUsageHint,
    // ) -> u64;
    // fn destroy_mesh(&self, mesh: u64);
    // fn create_accel()
}

// pub struct BackendWrapper<B: Backend> {
//     interface: LCDeviceInterface,
//     backend: B,
// }
