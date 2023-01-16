use std::{path::PathBuf, sync::Arc};

use crate::binding;
use backend::Backend;
use luisa_compute_api_types as api;
use luisa_compute_backend as backend;
use luisa_compute_ir::{
    ir::{KernelModule, Type},
    Gc,
};
use std::sync::Once;
static INIT_CPP: Once = Once::new();
static mut CPP_CONTEXT: binding::LCContext = binding::LCContext { _0: 0 };
pub(crate) fn _signal_handler(signal: libc::c_int) {
    if signal == libc::SIGABRT {
        panic!("std::abort() called inside LuisaCompute");
    }
}
#[macro_export]
macro_rules! catch_abort {
    ($stmts:expr) => {
        unsafe {
            let old = libc::signal(libc::SIGABRT, _signal_handler as libc::sighandler_t);
            let ret = $stmts;
            libc::signal(libc::SIGABRT, old);
            ret
        }
    };
}
fn init_cpp() {
    INIT_CPP.call_once(|| unsafe {
        let gc_ctx = luisa_compute_ir::ir::luisa_compute_gc_context();
        let ir_ctx = luisa_compute_ir::context::luisa_compute_ir_context();
        assert!(gc_ctx != std::ptr::null_mut());
        assert!(ir_ctx != std::ptr::null_mut());

        let ctx = binding::LCAppContext {
            gc_context: gc_ctx as *mut _,
            ir_context: ir_ctx as *mut _,
        };
        binding::luisa_compute_set_app_context(ctx);
        let self_path = std::env::current_exe().unwrap();
        let self_path_c_str = std::ffi::CString::new(self_path.to_str().unwrap()).unwrap();
        CPP_CONTEXT = binding::luisa_compute_context_create(self_path_c_str.as_ptr());
    });
}

pub struct CppProxyBackend {
    device: binding::LCDevice,
}
impl CppProxyBackend {
    pub fn new(backend: &str) -> Arc<Self> {
        init_cpp();
        let backend_c_str = std::ffi::CString::new(backend).unwrap();
        let device = catch_abort!({
            binding::luisa_compute_device_create(
                CPP_CONTEXT,
                backend_c_str.as_ptr(),
                b"\0".as_ptr() as *const _,
            )
        });
        Arc::new(Self { device })
    }
}
impl Backend for CppProxyBackend {
    fn create_buffer(&self, size_bytes: usize, _align: usize) -> backend::Result<api::Buffer> {
        Ok(api::Buffer(
            catch_abort!({ binding::luisa_compute_buffer_create(self.device, size_bytes as u64) })
                ._0,
        ))
    }

    fn destroy_buffer(&self, buffer: api::Buffer) {
        catch_abort!({
            binding::luisa_compute_buffer_destroy(self.device, binding::LCBuffer { _0: buffer.0 });
        })
    }

    fn buffer_native_handle(&self, buffer: api::Buffer) -> *mut std::ffi::c_void {
        catch_abort!({
            // binding::luisa_compute_buffer_native_handle(self.device, binding::LCBuffer { _0: buffer.0 })
            todo!()
        })
    }

    fn create_texture(
        &self,
        format: api::PixelFormat,
        dimension: u32,
        width: u32,
        height: u32,
        depth: u32,
        mipmap_levels: u32,
    ) -> backend::Result<api::Texture> {
        todo!()
    }

    fn destroy_texture(&self, texture: api::Texture) {
        todo!()
    }

    fn texture_native_handle(&self, texture: api::Texture) -> *mut std::ffi::c_void {
        todo!()
    }

    fn create_bindless_array(&self, size: usize) -> backend::Result<api::BindlessArray> {
        Ok(api::BindlessArray(
            catch_abort!({
                binding::luisa_compute_bindless_array_create(self.device, size as u64)
            })
            ._0,
        ))
    }

    fn destroy_bindless_array(&self, array: api::BindlessArray) {
        catch_abort!({
            binding::luisa_compute_bindless_array_destroy(
                self.device,
                binding::LCBindlessArray { _0: array.0 },
            )
        })
    }

    fn emplace_buffer_in_bindless_array(
        &self,
        array: api::BindlessArray,
        index: usize,
        handle: api::Buffer,
        offset_bytes: usize,
    ) {
        assert_eq!(offset_bytes, 0);
        catch_abort!({
            binding::luisa_compute_bindless_array_emplace_buffer(
                self.device,
                binding::LCBindlessArray { _0: array.0 },
                index as u64,
                binding::LCBuffer { _0: handle.0 },
            )
        })
    }

    fn emplace_tex2d_in_bindless_array(
        &self,
        array: api::BindlessArray,
        index: usize,
        handle: api::Texture,
        sampler: api::Sampler,
    ) {
        todo!()
    }

    fn emplace_tex3d_in_bindless_array(
        &self,
        array: api::BindlessArray,
        index: usize,
        handle: api::Texture,
        sampler: api::Sampler,
    ) {
        todo!()
    }

    fn remove_buffer_from_bindless_array(&self, array: api::BindlessArray, index: usize) {
        catch_abort!({
            binding::luisa_compute_bindless_array_remove_buffer(
                self.device,
                binding::LCBindlessArray { _0: array.0 },
                index as u64,
            )
        })
    }

    fn remove_tex2d_from_bindless_array(&self, array: api::BindlessArray, index: usize) {
        todo!()
    }

    fn remove_tex3d_from_bindless_array(&self, array: api::BindlessArray, index: usize) {
        todo!()
    }

    fn create_stream(&self) -> backend::Result<api::Stream> {
        Ok(api::Stream(
            catch_abort!({ binding::luisa_compute_stream_create(self.device) })._0,
        ))
    }

    fn destroy_stream(&self, stream: api::Stream) {
        catch_abort!({
            binding::luisa_compute_stream_destroy(self.device, binding::LCStream { _0: stream.0 })
        })
    }

    fn synchronize_stream(&self, stream: api::Stream) -> backend::Result<()> {
        catch_abort!({
            binding::luisa_compute_stream_synchronize(
                self.device,
                binding::LCStream { _0: stream.0 },
            )
        });
        Ok(())
    }

    fn stream_native_handle(&self, stream: api::Stream) -> *mut std::ffi::c_void {
        todo!()
    }

    fn dispatch(&self, stream: api::Stream, command_list: &[api::Command]) -> backend::Result<()> {
        catch_abort!({
            binding::luisa_compute_stream_dispatch(
                self.device,
                binding::LCStream { _0: stream.0 },
                binding::LCCommandList {
                    commands: command_list.as_ptr() as *const binding::LCCommand,
                    commands_count: command_list.len() as u64,
                },
            );
        });
        Ok(())
    }

    fn create_shader(&self, kernel: KernelModule) -> backend::Result<api::Shader> {
        Ok(api::Shader(
            catch_abort!({
                binding::luisa_compute_shader_create(
                    self.device,
                    &kernel as *const _ as *const binding::LCKernelModule,
                    std::ptr::null(),
                )
            })
            ._0,
        ))
    }

    fn create_shader_async(&self, kernel: KernelModule) -> backend::Result<api::Shader> {
        todo!()
    }

    fn shader_cache_dir(&self, shader: api::Shader) -> Option<std::path::PathBuf> {
        Some(PathBuf::new())
    }

    fn destroy_shader(&self, shader: api::Shader) {
        catch_abort!({
            binding::luisa_compute_shader_destroy(self.device, binding::LCShader { _0: shader.0 })
        })
    }

    fn create_event(&self) -> backend::Result<api::Event> {
        todo!()
    }

    fn destroy_event(&self, event: api::Event) {
        todo!()
    }

    fn signal_event(&self, event: api::Event) {
        todo!()
    }

    fn wait_event(&self, event: api::Event) -> backend::Result<()> {
        todo!()
    }

    fn synchronize_event(&self, event: api::Event) -> backend::Result<()> {
        todo!()
    }

    fn is_cpu_backend(&self) -> bool {
        false
    }

    fn set_buffer_type(&self, _buffer: api::Buffer, _ty: Gc<Type>) {}
}
