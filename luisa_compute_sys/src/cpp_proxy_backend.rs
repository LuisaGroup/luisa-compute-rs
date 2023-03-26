#![allow(unused_unsafe)]
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::{binding, LCStream};
use api::StreamTag;
use backend::Backend;
use libc::c_void;
use luisa_compute_api_types as api;
use luisa_compute_backend as backend;
use luisa_compute_ir::{
    ir::{KernelModule, Type},
    CArc,
};
use parking_lot::Mutex;
use std::sync::Once;
static INIT_CPP: Once = Once::new();
static mut CPP_CONTEXT: binding::LCContext = binding::LCContext { _0: 0 };
static mut OLD_SIGABRT_HANDLER: libc::sighandler_t = 0;
static CPP_MUTEX: Mutex<()> = Mutex::new(());
fn restore_signal_handler() {
    unsafe {
        libc::signal(libc::SIGABRT, OLD_SIGABRT_HANDLER);
        libc::signal(libc::SIGSEGV, OLD_SIGABRT_HANDLER);
    }
}
pub(crate) fn _signal_handler(signal: libc::c_int) {
    if signal == libc::SIGABRT {
        restore_signal_handler();
        panic!("std::abort() called inside LuisaCompute");
    }
    if signal == libc::SIGSEGV {
        restore_signal_handler();
        panic!("segfault inside LuisaCompute");
    }
}
#[macro_export]
macro_rules! catch_abort {
    ($stmts:expr) => {
        unsafe {
            let _guard = CPP_MUTEX.lock();
            OLD_SIGABRT_HANDLER =
                libc::signal(libc::SIGABRT, _signal_handler as libc::sighandler_t);
            OLD_SIGABRT_HANDLER =
                libc::signal(libc::SIGSEGV, _signal_handler as libc::sighandler_t);
            let ret = $stmts;
            restore_signal_handler();
            ret
        }
    };
}

pub fn init_cpp<P: AsRef<Path>>(_bin_path: P) {
    INIT_CPP.call_once(|| unsafe {});
}

pub struct CppProxyBackend {
    device: binding::LCDevice,
}
fn default_path() -> PathBuf {
    std::env::current_exe().unwrap()
}
impl CppProxyBackend {
    pub fn new(backend: &str) -> Arc<Self> {
        init_cpp(default_path());
        let backend_c_str = std::ffi::CString::new(backend).unwrap();
        let device = catch_abort!({
            binding::luisa_compute_device_create(
                CPP_CONTEXT,
                backend_c_str.as_ptr(),
                b"{}\0".as_ptr() as *const _,
            )
        });
        Arc::new(Self { device })
    }
}
impl Backend for CppProxyBackend {
    fn create_buffer(
        &self,
        ty: &CArc<Type>,
        count: usize,
    ) -> backend::Result<api::CreatedBufferInfo> {
        let buffer = catch_abort!({
            binding::luisa_compute_buffer_create(
                self.device,
                ty as *const _ as *const c_void,
                count as u64,
            )
        });
        unsafe { Ok(std::mem::transmute(buffer)) }
    }

    fn destroy_buffer(&self, buffer: api::Buffer) {
        catch_abort!({
            binding::luisa_compute_buffer_destroy(self.device, binding::LCBuffer { _0: buffer.0 });
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
    ) -> backend::Result<api::CreatedResourceInfo> {
        unsafe {
            let texture = catch_abort!({
                binding::luisa_compute_texture_create(
                    self.device,
                    std::mem::transmute(format),
                    dimension,
                    width,
                    height,
                    depth,
                    mipmap_levels,
                )
            });
            Ok(std::mem::transmute(texture))
        }
    }

    fn destroy_texture(&self, texture: api::Texture) {
        catch_abort!({
            binding::luisa_compute_texture_destroy(
                self.device,
                binding::LCTexture { _0: texture.0 },
            )
        })
    }

    fn create_bindless_array(&self, size: usize) -> backend::Result<api::CreatedResourceInfo> {
        let array = catch_abort!({
            binding::luisa_compute_bindless_array_create(self.device, size as u64)
        });
        unsafe { Ok(std::mem::transmute(array)) }
    }

    fn destroy_bindless_array(&self, array: api::BindlessArray) {
        catch_abort!({
            binding::luisa_compute_bindless_array_destroy(
                self.device,
                binding::LCBindlessArray { _0: array.0 },
            )
        })
    }

    fn create_stream(&self, tag: StreamTag) -> backend::Result<api::CreatedResourceInfo> {
        unsafe {
            let stream = catch_abort!({
                binding::luisa_compute_stream_create(self.device, std::mem::transmute(tag))
            });
            Ok(std::mem::transmute(stream))
        }
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

    fn dispatch(
        &self,
        stream: api::Stream,
        command_list: &[api::Command],
        callback: (extern "C" fn(*mut u8), *mut u8),
    ) -> backend::Result<()> {
        catch_abort!({
            binding::luisa_compute_stream_dispatch(
                self.device,
                binding::LCStream { _0: stream.0 },
                binding::LCCommandList {
                    commands: command_list.as_ptr() as *const binding::LCCommand,
                    commands_count: command_list.len() as u64,
                },
                Some(callback.0),
                callback.1,
            );
        });
        Ok(())
    }

    fn create_shader(
        &self,
        kernel: CArc<KernelModule>,
        option: &api::ShaderOption,
    ) -> backend::Result<api::CreatedShaderInfo> {
        //  let debug =
        //     luisa_compute_ir::ir::debug::luisa_compute_ir_dump_human_readable(&kernel.module);
        // let debug = std::ffi::CString::new(debug.as_ref()).unwrap();
        // println!("{}", debug.to_str().unwrap());
        unsafe {
            Ok(std::mem::transmute(catch_abort!({
                binding::luisa_compute_shader_create(
                    self.device,
                    binding::LCKernelModule {
                        ptr: CArc::as_ptr(&kernel) as u64,
                    },
                    option as *const _ as *const binding::LCShaderOption,
                )
            })))
        }
    }

    fn destroy_shader(&self, shader: api::Shader) {
        catch_abort!({
            binding::luisa_compute_shader_destroy(self.device, binding::LCShader { _0: shader.0 })
        })
    }

    fn create_event(&self) -> backend::Result<api::CreatedResourceInfo> {
        unsafe {
            let event = catch_abort!({ binding::luisa_compute_event_create(self.device) });
            Ok(std::mem::transmute(event))
        }
    }

    fn destroy_event(&self, event: api::Event) {
        catch_abort!({
            binding::luisa_compute_event_destroy(self.device, binding::LCEvent { _0: event.0 })
        })
    }

    fn signal_event(&self, event: api::Event, stream: api::Stream) {
        catch_abort!({
            binding::luisa_compute_event_signal(
                self.device,
                binding::LCEvent { _0: event.0 },
                binding::LCStream { _0: stream.0 },
            )
        })
    }

    fn wait_event(&self, event: api::Event, stream: api::Stream) -> backend::Result<()> {
        catch_abort!({
            binding::luisa_compute_event_wait(
                self.device,
                binding::LCEvent { _0: event.0 },
                binding::LCStream { _0: stream.0 },
            )
        });
        Ok(())
    }

    fn synchronize_event(&self, event: api::Event) -> backend::Result<()> {
        catch_abort!({
            binding::luisa_compute_event_synchronize(self.device, binding::LCEvent { _0: event.0 })
        });
        Ok(())
    }

    fn create_mesh(&self, option: api::AccelOption) -> backend::Result<api::CreatedResourceInfo> {
        unsafe {
            let mesh = catch_abort!({
                binding::luisa_compute_mesh_create(
                    self.device,
                    &option as *const _ as *const binding::LCAccelOption,
                )
            });
            Ok(std::mem::transmute(mesh))
        }
    }

    fn destroy_mesh(&self, mesh: api::Mesh) {
        catch_abort!(binding::luisa_compute_mesh_destroy(
            self.device,
            binding::LCMesh { _0: mesh.0 }
        ))
    }

    fn create_accel(&self, option: api::AccelOption) -> backend::Result<api::CreatedResourceInfo> {
        unsafe {
            let mesh = catch_abort!({
                binding::luisa_compute_accel_create(
                    self.device,
                    &option as *const _ as *const binding::LCAccelOption,
                )
            });
            Ok(std::mem::transmute(mesh))
        }
    }

    fn destroy_accel(&self, accel: api::Accel) {
        catch_abort!(binding::luisa_compute_accel_destroy(
            self.device,
            binding::LCAccel { _0: accel.0 }
        ))
    }

    fn query(&self, property: &str) -> Option<String> {
        catch_abort! {{
            let property = std::ffi::CString::new(property).unwrap();
            let property = property.as_ptr();
            let str_buf = vec![0u8; 1024];
            let result_len = binding::luisa_compute_device_query(self.device, property, str_buf.as_ptr() as *mut i8, str_buf.len() as u64);
            if result_len > 0 {
                let result_str = std::ffi::CStr::from_ptr(str_buf.as_ptr() as *const i8).to_str().unwrap().to_string();
                Some(result_str)
            } else {
                None
            }
        }}
    }

    fn shader_cache_dir(&self, _shader: api::Shader) -> Option<PathBuf> {
        todo!()
    }

    fn create_procedural_primitive(
        &self,
        _option: api::AccelOption,
    ) -> backend::Result<api::CreatedResourceInfo> {
        todo!()
    }

    fn destroy_procedural_primitive(&self, _primitive: api::ProceduralPrimitive) {
        todo!()
    }

    unsafe fn set_swapchain_contex(&self, _ctx: backend::SwapChainForCpuContext) {}

    fn create_swap_chain(
        &self,
        window_handle: u64,
        stream_handle: api::Stream,
        width: u32,
        height: u32,
        allow_hdr: bool,
        vsync: bool,
        back_buffer_size: u32,
    ) -> backend::Result<api::CreatedSwapchainInfo> {
        catch_abort!({
            let swap_chain = binding::luisa_compute_swapchain_create(
                self.device,
                window_handle,
                LCStream {
                    _0: stream_handle.0,
                },
                width,
                height,
                allow_hdr,
                vsync,
                back_buffer_size,
            );
            Ok(std::mem::transmute(swap_chain))
        })
    }

    fn destroy_swap_chain(&self, swap_chain: api::Swapchain) {
        catch_abort!({
            binding::luisa_compute_swapchain_destroy(
                self.device,
                crate::LCSwapchain { _0: swap_chain.0 },
            )
        })
    }

    fn present_display_in_stream(
        &self,
        stream_handle: api::Stream,
        swapchain_handle: api::Swapchain,
        image_handle: api::Texture,
    ) {
        catch_abort!({
            binding::luisa_compute_swapchain_present(
                self.device,
                std::mem::transmute(stream_handle),
                std::mem::transmute(swapchain_handle),
                std::mem::transmute(image_handle),
            )
        })
    }
}
