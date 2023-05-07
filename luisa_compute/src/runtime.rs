use crate::backend::Backend;
use crate::lang::KernelBuildOptions;
use crate::*;
use crate::{lang::Value, resource::*};

use api::AccelOption;
use lang::{KernelBuildFn, KernelBuilder, KernelParameter, KernelSignature};
pub use luisa_compute_api_types as api;
use luisa_compute_backend::proxy::ProxyBackend;
use luisa_compute_ir::ir::{self, KernelModule};
use luisa_compute_ir::CArc;
use parking_lot::lock_api::RawMutex as RawMutexTrait;
use parking_lot::{Condvar, Mutex, RawMutex, RwLock};
use raw_window_handle::HasRawWindowHandle;
use rtx::{Accel, Mesh, MeshHandle};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::CString;
use std::hash::Hash;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;
use winit::window::Window;

#[derive(Clone)]
pub struct Device {
    pub(crate) inner: Arc<DeviceHandle>,
}
impl Hash for Device {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let ptr = Arc::as_ptr(&self.inner);
        ptr.hash(state);
    }
}
impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}
impl Eq for Device {}
pub(crate) struct DeviceHandle {
    pub(crate) backend: ProxyBackend,
    pub(crate) default_stream: Option<Arc<StreamHandle>>,
}
unsafe impl Send for DeviceHandle {}
unsafe impl Sync for DeviceHandle {}
impl Deref for DeviceHandle {
    type Target = ProxyBackend;
    fn deref(&self) -> &Self::Target {
        &self.backend
    }
}
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Drop for DeviceHandle {
    fn drop(&mut self) {
        if let Some(s) = &self.default_stream {
            let handle = s.handle();
            self.backend.destroy_stream(handle);
        }
    }
}
impl Device {
    pub fn create_swapchain(
        &self,
        window: &Window,
        stream: &Stream,
        width: u32,
        height: u32,
        allow_hdr: bool,
        vsync: bool,
        back_buffer_size: u32,
    ) -> backend::Result<Swapchain> {
        let handle = window.raw_window_handle();
        let window_handle = match handle {
            raw_window_handle::RawWindowHandle::UiKit(_) => todo!(),
            raw_window_handle::RawWindowHandle::AppKit(_) => todo!(),
            raw_window_handle::RawWindowHandle::Orbital(_) => todo!(),
            raw_window_handle::RawWindowHandle::Xlib(h) => h.window as u64,
            raw_window_handle::RawWindowHandle::Xcb(_h) => {
                panic!("xcb not supported, use X11 instead")
            }
            raw_window_handle::RawWindowHandle::Wayland(_h) => {
                panic!("Wayland not supported, use X11 instead")
            }
            raw_window_handle::RawWindowHandle::Drm(_) => todo!(),
            raw_window_handle::RawWindowHandle::Gbm(_) => todo!(),
            raw_window_handle::RawWindowHandle::Win32(h) => {
                h.hwnd as u64 // TODO: test this
            }
            raw_window_handle::RawWindowHandle::WinRt(h) => h.core_window as u64,
            raw_window_handle::RawWindowHandle::Web(_) => todo!(),
            raw_window_handle::RawWindowHandle::AndroidNdk(_) => todo!(),
            raw_window_handle::RawWindowHandle::Haiku(_) => todo!(),
            _ => todo!(),
        };
        self.create_swapchain_raw_handle(
            window_handle,
            stream,
            width,
            height,
            allow_hdr,
            vsync,
            back_buffer_size,
        )
    }
    pub fn create_swapchain_raw_handle(
        &self,
        window_handle: u64,
        stream: &Stream,
        width: u32,
        height: u32,
        allow_hdr: bool,
        vsync: bool,
        back_buffer_size: u32,
    ) -> backend::Result<Swapchain> {
        let swapchain = self.inner.create_swapchain(
            window_handle,
            stream.handle(),
            width,
            height,
            allow_hdr,
            vsync,
            back_buffer_size,
        )?;
        let swapchain = Swapchain {
            device: self.clone(),
            handle: Arc::new(SwapchainHandle {
                device: self.inner.clone(),
                handle: api::Swapchain(swapchain.resource.handle),
                native_handle: swapchain.resource.native_handle,
                pixel_storage: swapchain.storage,
            }),
        };
        Ok(swapchain)
    }
    pub fn create_buffer<T: Value>(&self, count: usize) -> backend::Result<Buffer<T>> {
        assert!(
            std::mem::size_of::<T>() > 0,
            "size of T must be greater than 0"
        );
        let buffer = self.inner.create_buffer(&T::type_(), count)?;
        let buffer = Buffer {
            device: self.clone(),
            handle: Arc::new(BufferHandle {
                device: self.clone(),
                handle: api::Buffer(buffer.resource.handle),
                native_handle: buffer.resource.native_handle,
            }),
            _marker: std::marker::PhantomData {},
            len: count,
        };
        Ok(buffer)
    }
    pub fn create_buffer_from_slice<T: Value>(&self, data: &[T]) -> backend::Result<Buffer<T>> {
        let buffer = self.create_buffer(data.len())?;
        buffer.view(..).copy_from(data);
        Ok(buffer)
    }
    pub fn create_buffer_from_fn<T: Value>(
        &self,
        count: usize,
        f: impl FnMut(usize) -> T,
    ) -> backend::Result<Buffer<T>> {
        let buffer = self.create_buffer(count)?;
        buffer.view(..).fill_fn(f);
        Ok(buffer)
    }
    pub fn create_bindless_array(&self, slots: usize) -> backend::Result<BindlessArray> {
        let array = self.inner.create_bindless_array(slots)?;
        Ok(BindlessArray {
            device: self.clone(),
            handle: Arc::new(BindlessArrayHandle {
                device: self.clone(),
                handle: api::BindlessArray(array.handle),
                native_handle: array.native_handle,
            }),
            modifications: RefCell::new(Vec::new()),
            slots: RefCell::new(vec![
                BindlessArraySlot {
                    buffer: None,
                    tex2d: None,
                    tex3d: None,
                };
                slots
            ]),
            pending_slots: RefCell::new(Vec::new()),
            lock: Arc::new(RawMutex::INIT),
            dirty:Cell::new(false),
        })
    }
    pub fn create_tex2d<T: IoTexel>(
        &self,
        storage: PixelStorage,
        width: u32,
        height: u32,
        mips: u32,
    ) -> backend::Result<Tex2d<T>> {
        let format = T::pixel_format(storage);
        let texture = self
            .inner
            .create_texture(format, 2, width, height, 1, mips)?;
        let handle = Arc::new(TextureHandle {
            device: self.clone(),
            handle: api::Texture(texture.handle),
            native_handle: texture.native_handle,
            format,
            levels: mips,
            width,
            height,
            depth: 1,
            storage: format.storage(),
        });
        let tex = Tex2d {
            width,
            height,
            handle,
            marker: std::marker::PhantomData {},
        };
        Ok(tex)
    }
    pub fn create_tex3d<T: IoTexel>(
        &self,
        storage: PixelStorage,
        width: u32,
        height: u32,
        depth: u32,
        mips: u32,
    ) -> backend::Result<Tex3d<T>> {
        let format = T::pixel_format(storage);
        let texture = self
            .inner
            .create_texture(format, 3, width, height, depth, mips)?;
        let handle = Arc::new(TextureHandle {
            device: self.clone(),
            handle: api::Texture(texture.handle),
            native_handle: texture.native_handle,
            format,
            levels: mips,
            width,
            height,
            depth,
            storage: format.storage(),
        });
        let tex = Tex3d {
            width,
            height,
            depth,
            handle,
            marker: std::marker::PhantomData {},
        };
        Ok(tex)
    }

    pub fn default_stream(&self) -> Stream {
        Stream {
            device: self.clone(),
            handle: self.inner.default_stream.clone().unwrap(),
        }
    }
    pub fn create_stream(&self, tag: api::StreamTag) -> backend::Result<Stream> {
        let stream = self.inner.create_stream(tag)?;
        Ok(Stream {
            device: self.clone(),
            handle: Arc::new(StreamHandle::NonDefault {
                device: self.inner.clone(),
                handle: api::Stream(stream.handle),
                native_handle: stream.native_handle,
                mutex: RawMutex::INIT,
            }),
        })
    }
    pub fn create_mesh<V: Value>(
        &self,
        vbuffer: BufferView<'_, V>,
        tbuffer: BufferView<'_, rtx::Index>,
        option: AccelOption,
    ) -> backend::Result<Mesh> {
        let mesh = self.inner.create_mesh(option)?;
        let handle = mesh.handle;
        let native_handle = mesh.native_handle;
        let mesh = Mesh {
            handle: Arc::new(MeshHandle {
                device: self.clone(),
                handle: api::Mesh(handle),
                native_handle,
            }),
            vertex_buffer: vbuffer.handle(),
            vertex_buffer_offset: vbuffer.offset * std::mem::size_of::<V>() as usize,
            vertex_buffer_size: vbuffer.len * std::mem::size_of::<V>() as usize,
            vertex_stride: std::mem::size_of::<V>() as usize,
            index_buffer: tbuffer.handle(),
            index_buffer_offset: tbuffer.offset * std::mem::size_of::<rtx::Index>() as usize,
            index_buffer_size: tbuffer.len * std::mem::size_of::<rtx::Index>() as usize,
            index_stride: std::mem::size_of::<rtx::Index>() as usize,
        };
        Ok(mesh)
    }
    pub fn create_accel(&self, option: api::AccelOption) -> backend::Result<rtx::Accel> {
        let accel = self.inner.create_accel(option)?;
        Ok(rtx::Accel {
            handle: Arc::new(rtx::AccelHandle {
                device: self.clone(),
                handle: api::Accel(accel.handle),
                native_handle: accel.native_handle,
            }),
            mesh_handles: RwLock::new(Vec::new()),
            modifications: RwLock::new(HashMap::new()),
        })
    }
    pub fn create_callable<'a, S: CallableSignature<'a, R>, R: CallableRet>(
        &self,
        f: S::Fn,
    ) -> S::Callable {
        let mut builder = KernelBuilder::new(self.clone(), false);
        let raw_callable = KernelBuildFn::build_callable(&f, &mut builder);
        S::wrap_raw_callable(raw_callable)
    }
    pub fn create_kernel<'a, S: KernelSignature<'a>>(&self, f: S::Fn) -> Result<S::Kernel> {
        let mut builder = KernelBuilder::new(self.clone(), true);
        let raw_kernel =
            KernelBuildFn::build_kernel(&f, &mut builder, KernelBuildOptions::default());
        S::wrap_raw_kernel(raw_kernel)
    }
    pub fn create_kernel_async<'a, S: KernelSignature<'a>>(&self, f: S::Fn) -> Result<S::Kernel> {
        let mut builder = KernelBuilder::new(self.clone(), true);
        let raw_kernel = KernelBuildFn::build_kernel(
            &f,
            &mut builder,
            KernelBuildOptions {
                async_compile: true,
                ..Default::default()
            },
        );
        S::wrap_raw_kernel(raw_kernel)
    }
    pub fn create_kernel_with_options<'a, S: KernelSignature<'a>>(
        &self,
        f: S::Fn,
        options: KernelBuildOptions,
    ) -> Result<S::Kernel> {
        let mut builder = KernelBuilder::new(self.clone(), true);
        let raw_kernel = KernelBuildFn::build_kernel(&f, &mut builder, options);
        S::wrap_raw_kernel(raw_kernel)
    }
}
#[macro_export]
macro_rules! fn_n_args {
    (0)=>{ dyn Fn()};
    (1)=>{ dyn Fn(_)};
    (2)=>{ dyn Fn(_,_)};
    (3)=>{ dyn Fn(_,_,_)};
    (4)=>{ dyn Fn(_,_,_,_)};
    (5)=>{ dyn Fn(_,_,_,_,_)};
    (6)=>{ dyn Fn(_,_,_,_,_,_)};
    (7)=>{ dyn Fn(_,_,_,_,_,_,_)};
    (8)=>{ dyn Fn(_,_,_,_,_,_,_,_)};
    (9)=>{ dyn Fn(_,_,_,_,_,_,_,_,_)};
    (10)=>{dyn Fn(_,_,_,_,_,_,_,_,_,_)};
    (11)=>{dyn Fn(_,_,_,_,_,_,_,_,_,_,_)};
    (12)=>{dyn Fn(_,_,_,_,_,_,_,_,_,_,_,_)};
    (13)=>{dyn Fn(_,_,_,_,_,_,_,_,_,_,_,_,_)};
    (14)=>{dyn Fn(_,_,_,_,_,_,_,_,_,_,_,_,_,_)};
    (15)=>{dyn Fn(_,_,_,_,_,_,_,_,_,_,_,_,_,_,_)};
}
#[macro_export]
macro_rules! wrap_fn {
    ($arg_count:tt, $f:expr) => {
        &$f as &fn_n_args!($arg_count)
    };
}
#[macro_export]
macro_rules! create_kernel {
    ($device:expr, $arg_count:tt, $f:expr) => {{
        let kernel: fn_n_args!($arg_count) = Box::new($f);
        $device.create_kernel(kernel)
    }};
}
pub(crate) enum StreamHandle {
    Default {
        device: Weak<DeviceHandle>,
        handle: api::Stream,
        native_handle: *mut std::ffi::c_void,
        mutex: RawMutex,
    },
    NonDefault {
        device: Arc<DeviceHandle>,
        handle: api::Stream,
        native_handle: *mut std::ffi::c_void,
        mutex: RawMutex,
    },
}
pub(crate) struct SwapchainHandle {
    pub(crate) device: Arc<DeviceHandle>,
    pub(crate) handle: api::Swapchain,
    pub(crate) native_handle: *mut std::ffi::c_void,
    pub(crate) pixel_storage: PixelStorage,
}
unsafe impl Send for SwapchainHandle {}
unsafe impl Sync for SwapchainHandle {}
impl Drop for SwapchainHandle {
    fn drop(&mut self) {
        self.device.destroy_swapchain(self.handle);
    }
}
pub struct Swapchain {
    pub(crate) handle: Arc<SwapchainHandle>,
    #[allow(dead_code)]
    pub(crate) device: Device,
}
impl Swapchain {
    #[inline]
    pub fn handle(&self) -> api::Swapchain {
        self.handle.handle
    }
    #[inline]
    pub fn native_handle(&self) -> *mut std::ffi::c_void {
        self.handle.native_handle
    }
    #[inline]
    pub fn pixel_storage(&self) -> PixelStorage {
        self.handle.pixel_storage
    }
}
pub struct Stream {
    #[allow(dead_code)]
    pub(crate) device: Device,
    pub(crate) handle: Arc<StreamHandle>,
}
impl StreamHandle {
    #[inline]
    pub(crate) fn device(&self) -> Arc<DeviceHandle> {
        match self {
            StreamHandle::Default { device, .. } => device.upgrade().unwrap(),
            StreamHandle::NonDefault { device, .. } => device.clone(),
        }
    }
    #[inline]
    pub(crate) fn handle(&self) -> api::Stream {
        match self {
            StreamHandle::Default { handle, .. } => *handle,
            StreamHandle::NonDefault { handle, .. } => *handle,
        }
    }
    #[inline]
    pub(crate) fn native_handle(&self) -> *mut std::ffi::c_void {
        match self {
            StreamHandle::Default { native_handle, .. } => *native_handle,
            StreamHandle::NonDefault { native_handle, .. } => *native_handle,
        }
    }
    #[inline]
    pub(crate) fn lock(&self) {
        match self {
            StreamHandle::Default { mutex, .. } => mutex.lock(),
            StreamHandle::NonDefault { mutex, .. } => mutex.lock(),
        }
    }
    #[inline]
    pub(crate) fn unlock(&self) {
        unsafe {
            match self {
                StreamHandle::Default { mutex, .. } => mutex.unlock(),
                StreamHandle::NonDefault { mutex, .. } => mutex.unlock(),
            }
        }
    }
}
impl Drop for StreamHandle {
    fn drop(&mut self) {
        match self {
            StreamHandle::Default { .. } => {}
            StreamHandle::NonDefault { device, handle, .. } => {
                device.destroy_stream(*handle);
            }
        }
    }
}
pub struct Scope<'a> {
    handle: Arc<StreamHandle>,
    marker: std::marker::PhantomData<&'a ()>,
    synchronized: Cell<bool>,
    resource_tracker: RefCell<ResourceTracker>,
}
impl<'a> Scope<'a> {
    #[inline]
    pub fn handle(&self) -> api::Stream {
        self.handle.handle()
    }
    #[inline]
    pub fn native_handle(&self) -> *mut std::ffi::c_void {
        match self.handle.as_ref() {
            StreamHandle::Default { native_handle, .. } => *native_handle,
            StreamHandle::NonDefault { native_handle, .. } => *native_handle,
        }
    }
    #[inline]
    pub fn synchronize(&self) -> backend::Result<()> {
        self.handle.device().synchronize_stream(self.handle())?;
        self.synchronized.set(true);
        Ok(())
    }
    #[inline]
    pub fn command_list(&self) -> CommandList<'a> {
        CommandList::<'a> {
            marker: std::marker::PhantomData {},
            commands: Vec::new(),
        }
    }
    #[inline]
    pub fn submit(&self, commands: impl IntoIterator<Item = Command<'a>>) -> backend::Result<()> {
        self.submit_with_callback(commands, || {})
    }
    fn submit_impl<F: FnOnce() + Send + 'static>(
        &self,
        commands: Vec<Command<'a>>,
        callback: F,
    ) -> backend::Result<()> {
        self.synchronized.set(false);
        let mut command_buffer = self.command_list();
        command_buffer.extend(commands);
        let commands = command_buffer.commands;
        let api_commands = commands.iter().map(|c| c.inner).collect::<Vec<_>>();
        let ctx = CommandCallbackCtx {
            commands,
            f: callback,
        };
        let ptr = Box::into_raw(Box::new(ctx));
        extern "C" fn trampoline<'a, F: FnOnce() + Send + 'static>(ptr: *mut u8) {
            let ctx = unsafe { *Box::from_raw(ptr as *mut CommandCallbackCtx<'a, F>) };
            (ctx.f)();
        }
        self.handle.device().dispatch(
            self.handle(),
            &api_commands,
            (trampoline::<F>, ptr as *mut u8),
        )
    }
    #[inline]
    pub fn submit_with_callback<F: FnOnce() + Send + 'static>(
        &self,
        commands: impl IntoIterator<Item = Command<'a>>,
        callback: F,
    ) -> backend::Result<()> {
        let mut iter = commands.into_iter();
        loop {
            let mut commands = vec![];
            let mut end = false;
            loop {
                if let Some(cmd) = iter.next() {
                    let should_break = cmd.callback.is_some();
                    commands.push(cmd);
                    if should_break {
                        break;
                    }
                } else {
                    end = true;
                    break;
                }
            }
            if commands.is_empty() {
                return Ok(());
            }
            // self.submit_impl(commands, callback)
            let cb = commands.last_mut().unwrap().callback.take();
            if end {
                if let Some(cb) = cb {
                    return self.submit_impl(commands, move || {
                        cb();
                        callback();
                    });
                } else {
                    return self.submit_impl(commands, callback);
                }
            } else {
                self.submit_impl(commands, cb.unwrap())?;
            }
        }
    }
    #[inline]
    pub fn present<T: IoTexel>(
        &self,
        swapchain: &Swapchain,
        image: &Tex2d<T>,
    ) -> backend::Result<()> {
        assert_eq!(image.handle.storage, swapchain.handle.pixel_storage);
        let mut rt = self.resource_tracker.borrow_mut();
        rt.add(swapchain.handle.clone());
        rt.add(image.handle.clone());
        self.synchronized.set(false);
        self.handle.device().present_display_in_stream(
            self.handle(),
            swapchain.handle(),
            image.handle(),
        );
        Ok(())
    }
}
impl<'a> Drop for Scope<'a> {
    fn drop(&mut self) {
        if !self.synchronized.get() {
            self.synchronize().unwrap();
        }
        self.handle.unlock();
    }
}
impl Stream {
    #[inline]
    pub fn with_scope<'a, R>(&self, f: impl FnOnce(&Scope<'a>) -> R) -> R {
        let s = self.scope();
        f(&s)
    }
    #[inline]
    pub fn scope<'a>(&self) -> Scope<'a> {
        self.handle.lock();
        Scope {
            handle: self.handle.clone(),
            marker: std::marker::PhantomData {},
            synchronized: Cell::new(false),
            resource_tracker: RefCell::new(ResourceTracker::new()),
        }
    }
    #[inline]
    pub fn handle(&self) -> api::Stream {
        self.handle.handle()
    }
    #[inline]
    pub fn native_handle(&self) -> *mut std::ffi::c_void {
        self.handle.native_handle()
    }
}
pub struct CommandList<'a> {
    marker: std::marker::PhantomData<&'a ()>,
    commands: Vec<Command<'a>>,
}
struct CommandCallbackCtx<'a, F: FnOnce() + Send + 'static> {
    #[allow(dead_code)]
    commands: Vec<Command<'a>>,
    f: F,
}

impl<'a> CommandList<'a> {
    pub fn extend<I: IntoIterator<Item = Command<'a>>>(&mut self, commands: I) {
        self.commands.extend(commands);
    }
    pub fn push(&mut self, command: Command<'a>) {
        self.commands.push(command);
    }
}

pub fn submit_default_stream_and_sync<'a, I: IntoIterator<Item = Command<'a>>>(
    device: &Device,
    commands: I,
) -> backend::Result<()> {
    let default_stream = device.default_stream();
    default_stream.with_scope(|s| {
        s.submit(commands)?;
        s.synchronize()
    })
}
pub struct Command<'a> {
    #[allow(dead_code)]
    pub(crate) inner: api::Command,
    pub(crate) marker: std::marker::PhantomData<&'a ()>,
    pub(crate) callback: Option<Box<dyn FnOnce() + Send + 'static>>,
    #[allow(dead_code)]
    pub(crate) resource_tracker: ResourceTracker,
}
pub(crate) struct AsyncShaderArtifact {
    shader: Option<backend::Result<api::CreatedShaderInfo>>, // strange naming, huh?
    name: Arc<CString>,
}
pub(crate) enum ShaderArtifact {
    Async(Arc<(Mutex<AsyncShaderArtifact>, Condvar)>),
    Sync(api::CreatedShaderInfo),
}
impl AsyncShaderArtifact {
    pub(crate) fn new(
        device: Device,
        kernel: CArc<KernelModule>,
        options: api::ShaderOption,
        name: Arc<CString>,
    ) -> Arc<(Mutex<AsyncShaderArtifact>, Condvar)> {
        let artifact = Arc::new((
            Mutex::new(AsyncShaderArtifact { shader: None, name }),
            Condvar::new(),
        ));
        {
            let artifact = artifact.clone();
            rayon::spawn(move || {
                let shader = device.inner.create_shader(&kernel, &options);
                {
                    let mut artifact = artifact.0.lock();
                    artifact.shader = Some(shader);
                }
                artifact.1.notify_all();
            });
        }
        artifact
    }
}
pub struct RawKernel {
    pub(crate) device: Device,
    pub(crate) artifact: ShaderArtifact,
    #[allow(dead_code)]
    pub(crate) resource_tracker: ResourceTracker,
}
pub struct CallableArgEncoder {
    pub(crate) args: Vec<NodeRef>,
}
impl CallableArgEncoder {
    #[inline]
    pub fn new() -> CallableArgEncoder {
        CallableArgEncoder { args: Vec::new() }
    }
    pub fn buffer<T: Value>(&mut self, buffer: &BufferVar<T>) {
        self.args.push(buffer.node);
    }
    pub fn tex2d<T: IoTexel>(&mut self, tex2d: &Tex2dVar<T>) {
        self.args.push(tex2d.node);
    }
    pub fn tex3d<T: IoTexel>(&mut self, tex3d: &Tex3dVar<T>) {
        self.args.push(tex3d.node);
    }
    pub fn bindless_array(&mut self, array: &BindlessArrayVar) {
        self.args.push(array.node);
    }
    pub fn value<T: Value>(&mut self, value: Expr<T>) {
        self.args.push(value.node());
    }
    pub fn var<T: Value>(&mut self, var: Var<T>) {
        self.args.push(var.node());
    }
}
pub struct KernelArgEncoder {
    pub(crate) args: Vec<api::Argument>,
    pub(crate) uniform_data: Vec<Box<[u8]>>,
}
impl KernelArgEncoder {
    pub fn new() -> KernelArgEncoder {
        KernelArgEncoder {
            args: Vec::new(),
            uniform_data: vec![],
        }
    }
    pub fn uniform<T: Value>(&mut self, value: T) {
        let mut data_u8 = unsafe {
            let layout = std::alloc::Layout::new::<T>();
            let ptr = std::alloc::alloc(layout);
            let slice = std::slice::from_raw_parts_mut(ptr as *mut u8, layout.size());
            Box::from_raw(slice)
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                &value as *const _ as *const u8,
                data_u8.as_mut_ptr(),
                std::mem::size_of::<T>(),
            )
        }
        self.args.push(api::Argument::Uniform(api::UniformArgument {
            data: data_u8.as_ptr(),
            size: data_u8.len(),
        }));
        self.uniform_data.push(data_u8);
    }
    pub fn buffer<T: Value>(&mut self, buffer: &Buffer<T>) {
        self.args.push(api::Argument::Buffer(api::BufferArgument {
            buffer: buffer.handle.handle,
            offset: 0,
            size: buffer.len * std::mem::size_of::<T>(),
        }));
    }
    pub fn buffer_view<T: Value>(&mut self, buffer: &BufferView<T>) {
        self.args.push(api::Argument::Buffer(api::BufferArgument {
            buffer: buffer.handle(),
            offset: buffer.offset,
            size: buffer.len * std::mem::size_of::<T>(),
        }));
    }
    pub fn tex2d<T: IoTexel>(&mut self, tex: &Tex2dView<T>) {
        self.args.push(api::Argument::Texture(api::TextureArgument {
            texture: tex.handle(),
            level: tex.level,
        }));
    }
    pub fn tex3d<T: IoTexel>(&mut self, tex: &Tex3dView<T>) {
        self.args.push(api::Argument::Texture(api::TextureArgument {
            texture: tex.handle(),
            level: tex.level,
        }));
    }
    pub fn bindless_array(&mut self, array: &BindlessArray) {
        self.args
            .push(api::Argument::BindlessArray(array.handle.handle));
    }
    pub fn accel(&mut self, accel: &Accel) {
        self.args.push(api::Argument::Accel(accel.handle.handle));
    }
}
pub trait CallableArg {
    type Parameter: CallableParameter;
    fn encode(&self, encoder: &mut CallableArgEncoder);
}
// impl<T:Value> CallableArg for BufferVar<T> {
//     type Parameter = BufferVar<T>;
//     fn encode(&self, encoder: &mut CallableArgEncoder) {
//         encoder.buffer(self);
//     }
// }
pub trait KernelArg {
    type Parameter: KernelParameter;
    fn encode(&self, encoder: &mut KernelArgEncoder);
}

impl<T: Value> KernelArg for Buffer<T> {
    type Parameter = BufferVar<T>;
    fn encode(&self, encoder: &mut KernelArgEncoder) {
        encoder.buffer(self);
    }
}
impl<T: Value> KernelArg for T {
    type Parameter = Expr<T>;
    fn encode(&self, encoder: &mut KernelArgEncoder) {
        encoder.uniform(*self)
    }
}
impl<'a, T: Value> KernelArg for BufferView<'a, T> {
    type Parameter = BufferVar<T>;
    fn encode(&self, encoder: &mut KernelArgEncoder) {
        encoder.buffer_view(self);
    }
}
impl<T: IoTexel> KernelArg for Tex2d<T> {
    type Parameter = Tex2dVar<T>;
    fn encode(&self, encoder: &mut KernelArgEncoder) {
        encoder.tex2d(&self.view(0));
    }
}
impl<T: IoTexel> KernelArg for Tex3d<T> {
    type Parameter = Tex3dVar<T>;
    fn encode(&self, encoder: &mut KernelArgEncoder) {
        encoder.tex3d(&self.view(0));
    }
}
impl<'a, T: IoTexel> KernelArg for Tex2dView<'a, T> {
    type Parameter = Tex2dVar<T>;
    fn encode(&self, encoder: &mut KernelArgEncoder) {
        encoder.tex2d(self);
    }
}
impl<'a, T: IoTexel> KernelArg for Tex3dView<'a, T> {
    type Parameter = Tex3dVar<T>;
    fn encode(&self, encoder: &mut KernelArgEncoder) {
        encoder.tex3d(self);
    }
}
impl KernelArg for BindlessArray {
    type Parameter = BindlessArrayVar;
    fn encode(&self, encoder: &mut KernelArgEncoder) {
        encoder.bindless_array(self);
    }
}
impl KernelArg for Accel {
    type Parameter = rtx::AccelVar;
    fn encode(&self, encoder: &mut KernelArgEncoder) {
        encoder.accel(self)
    }
}
macro_rules! impl_kernel_arg_for_tuple {
    ()=>{
        impl KernelArg for () {
            type Parameter = ();
            fn encode(&self, _: &mut KernelArgEncoder) { }
        }
    };
    ($first:ident  $($rest:ident) *) => {
        impl<$first:KernelArg, $($rest: KernelArg),*> KernelArg for ($first, $($rest,)*) {
            type Parameter = ($first::Parameter, $($rest::Parameter),*);
            #[allow(non_snake_case)]
            fn encode(&self, encoder: &mut KernelArgEncoder) {
                let ($first, $($rest,)*) = self;
                $first.encode(encoder);
                $($rest.encode(encoder);)*
            }
        }
        impl_kernel_arg_for_tuple!($($rest)*);
    };

}
impl_kernel_arg_for_tuple!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);

impl RawKernel {
    fn unwrap(&self) -> api::Shader {
        match &self.artifact {
            ShaderArtifact::Sync(shader) => api::Shader(shader.resource.handle),
            ShaderArtifact::Async(artifact) => {
                let condvar = &artifact.1;
                let mut artifact = artifact.0.lock();
                if let Some(shader) = &artifact.shader {
                    return api::Shader(shader.as_ref().unwrap().resource.handle);
                }
                condvar.wait(&mut artifact);
                api::Shader(
                    artifact
                        .shader
                        .as_ref()
                        .unwrap()
                        .as_ref()
                        .unwrap()
                        .resource
                        .handle,
                )
            }
        }
    }
    pub fn dispatch_async(&self, args: KernelArgEncoder, dispatch_size: [u32; 3]) -> Command {
        let mut rt = ResourceTracker::new();
        rt.add(Arc::new(args.uniform_data));
        let args = args.args;
        let args = Arc::new(args);
        rt.add(args.clone());
        Command {
            inner: api::Command::ShaderDispatch(api::ShaderDispatchCommand {
                shader: self.unwrap(),
                args: args.as_ptr(),
                args_count: args.len(),
                dispatch_size,
            }),
            marker: std::marker::PhantomData,
            resource_tracker: rt,
            callback: None,
        }
    }
    pub fn dispatch(&self, args: KernelArgEncoder, dispatch_size: [u32; 3]) -> backend::Result<()> {
        submit_default_stream_and_sync(&self.device, vec![self.dispatch_async(args, dispatch_size)])
    }
}
pub trait CallableRet {}
impl CallableRet for () {}
impl<T: Value> CallableRet for T {}
pub struct Callable<T: KernelArg, R: CallableRet> {
    #[allow(dead_code)]
    pub(crate) inner: ir::CallableModuleRef,
    pub(crate) _marker: std::marker::PhantomData<(T, R)>,
}
pub struct Kernel<T: KernelArg> {
    pub(crate) inner: RawKernel,
    pub(crate) _marker: std::marker::PhantomData<T>,
}
impl<T: KernelArg> Kernel<T> {
    pub fn cache_dir(&self) -> Option<PathBuf> {
        let handle = self.inner.unwrap();
        let device = &self.inner.device;
        device.inner.shader_cache_dir(handle)
    }
}
pub trait AsKernelArg<T: KernelArg>: KernelArg {}
impl<T: Value> AsKernelArg<T> for T {}
impl<T: Value> AsKernelArg<Buffer<T>> for Buffer<T> {}
impl<'a, T: Value> AsKernelArg<Buffer<T>> for BufferView<'a, T> {}
impl<'a, T: Value> AsKernelArg<BufferView<'a, T>> for BufferView<'a, T> {}
impl<'a, T: Value> AsKernelArg<BufferView<'a, T>> for Buffer<T> {}
impl<'a, T: IoTexel> AsKernelArg<Tex2d<T>> for Tex2dView<'a, T> {}
impl<'a, T: IoTexel> AsKernelArg<Tex3d<T>> for Tex3dView<'a, T> {}
impl<'a, T: IoTexel> AsKernelArg<Tex3dView<'a, T>> for Tex2dView<'a, T> {}
impl<'a, T: IoTexel> AsKernelArg<Tex3dView<'a, T>> for Tex3dView<'a, T> {}
impl<'a, T: IoTexel> AsKernelArg<Tex3dView<'a, T>> for Tex2d<T> {}
impl<'a, T: IoTexel> AsKernelArg<Tex3dView<'a, T>> for Tex3d<T> {}
impl<T: IoTexel> AsKernelArg<Tex2d<T>> for Tex2d<T> {}
impl<T: IoTexel> AsKernelArg<Tex3d<T>> for Tex3d<T> {}
impl AsKernelArg<BindlessArray> for BindlessArray {}
impl AsKernelArg<Accel> for Accel {}
macro_rules! impl_call_for_callable {
    ($first:ident  $($rest:ident)*) => {
        impl <R:CallableRet, $first:KernelArg, $($rest: KernelArg),*> Callable<($first, $($rest,)*), R> {
            #[allow(non_snake_case)]
            pub fn call(&self, $first:&impl AsKernelArg<$first>, $($rest:&impl AsKernelArg<$rest>),*) -> ->Expr<R> {
                let mut encoder = KernelArgEncoder::new();
                $first.encode(&mut encoder);
                $($rest.encode(&mut encoder);)*
                self.inner.dispatch(encoder, dispatch_size)
            }
        }
        impl_dispatch_for_kernel!($($rest)*);
   };
   ()=>{
        impl<R:CallableRet> Callable<(), R> {
            pub fn call(&self)->Expr<R> {

            }
        }
    }
}
macro_rules! impl_dispatch_for_kernel {

   ($first:ident  $($rest:ident)*) => {
        impl <$first:KernelArg, $($rest: KernelArg),*> Kernel<($first, $($rest,)*)> {
            #[allow(non_snake_case)]
            pub fn dispatch(&self, dispatch_size: [u32; 3], $first:&impl AsKernelArg<$first>, $($rest:&impl AsKernelArg<$rest>),*) -> backend::Result<()> {
                let mut encoder = KernelArgEncoder::new();
                $first.encode(&mut encoder);
                $($rest.encode(&mut encoder);)*
                self.inner.dispatch(encoder, dispatch_size)
            }
            #[allow(non_snake_case)]
            pub fn dispatch_async<'a>(
                &'a self,
                dispatch_size: [u32; 3], $first: &impl AsKernelArg<$first>, $($rest:&impl AsKernelArg<$rest>),*
            ) -> Command<'a> {
                let mut encoder = KernelArgEncoder::new();
                $first.encode(&mut encoder);
                $($rest.encode(&mut encoder);)*
                self.inner.dispatch_async(encoder, dispatch_size)
            }
        }
        impl_dispatch_for_kernel!($($rest)*);
   };
   ()=>{
    impl Kernel<()> {
        pub fn dispatch(&self, dispatch_size: [u32; 3]) -> backend::Result<()> {
            self.inner.dispatch(KernelArgEncoder::new(), dispatch_size)
        }
        pub fn dispatch_async<'a>(
            &'a self,
            dispatch_size: [u32; 3],
        ) -> Command<'a> {
            self.inner.dispatch_async(KernelArgEncoder::new(), dispatch_size)
        }
    }
}
}
impl_dispatch_for_kernel!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);
