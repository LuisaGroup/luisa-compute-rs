use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::CString;
use std::hash::Hash;
use std::ops::Deref;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Weak};

use parking_lot::lock_api::RawMutex as RawMutexTrait;
use parking_lot::{Condvar, Mutex, RawMutex, RwLock};

use raw_window_handle::HasRawWindowHandle;
use winit::window::Window;

use crate::internal_prelude::*;
use crate::lang::soa::SoaBuffer;
use ir::{
    CallableModule, CallableModuleRef, Capture, CpuCustomOp, KernelModule, Module, ModuleFlags,
    ModuleKind, ModulePools,
};

use crate::backend::Backend;
use crate::rtx;
use crate::rtx::{Accel, Mesh, MeshHandle, ProceduralPrimitiveHandle};

use api::AccelOption;
pub use luisa_compute_api_types as api;
use luisa_compute_backend::proxy::ProxyBackend;

mod kernel;

pub use kernel::*;

#[derive(Clone)]
pub struct Device {
    pub(crate) inner: Arc<DeviceHandle>,
}
#[derive(Clone)]
pub struct WeakDevice {
    pub(crate) inner: Weak<DeviceHandle>,
}
impl WeakDevice {
    pub fn new(device: &Device) -> Self {
        Self {
            inner: Arc::downgrade(&device.inner),
        }
    }
    pub fn upgrade(&self) -> Option<Device> {
        self.inner.upgrade().map(|inner| Device { inner })
    }
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
    #[allow(dead_code)]
    pub(crate) ctx: Arc<crate::backend::Context>,
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
    pub fn query(&self, name: &str) -> Option<String> {
        self.inner.query(name)
    }
    pub fn name(&self) -> String {
        self.query("device_name").unwrap_or("unknown".to_string())
    }
    pub fn create_swapchain(
        &self,
        window: &Window,
        stream: &Stream,
        width: u32,
        height: u32,
        allow_hdr: bool,
        vsync: bool,
        back_buffer_size: u32,
    ) -> Swapchain {
        let handle = window.raw_window_handle();
        let window_handle = match handle {
            raw_window_handle::RawWindowHandle::UiKit(h) => h.ui_window as u64,
            raw_window_handle::RawWindowHandle::AppKit(h) => h.ns_window as u64,
            raw_window_handle::RawWindowHandle::Orbital(_) => todo!(),
            raw_window_handle::RawWindowHandle::Xlib(h) => h.window as u64,
            raw_window_handle::RawWindowHandle::Xcb(h) => h.window as u64,
            raw_window_handle::RawWindowHandle::Wayland(_h) => {
                panic!("Wayland not supported, use X11 instead")
            }
            raw_window_handle::RawWindowHandle::Drm(_) => todo!(),
            raw_window_handle::RawWindowHandle::Gbm(_) => todo!(),
            raw_window_handle::RawWindowHandle::Win32(h) => h.hwnd as u64,
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
    ) -> Swapchain {
        let swapchain = self.inner.create_swapchain(
            window_handle,
            stream.handle(),
            width,
            height,
            allow_hdr,
            vsync,
            back_buffer_size,
        );
        let swapchain = Swapchain {
            device: self.clone(),
            handle: Arc::new(SwapchainHandle {
                device: self.inner.clone(),
                handle: api::Swapchain(swapchain.resource.handle),
                native_handle: swapchain.resource.native_handle,
                pixel_storage: swapchain.storage,
            }),
        };
        swapchain
    }
    pub fn create_byte_buffer(&self, len: usize) -> Buffer<u8> {
        let name = self.name();
        if name == "dx" {
            assert!(
                len < u32::MAX as usize,
                "numer of bytes must be less than u32::MAX on dx"
            );
        }
        let buffer = self.inner.create_buffer(&Type::void(), len);
        let buffer = Buffer {
            device: self.clone(),
            handle: Arc::new(BufferHandle {
                device: self.clone(),
                handle: api::Buffer(buffer.resource.handle),
                native_handle: buffer.resource.native_handle,
            }),
            len,
            _marker: PhantomData,
        };
        buffer
    }
    pub fn create_soa_buffer<T: Value>(&self, count: usize) -> SoaBuffer<T> {
        // let inner = self.create_byte_buffer(len)
        todo!()
    }
    pub fn create_buffer<T: Value>(&self, count: usize) -> Buffer<T> {
        let name = self.name();
        if name == "dx" {
            assert!(
                std::mem::align_of::<T>() >= 4,
                "T must be aligned to 4 bytes on dx"
            );
            assert!(
                count < u32::MAX as usize,
                "count must be less than u32::MAX on dx"
            );
        }
        assert!(
            std::mem::size_of::<T>() > 0,
            "size of T must be greater than 0"
        );
        let buffer = self.inner.create_buffer(&T::type_(), count);
        let buffer = Buffer {
            device: self.clone(),
            handle: Arc::new(BufferHandle {
                device: self.clone(),
                handle: api::Buffer(buffer.resource.handle),
                native_handle: buffer.resource.native_handle,
            }),
            _marker: PhantomData {},
            len: count,
        };
        buffer
    }
    pub fn create_buffer_from_slice<T: Value>(&self, data: &[T]) -> Buffer<T> {
        let buffer = self.create_buffer(data.len());
        buffer.view(..).copy_from(data);
        buffer
    }
    pub fn create_buffer_from_fn<T: Value>(
        &self,
        count: usize,
        f: impl FnMut(usize) -> T,
    ) -> Buffer<T> {
        let buffer = self.create_buffer(count);
        buffer.view(..).fill_fn(f);
        buffer
    }
    pub fn create_buffer_heap<T: Value>(&self, slots: usize) -> BufferHeap<T> {
        let array = self.create_bindless_array(slots);
        BufferHeap {
            inner: array,
            _marker: PhantomData {},
        }
    }
    pub fn create_bindless_array(&self, slots: usize) -> BindlessArray {
        assert!(slots > 0, "slots must be greater than 0");
        let array = self.inner.create_bindless_array(slots);
        BindlessArray {
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
            dirty: Cell::new(false),
        }
    }
    pub fn create_tex2d<T: IoTexel>(
        &self,
        storage: PixelStorage,
        width: u32,
        height: u32,
        mips: u32,
    ) -> Tex2d<T> {
        let format = T::pixel_format(storage);
        let texture = self
            .inner
            .create_texture(format, 2, width, height, 1, mips, true);
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
            marker: PhantomData {},
        };
        tex
    }
    pub fn create_tex3d<T: IoTexel>(
        &self,
        storage: PixelStorage,
        width: u32,
        height: u32,
        depth: u32,
        mips: u32,
    ) -> Tex3d<T> {
        let format = T::pixel_format(storage);
        let texture = self
            .inner
            .create_texture(format, 3, width, height, depth, mips, true);
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
            marker: PhantomData {},
        };
        tex
    }

    pub fn default_stream(&self) -> Stream {
        Stream {
            device: self.clone(),
            handle: self.inner.default_stream.clone().unwrap(),
        }
    }
    pub fn create_stream(&self, tag: api::StreamTag) -> Stream {
        let stream = self.inner.create_stream(tag);
        Stream {
            device: self.clone(),
            handle: Arc::new(StreamHandle::NonDefault {
                device: self.inner.clone(),
                handle: api::Stream(stream.handle),
                native_handle: stream.native_handle,
            }),
        }
    }
    pub fn create_event(&self) -> Event {
        let event = self.inner.create_event();
        Event {
            handle: Arc::new(EventHandle {
                device: self.clone(),
                handle: api::Event(event.handle),
                native_handle: event.native_handle,
            }),
        }
    }
    pub fn create_procedural_primitive(
        &self,
        aabb_buffer: BufferView<'_, rtx::Aabb>,
        option: AccelOption,
    ) -> rtx::ProceduralPrimitive {
        let primitive = self.inner.create_procedural_primitive(option);
        rtx::ProceduralPrimitive {
            handle: Arc::new(ProceduralPrimitiveHandle {
                device: self.clone(),
                handle: api::ProceduralPrimitive(primitive.handle),
                native_handle: primitive.native_handle,
                aabb_buffer: aabb_buffer.buffer.handle.clone(),
            }),
            aabb_buffer: aabb_buffer.handle(),
            aabb_buffer_offset: aabb_buffer.offset * std::mem::size_of::<rtx::Aabb>() as usize,
            aabb_buffer_count: aabb_buffer.len,
        }
    }
    pub fn create_mesh<V: Value>(
        &self,
        vbuffer: BufferView<'_, V>,
        tbuffer: BufferView<'_, rtx::Index>,
        option: AccelOption,
    ) -> Mesh {
        let mesh = self.inner.create_mesh(option);
        let handle = mesh.handle;
        let native_handle = mesh.native_handle;
        let mesh = Mesh {
            handle: Arc::new(MeshHandle {
                device: self.clone(),
                handle: api::Mesh(handle),
                native_handle,
                vbuffer: vbuffer.buffer.handle.clone(),
                ibuffer: tbuffer.buffer.handle.clone(),
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
        mesh
    }
    pub fn create_accel(&self, option: api::AccelOption) -> rtx::Accel {
        let accel = self.inner.create_accel(option);
        rtx::Accel {
            handle: Arc::new(rtx::AccelHandle {
                device: self.clone(),
                handle: api::Accel(accel.handle),
                native_handle: accel.native_handle,
            }),
            instance_handles: RwLock::new(Vec::new()),
            modifications: RwLock::new(HashMap::new()),
        }
    }

    // pub fn create_callable<S: CallableSignature>(
    //     &self,
    //     f: impl CallableBuildFn<S>,
    // ) -> Callable<S> {
    //     let mut builder = KernelBuilder::new(Some(self.clone()), false);
    //     let raw_callable = CallableBuildFn::build_callable(&f, None, &mut builder);
    //     Callable {
    //         inner: raw_callable,
    //         _marker: PhantomData,
    //     }
    // }

    pub fn create_kernel<'a, S: KernelSignature2<'a>>(&self, f: S::Fn) -> Kernel<S> {
        let mut builder = KernelBuilder::new(Some(self.clone()), true);
        let k = KernelBuildFn::build_kernel(&f, &mut builder);
        self.compile_kernel_def(&k)
    }

    pub fn create_kernel_async<'a, S: KernelSignature2<'a>>(&self, f: S::Fn) -> Kernel<S> {
        let mut builder = KernelBuilder::new(Some(self.clone()), true);
        let k = KernelBuildFn::build_kernel(&f, &mut builder);
        self.compile_kernel_def_async(&k)
    }

    pub fn create_kernel_with_options<'a, S: KernelSignature2<'a>>(
        &self,
        options: KernelBuildOptions,
        f: S::Fn,
    ) -> Kernel<S> {
        let mut builder = KernelBuilder::new(Some(self.clone()), true);
        let k = KernelBuildFn::build_kernel(&f, &mut builder);
        self.compile_kernel_def_with_options(&k, options)
    }
    /// Compile a [`KernelDef`] into a [`Kernel`]. See [`Kernel`] for more
    /// details on kernel creation
    pub fn compile_kernel_def<S: KernelSignature>(&self, k: &KernelDef<S>) -> Kernel<S> {
        self.compile_kernel_def_with_options(k, KernelBuildOptions::default())
    }

    /// Compile a [`KernelDef`] into a [`Kernel`] asynchronously. See [`Kernel`]
    /// for more details on kernel creation
    pub fn compile_kernel_def_async<S: KernelSignature>(&self, k: &KernelDef<S>) -> Kernel<S> {
        self.compile_kernel_def_with_options(
            k,
            KernelBuildOptions {
                async_compile: true,
                ..Default::default()
            },
        )
    }

    /// Compile a [`KernelDef`] into a [`Kernel`] with options. See [`Kernel`]
    /// for more details on kernel creation
    pub fn compile_kernel_def_with_options<S: KernelSignature>(
        &self,
        k: &KernelDef<S>,
        options: KernelBuildOptions,
    ) -> Kernel<S> {
        let name = options.name.unwrap_or("".to_string());
        let name = Arc::new(CString::new(name).unwrap());
        let shader_options = api::ShaderOption {
            enable_cache: options.enable_cache,
            enable_fast_math: options.enable_fast_math,
            enable_debug_info: options.enable_debug_info,
            compile_only: false,
            name: name.as_ptr(),
        };
        let module = k.inner.module.clone();
        let artifact = if options.async_compile {
            ShaderArtifact::Async(AsyncShaderArtifact::new(
                self.clone(),
                module.clone(),
                shader_options,
                name,
            ))
        } else {
            ShaderArtifact::Sync(self.inner.create_shader(&module, &shader_options))
        };
        Kernel {
            inner: Arc::new(RawKernel {
                device: self.clone(),
                artifact,
                module,
                resource_tracker: k.inner.resource_tracker.clone(),
            }),
            _marker: PhantomData {},
        }
    }
}
pub(crate) enum StreamHandle {
    Default {
        device: Weak<DeviceHandle>,
        handle: api::Stream,
        native_handle: *mut std::ffi::c_void,
    },
    NonDefault {
        device: Arc<DeviceHandle>,
        handle: api::Stream,
        native_handle: *mut std::ffi::c_void,
    },
}
unsafe impl Send for StreamHandle {}
unsafe impl Sync for StreamHandle {}

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

/// Synchronization primitives between streams, resemble timeline semaphores.
/// `scope.signal(event, ticket)` signals the event with a ticket.
/// `scope.wait(event, ticket)` waits until the event is signaled with a ticket
#[derive(Clone)]
pub struct Event {
    pub(crate) handle: Arc<EventHandle>,
}

impl Event {
    #[inline]
    pub fn handle(&self) -> api::Event {
        self.handle.handle
    }
    #[inline]
    pub fn native_handle(&self) -> *mut std::ffi::c_void {
        self.handle.native_handle
    }
    #[inline]
    pub fn synchronize(&self, ticket: u64) {
        self.handle
            .device
            .inner
            .synchronize_event(self.handle.handle, ticket);
    }
    #[inline]
    pub fn is_completed(&self, ticket: u64) -> bool {
        self.handle
            .device
            .inner
            .is_event_completed(self.handle.handle, ticket)
    }
}

pub(crate) struct EventHandle {
    pub(crate) device: Device,
    handle: api::Event,
    native_handle: *mut std::ffi::c_void,
}
unsafe impl Send for EventHandle {}
unsafe impl Sync for EventHandle {}

impl Drop for EventHandle {
    fn drop(&mut self) {
        self.device.inner.destroy_event(self.handle);
    }
}

/// A [`Stream`] is a sequence of commands that are executed in order.
/// Commands are submitted to a stream via [`Scope`].
/// There is a default stream attached to each [`Device`], which can be accessed
/// via [`Device::default_stream`].
/// All synchronous api implicitly use the default stream.
/// To create a new stream, use [`Device::create_stream`].
/// Multiple stream executes in parallel, but commands within a stream are
/// executed in order.
/// To synchronize between streams, use [`Event`].
pub struct Stream {
    #[allow(dead_code)]
    pub(crate) device: Device,
    pub(crate) handle: Arc<StreamHandle>,
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

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

/// Scope<'a> represents the lifetime of command execution.
/// Commands can only be submitted within a scope.
/// When scope drops, the underlying [`Stream`] is autoatically synchronized.
///
/// To use a scope, the easiest way is to use [`Stream::with_scope`].
///
/// To create a scope with manually assigned lifetime, use [`Stream::scope`].
/// For example, `let scope:Scope<'static> = stream.scope();` creates a scope that
/// accepts commands with `'static` lifetime.
///
/// To prevent a scope from synchronizing on drop, use [`Scope::detach`].Note this is only
/// available for `Scope<'static>` as certain commands such as `copy_to_async<'a>` requires
/// synchronization.
pub struct Scope<'a> {
    handle: Arc<StreamHandle>,
    marker: PhantomData<&'a ()>,
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
    pub fn synchronize(&self) -> &Self {
        self.handle.device().synchronize_stream(self.handle());
        self.synchronized.set(true);
        self
    }
    #[inline]
    pub fn submit<'cmd>(&self, commands: impl IntoIterator<Item = Command<'cmd, 'a>>) -> &Self {
        self.submit_with_callback(commands, || {})
    }
    fn submit_impl<'cmd, F: FnOnce() + Send + 'static>(
        &self,
        commands: Vec<Command<'cmd, 'a>>,
        callback: F,
    ) {
        self.synchronized.set(false);
        let api_commands = commands.iter().map(|c| c.inner).collect::<Vec<_>>();
        let ctx = CommandCallbackCtx {
            commands,
            f: callback,
        };
        let ptr = Box::into_raw(Box::new(ctx));
        extern "C" fn trampoline<'a, F: FnOnce() + Send + 'static>(ptr: *mut u8) {
            let ctx = unsafe { *Box::from_raw(ptr as *mut CommandCallbackCtx<'static, 'a, F>) };
            (ctx.f)();
        }
        self.handle.device().dispatch(
            self.handle(),
            &api_commands,
            (trampoline::<F>, ptr as *mut u8),
        )
    }
    #[inline]
    pub fn submit_with_callback<'cmd, F: FnOnce() + Send + 'static>(
        &self,
        commands: impl IntoIterator<Item = Command<'cmd, 'a>>,
        callback: F,
    ) -> &Self {
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
                return self;
            }
            // self.submit_impl(commands, callback)
            let cb = commands.last_mut().unwrap().callback.take();
            if end {
                if let Some(cb) = cb {
                    self.submit_impl(commands, move || {
                        cb();
                        callback();
                    });
                    return self;
                } else {
                    self.submit_impl(commands, callback);
                    return self;
                }
            } else {
                self.submit_impl(commands, cb.unwrap());
            }
        }
    }
    #[inline]
    pub fn wait(&self, event: &Event, ticket: u64) -> &Self {
        self.handle
            .device()
            .wait_event(event.handle(), self.handle(), ticket);
        self
    }
    #[inline]
    pub fn signal(&self, event: &Event, ticket: u64) -> &Self {
        self.handle
            .device()
            .signal_event(event.handle(), self.handle(), ticket);
        self
    }
    #[inline]
    pub fn present<T: IoTexel>(&self, swapchain: &Swapchain, image: &Tex2d<T>) -> &Self {
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
        self
    }
}
impl Scope<'static> {
    pub fn detach(self) {
        self.synchronized.set(true);
    }
}
impl<'a> Drop for Scope<'a> {
    fn drop(&mut self) {
        if !self.synchronized.get() {
            self.synchronize();
        }
    }
}

impl Stream {
    #[inline]
    pub fn with_scope<'a, R>(&self, f: impl FnOnce(&Scope<'a>) -> R) -> R {
        let s = self.scope();
        f(&s)
    }

    /// Creates a new scope.
    /// See [`Scope`] for more details.
    #[inline]
    pub fn scope<'a>(&self) -> Scope<'a> {
        Scope {
            handle: self.handle.clone(),
            marker: PhantomData {},
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

struct CommandCallbackCtx<'cmd, 'scope, F: FnOnce() + Send + 'static> {
    #[allow(dead_code)]
    commands: Vec<Command<'cmd, 'scope>>,
    f: F,
}

pub fn submit_default_stream_and_sync<
    'cmd,
    'scope,
    I: IntoIterator<Item = Command<'cmd, 'scope>>,
>(
    device: &Device,
    commands: I,
) {
    let default_stream = device.default_stream();
    default_stream.with_scope(|s| {
        s.submit(commands);
        s.synchronize();
    })
}

/// 'from_data is the lifetime of the data that is copied from
/// 'to_data is the lifetime of the data that is copied to. It is also the lifetime of [`Scope<'a>`]
/// Commands are created by resources and submitted to a [`Scope<'a>`] via `scope.submit` and `scope.submit_with_callback`.
#[must_use]
pub struct Command<'from_data, 'to_data> {
    #[allow(dead_code)]
    pub(crate) inner: api::Command,
    // is this really necessary?
    pub(crate) marker: PhantomData<(&'from_data (), &'to_data ())>,
    pub(crate) callback: Option<Box<dyn FnOnce() + Send + 'static>>,
    #[allow(dead_code)]
    pub(crate) resource_tracker: ResourceTracker,
}

impl<'cmd, 'scope> Command<'cmd, 'scope> {
    pub unsafe fn lift(self) -> Command<'static, 'static> {
        Command {
            inner: self.inner,
            marker: PhantomData {},
            callback: self.callback,
            resource_tracker: self.resource_tracker,
        }
    }
}

pub(crate) struct AsyncShaderArtifact {
    shader: Option<api::CreatedShaderInfo>,
    // strange naming, huh?
    #[allow(dead_code)]
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
    pub(crate) module: CArc<KernelModule>,
}
impl Drop for RawKernel {
    fn drop(&mut self) {
        let shader = self.unwrap();
        self.device.inner.destroy_shader(shader);
    }
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
    pub fn byte_buffer(&mut self, buffer: &ByteBufferVar) {
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
    pub fn var(&mut self, value: impl NodeLike) {
        self.args.push(value.node());
    }
    pub fn accel(&mut self, accel: &rtx::AccelVar) {
        self.args.push(accel.node);
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
    pub fn byte_buffer(&mut self, buffer: &ByteBuffer) {
        self.args.push(api::Argument::Buffer(api::BufferArgument {
            buffer: buffer.handle.handle,
            offset: 0,
            size: buffer.len,
        }));
    }
    pub fn byte_buffer_view(&mut self, buffer: &ByteBufferView) {
        self.args.push(api::Argument::Buffer(api::BufferArgument {
            buffer: buffer.handle(),
            offset: buffer.offset,
            size: buffer.len,
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
// impl KernelArg for ByteBuffer {
//     type Parameter = ByteBufferVar;
//     fn encode(&self, encoder: &mut KernelArgEncoder) {
//         encoder.byte_buffer(self);
//     }
// }
// impl<'a> KernelArg for ByteBufferView<'a> {
//     type Parameter = ByteBufferVar;
//     fn encode(&self, encoder: &mut KernelArgEncoder) {
//         encoder.byte_buffer_view(self);
//     }
// }
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
    ($first:ident  $($Ts:ident) *) => {
        impl<$first:KernelArg, $($Ts: KernelArg),*> KernelArg for ($first, $($Ts,)*) {
            type Parameter = ($first::Parameter, $($Ts::Parameter),*);
            #[allow(non_snake_case)]
            fn encode(&self, encoder: &mut KernelArgEncoder) {
                let ($first, $($Ts,)*) = self;
                $first.encode(encoder);
                $($Ts.encode(encoder);)*
            }
        }
        impl_kernel_arg_for_tuple!($($Ts)*);
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
                    return api::Shader(shader.resource.handle);
                }
                condvar.wait(&mut artifact);
                api::Shader(artifact.shader.as_ref().unwrap().resource.handle)
            }
        }
    }

    pub fn dispatch_async(
        self: &Arc<Self>,
        args: KernelArgEncoder,
        dispatch_size: [u32; 3],
    ) -> Command<'static, 'static> {
        let mut rt = ResourceTracker::new();
        rt.add(Arc::new(args.uniform_data));
        rt.add(self.clone());
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
            marker: PhantomData,
            resource_tracker: rt,
            callback: None,
        }
    }
    pub fn dispatch(self: &Arc<Self>, args: KernelArgEncoder, dispatch_size: [u32; 3]) {
        submit_default_stream_and_sync(&self.device, vec![self.dispatch_async(args, dispatch_size)])
    }
}

pub struct Callable<S: CallableSignature> {
    #[allow(dead_code)]
    pub(crate) inner: RawCallable,
    pub(crate) _marker: PhantomData<S>,
}
pub(crate) struct DynCallableInner<S: CallableSignature> {
    builder: Box<dyn Fn(std::rc::Rc<dyn Any>, &mut KernelBuilder) -> Callable<S>>,
    callables: Vec<Callable<S>>,
}
pub struct DynCallable<S: CallableSignature> {
    #[allow(dead_code)]
    pub(crate) inner: RefCell<DynCallableInner<S>>,
    pub(crate) device: Device,
    pub(crate) init_once: bool,
}
impl<S: CallableSignature> DynCallable<S> {
    pub(crate) fn _new(
        device: Device,
        init_once: bool,
        builder: Box<dyn Fn(std::rc::Rc<dyn Any>, &mut KernelBuilder) -> Callable<S>>,
    ) -> Self {
        Self {
            device,
            inner: RefCell::new(DynCallableInner {
                builder,
                callables: Vec::new(),
            }),
            init_once,
        }
    }
    fn call_impl(&self, args: std::rc::Rc<dyn Any>, nodes: &[NodeRef]) -> S::Ret {
        RECORDER.with(|r| {
            if let Some(device) = r.borrow().device.as_ref() {
                assert!(
                    Arc::ptr_eq(&device.inner.upgrade().unwrap(), &self.device.inner),
                    "Callable created on a different device than the one it is called on"
                );
            }
        });
        let mut inner = self.inner.borrow_mut();

        {
            let callables = &mut inner.callables;
            for c in callables {
                if crate::lang::__check_callable(&c.inner.module, nodes) {
                    return CallableRet::_from_return(crate::lang::__invoke_callable(
                        &c.inner.module,
                        nodes,
                    ));
                }
            }
            let callables = &inner.callables;
            if callables.len() > 0 && self.init_once {
                panic!("Callable has already initialized but arguments do not match any of the previous calls");
            }
        }
        let (r_backup, device) = RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            let device = r.device.clone().unwrap();
            (
                std::mem::replace(&mut *r, Recorder::new()),
                device.upgrade().unwrap(),
            )
        });
        let mut builder = KernelBuilder::new(Some(device), false);
        let new_callable = (inner.builder)(args, &mut builder);
        RECORDER.with(|r| {
            *r.borrow_mut() = r_backup;
        });
        assert!(
            crate::lang::__check_callable(&new_callable.inner.module, nodes),
            "Callable builder returned a callable that does not match the arguments"
        );
        let callables = &mut inner.callables;
        callables.push(new_callable);
        CallableRet::_from_return(crate::lang::__invoke_callable(
            &callables.last().unwrap().inner.module,
            nodes,
        ))
    }
}
unsafe impl Send for RawCallable {}
unsafe impl Sync for RawCallable {}
pub struct RawCallable {
    #[allow(dead_code)]
    pub(crate) device: Option<Device>,
    pub(crate) module: ir::CallableModuleRef,
    #[allow(dead_code)]
    pub(crate) resource_tracker: ResourceTracker,
}
impl RawCallable {
    pub(crate) fn check_on_same_device(&self) {
        RECORDER.with(|r| {
            let mut r =r.borrow_mut();
            if let Some(device) = &self.device {
                if let Some((a,b)) = r.check_on_same_device(device) {
                    panic!("Callable created on a different device than the one it is called on: {:?} vs {:?}", a,b);
                }
            }
        });
    }
}
pub struct RawKernelDef {
    #[allow(dead_code)]
    pub(crate) device: Option<Device>,
    pub(crate) module: CArc<KernelModule>,
    #[allow(dead_code)]
    pub(crate) resource_tracker: ResourceTracker,
}

/// A kernel definition
/// See [`Kernel<T>`] for more information
pub struct KernelDef<T: KernelSignature> {
    pub(crate) inner: RawKernelDef,
    pub(crate) _marker: PhantomData<T>,
}

/// An executable kernel
/// Kernel creation can be done in multiple ways:
/// - Seperate recording and compilation:
///
/// ```no_run
/// // Recording:
/// use luisa_compute::prelude::*;
/// let ctx = Context::new(std::env::current_exe().unwrap());
/// let device = ctx.create_device("cpu");
/// type Signature = fn(Buffer<f32>, Buffer<f32>, Buffer<f32>);
/// let kernel = KernelDef::<Signature>::new(&device, &track!(|a,b,c|{  }));
/// // Compilation:
/// let kernel = device.compile_kernel_def(&kernel);
/// ```
///
/// - Recording and compilation in one step:
///
/// ```no_run
/// use luisa_compute::prelude::*;
/// let ctx = Context::new(std::env::current_exe().unwrap());
/// let device = ctx.create_device("cpu");
/// type Signature = fn(Buffer<f32>, Buffer<f32>, Buffer<f32>);
/// let kernel = device.
///     create_kernel::<Signature>(&track!(|a,b,c|{ }));
/// ```
/// - Asynchronous compilation use [`Kernel::<T>::new_async`] or [`Device::create_kernel_async`]
/// - Custom build options using [`Kernel::<T>::new_with_options`] or [`Device::create_kernel_async`]
pub struct Kernel<T: KernelSignature> {
    pub(crate) inner: Arc<RawKernel>,
    pub(crate) _marker: PhantomData<T>,
}
unsafe impl<T: KernelSignature> Send for Kernel<T> {}
unsafe impl<T: KernelSignature> Sync for Kernel<T> {}
impl<T: KernelSignature> Kernel<T> {
    pub fn cache_dir(&self) -> Option<PathBuf> {
        let handle = self.inner.unwrap();
        let device = &self.inner.device;
        device.inner.shader_cache_dir(handle)
    }
    pub fn dump(&self) -> String {
        ir::debug::dump_ir_human_readable(&self.inner.module.module)
    }
}

pub trait AsKernelArg<T: KernelArg>: KernelArg {}

impl<T: Value> AsKernelArg<T> for T {}

impl<T: Value> AsKernelArg<Buffer<T>> for Buffer<T> {}

impl<'a, T: Value> AsKernelArg<Buffer<T>> for BufferView<'a, T> {}

impl<'a, T: Value> AsKernelArg<BufferView<'a, T>> for BufferView<'a, T> {}

impl<'a, T: Value> AsKernelArg<BufferView<'a, T>> for Buffer<T> {}

// impl AsKernelArg<ByteBuffer> for ByteBuffer {}

// impl<'a> AsKernelArg<ByteBuffer> for ByteBufferView<'a> {}

// impl<'a> AsKernelArg<ByteBufferView<'a>> for ByteBufferView<'a> {}

// impl<'a> AsKernelArg<ByteBufferView<'a>> for ByteBuffer {}

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
    ( $($Ts:ident)*) => {
        impl <R:CallableRet+'static, $($Ts: CallableParameter),*> Callable<fn($($Ts,)*)->R> {
            #[allow(non_snake_case)]
            #[allow(unused_mut)]
            pub fn call(&self, $($Ts:$Ts),*) -> R {
                let mut encoder = CallableArgEncoder::new();
                self.inner.check_on_same_device();
                $($Ts.encode(&mut encoder);)*
                CallableRet::_from_return(
                    crate::lang::__invoke_callable(&self.inner.module, &encoder.args))
            }
        }
        impl <R:CallableRet+'static, $($Ts: CallableParameter),*> DynCallable<fn($($Ts,)*)->R> {
            #[allow(non_snake_case)]
            #[allow(unused_mut)]
            pub fn call(&self, $($Ts:$Ts),*) -> R {
                let mut encoder = CallableArgEncoder::new();
                $($Ts.encode(&mut encoder);)*
                self.call_impl(std::rc::Rc::new(($($Ts,)*)), &encoder.args)
            }
        }
   };
}
impl_call_for_callable!();
impl_call_for_callable!(T0);
impl_call_for_callable!(T0 T1 );
impl_call_for_callable!(T0 T1 T2 );
impl_call_for_callable!(T0 T1 T2 T3 );
impl_call_for_callable!(T0 T1 T2 T3 T4 );
impl_call_for_callable!(T0 T1 T2 T3 T4 T5 );
impl_call_for_callable!(T0 T1 T2 T3 T4 T5 T6 );
impl_call_for_callable!(T0 T1 T2 T3 T4 T5 T6 T7 );
impl_call_for_callable!(T0 T1 T2 T3 T4 T5 T6 T7 T8 );
impl_call_for_callable!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 );
impl_call_for_callable!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 );
impl_call_for_callable!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 );
impl_call_for_callable!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 );
impl_call_for_callable!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 );
impl_call_for_callable!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 );
impl_call_for_callable!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);

macro_rules! impl_dispatch_for_kernel {

   ($($Ts:ident)*) => {
        impl <$($Ts: KernelArg+'static),*> Kernel<fn($($Ts,)*)> {
            #[allow(non_snake_case)]
            #[allow(unused_mut)]
            pub fn dispatch(&self, dispatch_size: [u32; 3], $($Ts:&impl AsKernelArg<$Ts>),*)  {
                let mut encoder = KernelArgEncoder::new();
                $($Ts.encode(&mut encoder);)*
                self.inner.dispatch(encoder, dispatch_size)
            }
            #[allow(non_snake_case)]
            #[allow(unused_mut)]
            pub fn dispatch_async(
                &self,
                dispatch_size: [u32; 3], $($Ts:&impl AsKernelArg<$Ts>),*
            ) -> Command<'static, 'static> {
                let mut encoder = KernelArgEncoder::new();
                $($Ts.encode(&mut encoder);)*
                self.inner.dispatch_async(encoder, dispatch_size)
            }
            /// Blocks until the kernel is compiled
            pub fn ensure_ready(&self) {
                self.inner.unwrap();
            }
        }
   };
}

impl_dispatch_for_kernel!();
impl_dispatch_for_kernel!(T0);
impl_dispatch_for_kernel!(T0 T1 );
impl_dispatch_for_kernel!(T0 T1 T2 );
impl_dispatch_for_kernel!(T0 T1 T2 T3 );
impl_dispatch_for_kernel!(T0 T1 T2 T3 T4 );
impl_dispatch_for_kernel!(T0 T1 T2 T3 T4 T5 );
impl_dispatch_for_kernel!(T0 T1 T2 T3 T4 T5 T6 );
impl_dispatch_for_kernel!(T0 T1 T2 T3 T4 T5 T6 T7 );
impl_dispatch_for_kernel!(T0 T1 T2 T3 T4 T5 T6 T7 T8 );
impl_dispatch_for_kernel!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 );
impl_dispatch_for_kernel!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 );
impl_dispatch_for_kernel!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 );
impl_dispatch_for_kernel!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 );
impl_dispatch_for_kernel!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 );
impl_dispatch_for_kernel!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 );
impl_dispatch_for_kernel!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);

#[macro_export]
macro_rules! remove_last {
    ($x:ident) => {

    };
    ($first:ident $($xs:ident)*) => {
        $first remove_last!($($xs)*)
    };
}
