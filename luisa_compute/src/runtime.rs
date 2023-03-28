use crate::backend::Backend;
use crate::lang::ShaderBuildOptions;
use crate::*;
use crate::{lang::Value, resource::*};

use api::AccelOption;
use lang::{KernelBuildFn, KernelBuilder, KernelParameter, KernelSignature};
pub use luisa_compute_api_types as api;
use luisa_compute_ir::ir::{self, KernelModule};
use luisa_compute_ir::CArc;
use parking_lot::{Condvar, Mutex};
use raw_window_handle::HasRawWindowHandle;
use rtx::{Accel, Mesh, MeshHandle};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
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
    pub(crate) backend: Arc<dyn Backend>,
    pub(crate) default_stream: api::Stream,
}
impl Deref for DeviceHandle {
    type Target = dyn Backend;
    fn deref(&self) -> &Self::Target {
        self.backend.deref()
    }
}
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Drop for DeviceHandle {
    fn drop(&mut self) {
        self.backend.destroy_stream(self.default_stream);
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
            resource_tracker: RefCell::new(ResourceTracker::new()),
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
            handle,
            marker: std::marker::PhantomData {},
        };
        Ok(tex)
    }

    pub fn default_stream(&self) -> Stream {
        Stream {
            device: self.clone(),
            handle: Arc::new(StreamHandle::Default(
                self.inner.clone(),
                self.inner.default_stream,
            )),
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
            }),
        })
    }
    pub fn create_mesh<V: Value, T: Value>(
        &self,
        vbuffer: BufferView<'_, V>,
        tbuffer: BufferView<'_, T>,
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
            index_buffer_offset: tbuffer.offset * std::mem::size_of::<T>() as usize,
            index_buffer_size: tbuffer.len * std::mem::size_of::<T>() as usize,
            index_stride: std::mem::size_of::<T>() as usize,
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
            mesh_handles: RefCell::new(Vec::new()),
            modifications: RefCell::new(HashMap::new()),
        })
    }
    // pub fn create_callable(&self, ) {

    // }
    pub fn create_kernel<'a, S: KernelSignature<'a>>(
        &self,
        f: S::Fn,
    ) -> Result<S::Kernel, crate::backend::BackendError> {
        let mut builder = KernelBuilder::new(self.clone());
        let raw_kernel = KernelBuildFn::build(&f, &mut builder, ShaderBuildOptions::default());
        S::wrap_raw_shader(raw_kernel)
    }
    pub fn create_kernel_async<'a, S: KernelSignature<'a>>(
        &self,
        f: S::Fn,
    ) -> Result<S::Kernel, crate::backend::BackendError> {
        let mut builder = KernelBuilder::new(self.clone());
        let raw_kernel = KernelBuildFn::build(
            &f,
            &mut builder,
            ShaderBuildOptions {
                async_compile: true,
                ..Default::default()
            },
        );
        S::wrap_raw_shader(raw_kernel)
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
    Default(Arc<DeviceHandle>, api::Stream),
    NonDefault {
        device: Arc<DeviceHandle>,
        handle: api::Stream,
        native_handle: *mut std::ffi::c_void,
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
}
pub struct Stream {
    #[allow(dead_code)]
    pub(crate) device: Device,
    pub(crate) handle: Arc<StreamHandle>,
}
impl StreamHandle {
    pub(crate) fn device(&self) -> Arc<DeviceHandle> {
        match self {
            StreamHandle::Default(device, _) => device.clone(),
            StreamHandle::NonDefault { device, .. } => device.clone(),
        }
    }
    pub(crate) fn handle(&self) -> api::Stream {
        match self {
            StreamHandle::Default(_, stream) => *stream,
            StreamHandle::NonDefault { handle, .. } => *handle,
        }
    }
    pub(crate) fn native_handle(&self) -> *mut std::ffi::c_void {
        match self {
            StreamHandle::Default(_, _) => todo!(),
            StreamHandle::NonDefault { native_handle, .. } => *native_handle,
        }
    }
}
impl Drop for StreamHandle {
    fn drop(&mut self) {
        match self {
            StreamHandle::Default(_, _) => {}
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
            StreamHandle::Default(_, _) => todo!(),
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
    #[inline]
    pub fn submit_with_callback<F: FnOnce() + Send + 'static>(
        &self,
        commands: impl IntoIterator<Item = Command<'a>>,
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
    #[allow(dead_code)]
    pub(crate) resource_tracker: ResourceTracker,
}
pub(crate) struct AsyncShaderArtifact {
    shader: Option<backend::Result<api::CreatedShaderInfo>>, // strange naming, huh?
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
    ) -> Arc<(Mutex<AsyncShaderArtifact>, Condvar)> {
        let artifact = Arc::new((
            Mutex::new(AsyncShaderArtifact { shader: None }),
            Condvar::new(),
        ));
        {
            let artifact = artifact.clone();
            rayon::spawn(move || {
                let shader = device.inner.create_shader(kernel, &options);
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
pub struct RawShader {
    pub(crate) device: Device,
    pub(crate) artifact: ShaderArtifact,
    #[allow(dead_code)]
    pub(crate) resource_tracker: ResourceTracker,
}
pub struct ArgEncoder {
    pub(crate) args: Vec<api::Argument>,
}
impl ArgEncoder {
    pub fn new() -> ArgEncoder {
        ArgEncoder { args: Vec::new() }
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
pub trait KernelArg {
    type Parameter: KernelParameter;
    fn encode(&self, encoder: &mut ArgEncoder);
}
impl<T: Value> KernelArg for Buffer<T> {
    type Parameter = lang::BufferVar<T>;
    fn encode(&self, encoder: &mut ArgEncoder) {
        encoder.buffer(self);
    }
}
impl<'a, T: Value> KernelArg for BufferView<'a, T> {
    type Parameter = lang::BufferVar<T>;
    fn encode(&self, encoder: &mut ArgEncoder) {
        encoder.buffer_view(self);
    }
}
impl<T: IoTexel> KernelArg for Tex2d<T> {
    type Parameter = lang::Tex2dVar<T>;
    fn encode(&self, encoder: &mut ArgEncoder) {
        encoder.tex2d(&self.view(0));
    }
}
impl<T: IoTexel> KernelArg for Tex3d<T> {
    type Parameter = lang::Tex3dVar<T>;
    fn encode(&self, encoder: &mut ArgEncoder) {
        encoder.tex3d(&self.view(0));
    }
}
impl<'a, T: IoTexel> KernelArg for Tex2dView<'a, T> {
    type Parameter = lang::Tex2dVar<T>;
    fn encode(&self, encoder: &mut ArgEncoder) {
        encoder.tex2d(self);
    }
}
impl<'a, T: IoTexel> KernelArg for Tex3dView<'a, T> {
    type Parameter = lang::Tex3dVar<T>;
    fn encode(&self, encoder: &mut ArgEncoder) {
        encoder.tex3d(self);
    }
}
impl KernelArg for BindlessArray {
    type Parameter = lang::BindlessArrayVar;
    fn encode(&self, encoder: &mut ArgEncoder) {
        encoder.bindless_array(self);
    }
}
impl KernelArg for Accel {
    type Parameter = lang::AccelVar;
    fn encode(&self, encoder: &mut ArgEncoder) {
        encoder.accel(self)
    }
}
macro_rules! impl_kernel_arg_for_tuple {
    ()=>{
        impl KernelArg for () {
            type Parameter = ();
            fn encode(&self, _: &mut ArgEncoder) { }
        }
    };
    ($first:ident  $($rest:ident) *) => {
        impl<$first:KernelArg, $($rest: KernelArg),*> KernelArg for ($first, $($rest,)*) {
            type Parameter = ($first::Parameter, $($rest::Parameter),*);
            #[allow(non_snake_case)]
            fn encode(&self, encoder: &mut ArgEncoder) {
                let ($first, $($rest,)*) = self;
                $first.encode(encoder);
                $($rest.encode(encoder);)*
            }
        }
        impl_kernel_arg_for_tuple!($($rest)*);
    };

}
impl_kernel_arg_for_tuple!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);

impl RawShader {
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
    pub fn dispatch_async<'a>(&'a self, args: &ArgEncoder, dispatch_size: [u32; 3]) -> Command<'a> {
        Command {
            inner: api::Command::ShaderDispatch(api::ShaderDispatchCommand {
                shader: self.unwrap(),
                args: args.args.as_ptr(),
                args_count: args.args.len(),
                dispatch_size,
            }),
            marker: std::marker::PhantomData,
            resource_tracker: ResourceTracker::new(),
        }
    }
    pub fn dispatch(&self, args: &ArgEncoder, dispatch_size: [u32; 3]) -> backend::Result<()> {
        submit_default_stream_and_sync(&self.device, vec![self.dispatch_async(args, dispatch_size)])
    }
}
pub trait CallableArg {
    fn arg_node(&self)->NodeRef;
}
// impl <T> CallableArg for T where T: KernelArg {}
pub struct Callable<T: CallableArg> {
    pub(crate) inner: ir::CallableModuleRef,
    marker: std::marker::PhantomData<T>,
}
pub struct Kernel<T: KernelArg> {
    pub(crate) inner: RawShader,
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
macro_rules! impl_dispatch_for_kernel {

   ($first:ident  $($rest:ident)*) => {
        impl <$first:KernelArg, $($rest: KernelArg),*> Kernel<($first, $($rest,)*)> {
            #[allow(non_snake_case)]
            pub fn dispatch(&self, dispatch_size: [u32; 3], $first:&impl AsKernelArg<$first>, $($rest:&impl AsKernelArg<$rest>),*) -> backend::Result<()> {
                let mut encoder = ArgEncoder::new();
                $first.encode(&mut encoder);
                $($rest.encode(&mut encoder);)*
                self.inner.dispatch(&encoder, dispatch_size)
            }
            #[allow(non_snake_case)]
            pub fn dispatch_async<'a>(
                &'a self,
                dispatch_size: [u32; 3], $first: &impl AsKernelArg<$first>, $($rest:impl AsKernelArg<$rest>),*
            ) -> Command<'a> {
                let mut encoder = ArgEncoder::new();
                $first.encode(&mut encoder);
                $($rest.encode(&mut encoder);)*
                self.inner.dispatch_async(&encoder, dispatch_size)
            }
        }
        impl_dispatch_for_kernel!($($rest)*);
   };
   ()=>{
    impl Kernel<()> {
        pub fn dispatch(&self, dispatch_size: [u32; 3]) -> backend::Result<()> {
            self.inner.dispatch(&ArgEncoder::new(), dispatch_size)
        }
        pub fn dispatch_async<'a>(
            &'a self,
            dispatch_size: [u32; 3],
        ) -> Command<'a> {
            self.inner.dispatch_async(&ArgEncoder::new(), dispatch_size)
        }
    }
}
}
impl_dispatch_for_kernel!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);

#[cfg(all(test, feature = "_cpp"))]
mod test {
    use super::*;
    #[test]
    fn test_layout() {
        assert_eq!(
            std::mem::size_of::<api::Command>(),
            std::mem::size_of::<sys::LCCommand>()
        );
    }
}
