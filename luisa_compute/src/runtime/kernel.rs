use super::*;

impl<T: Value> CallableParameter for Expr<T> {
    fn def_param(_: Option<Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self {
        builder.value::<T>()
    }
    fn encode(&self, encoder: &mut CallableArgEncoder) {
        encoder.var(self.clone())
    }
}
impl<T: Value> CallableParameter for Var<T> {
    fn def_param(_: Option<Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self {
        builder.var::<T>()
    }
    fn encode(&self, encoder: &mut CallableArgEncoder) {
        encoder.var(self.clone())
    }
}

// Not recommended to use this directly
pub struct KernelBuilder {
    device: Option<crate::runtime::Device>,
    args: Vec<NodeRef>,
}

pub trait CallableParameter: Sized + Clone + 'static {
    fn def_param(arg: Option<Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self;
    fn encode(&self, encoder: &mut CallableArgEncoder);
}
macro_rules! impl_callable_parameter_for_tuple {
    ()=>{
        impl CallableParameter for () {
            fn def_param(_: Option<Rc<dyn Any>>, _: &mut KernelBuilder) {}
            fn encode(&self, _: &mut CallableArgEncoder) { }
        }
    };
    ($first:ident  $($rest:ident) *) => {
        impl<$first:CallableParameter, $($rest: CallableParameter),*> CallableParameter for ($first, $($rest,)*) {
            #[allow(non_snake_case)]
            fn def_param(arg: Option<Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self {
                if let Some(arg) = arg {
                    let ($first, $($rest,)*) = arg.downcast_ref::<($first, $($rest,)*)>().cloned().unwrap();
                    let $first = $first::def_param(Some(std::rc::Rc::new($first)), builder);
                    let ($($rest,)*) = ($($rest::def_param(Some(std::rc::Rc::new($rest)), builder),)*);
                    ($first, $($rest,)*)
                }else {
                    let $first = $first::def_param(None, builder);
                    let ($($rest,)*) = ($($rest::def_param(None, builder),)*);
                    ($first, $($rest,)*)
                }
            }
            #[allow(non_snake_case)]
            fn encode(&self, encoder: &mut CallableArgEncoder) {
                let ($first, $($rest,)*) = self;
                $first.encode(encoder);
                $($rest.encode(encoder);)*
            }
        }
        impl_callable_parameter_for_tuple!($($rest)*);
    };

}
impl_callable_parameter_for_tuple!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);

impl<T: Value + 'static> CallableParameter for BufferVar<T> {
    fn def_param(_: Option<Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self {
        builder.buffer()
    }
    fn encode(&self, encoder: &mut CallableArgEncoder) {
        encoder.buffer(self)
    }
}
impl<T: IoTexel + 'static> CallableParameter for Tex2dVar<T> {
    fn def_param(_: Option<Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self {
        builder.tex2d()
    }
    fn encode(&self, encoder: &mut CallableArgEncoder) {
        encoder.tex2d(self)
    }
}

impl<T: IoTexel + 'static> CallableParameter for Tex3dVar<T> {
    fn def_param(_: Option<Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self {
        builder.tex3d()
    }
    fn encode(&self, encoder: &mut CallableArgEncoder) {
        encoder.tex3d(self)
    }
}

impl CallableParameter for BindlessArrayVar {
    fn def_param(_: Option<Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self {
        builder.bindless_array()
    }
    fn encode(&self, encoder: &mut CallableArgEncoder) {
        encoder.bindless_array(self)
    }
}
impl KernelParameter for rtx::AccelVar {
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.accel()
    }
}

pub trait KernelParameter {
    fn def_param(builder: &mut KernelBuilder) -> Self;
}

impl<T: Value> KernelParameter for Expr<T> {
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.uniform::<T>()
    }
}

impl<T: Value> KernelParameter for BufferVar<T> {
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.buffer()
    }
}

impl<T: IoTexel> KernelParameter for Tex2dVar<T> {
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.tex2d()
    }
}

impl<T: IoTexel> KernelParameter for Tex3dVar<T> {
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.tex3d()
    }
}

impl KernelParameter for BindlessArrayVar {
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.bindless_array()
    }
}

macro_rules! impl_kernel_param_for_tuple {
    ($first:ident  $($rest:ident)*) => {
        impl<$first:KernelParameter, $($rest: KernelParameter),*> KernelParameter for ($first, $($rest,)*) {
            #[allow(non_snake_case)]
            fn def_param(builder: &mut KernelBuilder) -> Self {
                ($first::def_param(builder), $($rest::def_param(builder)),*)
            }
        }
        impl_kernel_param_for_tuple!($($rest)*);
    };
    ()=>{
        impl KernelParameter for () {
            fn def_param(_: &mut KernelBuilder) -> Self {
            }
        }
    }
}
impl_kernel_param_for_tuple!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);
impl KernelBuilder {
    pub fn new(device: Option<crate::runtime::Device>, is_kernel: bool) -> Self {
        RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            assert!(!r.lock, "Cannot record multiple kernels at the same time");
            assert!(
                r.scopes.is_empty(),
                "Cannot record multiple kernels at the same time"
            );
            r.lock = true;
            r.device = device.as_ref().map(|d| WeakDevice::new(d));
            r.pools = Some(CArc::new(ModulePools::new()));
            r.scopes.clear();
            r.building_kernel = is_kernel;
            let pools = r.pools.clone().unwrap();
            r.scopes.push(IrBuilder::new(pools));
        });
        Self {
            device,
            args: vec![],
        }
    }
    pub(crate) fn arg(&mut self, ty: CArc<Type>, by_value: bool) -> NodeRef {
        let node = new_node(
            __module_pools(),
            Node::new(CArc::new(Instruction::Argument { by_value }), ty),
        );
        self.args.push(node);
        node
    }
    pub fn value<T: Value>(&mut self) -> Expr<T> {
        let node = self.arg(T::type_(), true);
        FromNode::from_node(node)
    }
    pub fn var<T: Value>(&mut self) -> Var<T> {
        let node = self.arg(T::type_(), false);
        FromNode::from_node(node)
    }
    pub fn uniform<T: Value>(&mut self) -> Expr<T> {
        let node = new_node(
            __module_pools(),
            Node::new(CArc::new(Instruction::Uniform), T::type_()),
        );
        self.args.push(node);
        FromNode::from_node(node)
    }
    pub fn buffer<T: Value>(&mut self) -> BufferVar<T> {
        let node = new_node(
            __module_pools(),
            Node::new(CArc::new(Instruction::Buffer), T::type_()),
        );
        self.args.push(node);
        BufferVar {
            node,
            marker: PhantomData,
            handle: None,
        }
    }
    pub fn tex2d<T: IoTexel>(&mut self) -> Tex2dVar<T> {
        let node = new_node(
            __module_pools(),
            Node::new(CArc::new(Instruction::Texture2D), T::type_()),
        );
        self.args.push(node);
        Tex2dVar {
            node,
            marker: PhantomData,
            handle: None,
            level: None,
        }
    }
    pub fn tex3d<T: IoTexel>(&mut self) -> Tex3dVar<T> {
        let node = new_node(
            __module_pools(),
            Node::new(CArc::new(Instruction::Texture3D), T::type_()),
        );
        self.args.push(node);
        Tex3dVar {
            node,
            marker: PhantomData,
            handle: None,
            level: None,
        }
    }
    pub fn bindless_array(&mut self) -> BindlessArrayVar {
        let node = new_node(
            __module_pools(),
            Node::new(CArc::new(Instruction::Bindless), Type::void()),
        );
        self.args.push(node);
        BindlessArrayVar { node, handle: None }
    }
    pub fn accel(&mut self) -> rtx::AccelVar {
        let node = new_node(
            __module_pools(),
            Node::new(CArc::new(Instruction::Accel), Type::void()),
        );
        self.args.push(node);
        rtx::AccelVar { node, handle: None }
    }
    fn collect_module_info(&self) -> (ResourceTracker, Vec<CArc<CpuCustomOp>>, Vec<Capture>) {
        RECORDER.with(|r| {
            let mut resource_tracker = ResourceTracker::new();
            let r = r.borrow_mut();
            let mut captured: Vec<Capture> = Vec::new();
            let mut captured_buffers: Vec<_> = r.captured_buffer.values().cloned().collect();
            captured_buffers.sort_by_key(|(i, _, _, _)| *i);
            for (j, (i, node, binding, handle)) in captured_buffers.into_iter().enumerate() {
                assert_eq!(j, i);
                captured.push(Capture { node, binding });
                resource_tracker.add_any(handle);
            }
            let mut cpu_custom_ops: Vec<_> = r.cpu_custom_ops.values().cloned().collect();
            cpu_custom_ops.sort_by_key(|(i, _)| *i);
            let mut cpu_custom_ops: Vec<CArc<CpuCustomOp>> = cpu_custom_ops
                .iter()
                .enumerate()
                .map(|(j, (i, op))| {
                    assert_eq!(j, *i);
                    (*op).clone()
                })
                .collect::<Vec<_>>();
            let callables: Vec<CallableModuleRef> = r.callables.values().cloned().collect();
            let mut captured_set = HashSet::<Capture>::new();
            let mut cpu_custom_ops_set = HashSet::<u64>::new();
            let mut callable_set = HashSet::<u64>::new();
            for capture in captured.iter() {
                captured_set.insert(*capture);
            }
            for op in &cpu_custom_ops {
                cpu_custom_ops_set.insert(CArc::as_ptr(op) as u64);
            }
            for c in &callables {
                callable_set.insert(CArc::as_ptr(&c.0) as u64);
                for capture in c.0.captures.as_ref() {
                    if !captured_set.contains(capture) {
                        captured_set.insert(*capture);
                        captured.push(*capture);
                    }
                }
                for op in c.0.cpu_custom_ops.as_ref() {
                    let id = CArc::as_ptr(op) as u64;
                    if !cpu_custom_ops_set.contains(&id) {
                        cpu_custom_ops_set.insert(id);
                        cpu_custom_ops.push(op.clone());
                    }
                }
            }
            (resource_tracker, cpu_custom_ops, captured)
        })
    }
    fn build_callable<R: CallableRet>(&mut self, body: impl FnOnce(&mut Self) -> R) -> RawCallable {
        let ret = body(self);
        let ret_type = ret._return();
        let (rt, cpu_custom_ops, captures) = self.collect_module_info();
        RECORDER.with(|r| {
            let mut r = r.borrow_mut();
            assert!(r.lock);
            r.lock = false;
            if let Some(t) = &r.callable_ret_type {
                assert!(
                    luisa_compute_ir::context::is_type_equal(t, &ret_type),
                    "Return type mismatch"
                );
            } else {
                r.callable_ret_type = Some(ret_type.clone());
            }
            assert_eq!(r.scopes.len(), 1);
            let scope = r.scopes.pop().unwrap();
            let entry = scope.finish();
            let ir_module = Module {
                entry,
                kind: ModuleKind::Kernel,
                pools: r.pools.clone().unwrap(),
                flags: ModuleFlags::REQUIRES_REV_AD_TRANSFORM
                    | ModuleFlags::REQUIRES_FWD_AD_TRANSFORM,
            };
            let ir_module = luisa_compute_ir::transform::luisa_compute_ir_transform_auto(ir_module);
            let module = CallableModule {
                module: ir_module,
                ret_type,
                cpu_custom_ops: CBoxedSlice::new(cpu_custom_ops),
                captures: CBoxedSlice::new(captures),
                args: CBoxedSlice::new(self.args.clone()),
                pools: r.pools.clone().unwrap(),
            };
            let module = CallableModuleRef(CArc::new(module));
            r.reset();
            RawCallable {
                module,
                resource_tracker: rt,
            }
        })
    }
    fn build_kernel(
        &mut self,
        options: KernelBuildOptions,
        body: impl FnOnce(&mut Self),
    ) -> crate::runtime::RawKernel {
        body(self);
        let (rt, cpu_custom_ops, captures) = self.collect_module_info();
        RECORDER.with(|r| -> crate::runtime::RawKernel {
            let mut r = r.borrow_mut();
            assert!(r.lock);
            r.lock = false;
            assert_eq!(r.scopes.len(), 1);
            let scope = r.scopes.pop().unwrap();
            let entry = scope.finish();

            let ir_module = Module {
                entry,
                kind: ModuleKind::Kernel,
                pools: r.pools.clone().unwrap(),
                flags: ModuleFlags::REQUIRES_REV_AD_TRANSFORM
                    | ModuleFlags::REQUIRES_FWD_AD_TRANSFORM,
            };
            let ir_module = luisa_compute_ir::transform::luisa_compute_ir_transform_auto(ir_module);
            let module = KernelModule {
                module: ir_module,
                cpu_custom_ops: CBoxedSlice::new(cpu_custom_ops),
                captures: CBoxedSlice::new(captures),
                shared: CBoxedSlice::new(r.shared.clone()),
                args: CBoxedSlice::new(self.args.clone()),
                block_size: r.block_size.unwrap_or([64, 1, 1]),
                pools: r.pools.clone().unwrap(),
            };

            let module = CArc::new(module);
            let name = options.name.unwrap_or("".to_string());
            let name = Arc::new(CString::new(name).unwrap());
            let shader_options = api::ShaderOption {
                enable_cache: options.enable_cache,
                enable_fast_math: options.enable_fast_math,
                enable_debug_info: options.enable_debug_info,
                compile_only: false,
                name: name.as_ptr(),
            };
            let artifact = if options.async_compile {
                ShaderArtifact::Async(AsyncShaderArtifact::new(
                    self.device.clone().unwrap(),
                    module.clone(),
                    shader_options,
                    name,
                ))
            } else {
                ShaderArtifact::Sync(
                    self.device
                        .as_ref()
                        .unwrap()
                        .inner
                        .create_shader(&module, &shader_options),
                )
            };
            //
            r.reset();
            RawKernel {
                artifact,
                device: self.device.clone().unwrap(),
                resource_tracker: rt,
                module,
            }
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelBuildOptions {
    pub enable_debug_info: bool,
    pub enable_optimization: bool,
    pub async_compile: bool,
    pub enable_cache: bool,
    pub enable_fast_math: bool,
    pub name: Option<String>,
}

impl Default for KernelBuildOptions {
    fn default() -> Self {
        let enable_debug_info = match env::var("LUISA_DEBUG") {
            Ok(s) => s == "1",
            Err(_) => false,
        };
        Self {
            enable_debug_info,
            enable_optimization: true,
            async_compile: false,
            enable_cache: true,
            enable_fast_math: true,
            name: None,
        }
    }
}

pub trait KernelBuildFn {
    fn build_kernel(
        &self,
        builder: &mut KernelBuilder,
        options: KernelBuildOptions,
    ) -> crate::runtime::RawKernel;
}

pub trait CallableBuildFn {
    fn build_callable(&self, args: Option<Rc<dyn Any>>, builder: &mut KernelBuilder)
        -> RawCallable;
}

pub trait StaticCallableBuildFn: CallableBuildFn {}

// @FIXME: this looks redundant
pub unsafe trait CallableRet {
    fn _return(&self) -> CArc<Type>;
    fn _from_return(node: NodeRef) -> Self;
}

unsafe impl CallableRet for () {
    fn _return(&self) -> CArc<Type> {
        Type::void()
    }
    fn _from_return(_: NodeRef) -> Self {}
}

unsafe impl<V: Value> CallableRet for Expr<V> {
    fn _return(&self) -> CArc<Type> {
        __current_scope(|b| {
            b.return_(self.node());
        });
        V::type_()
    }
    fn _from_return(node: NodeRef) -> Self {
        Self::from_node(node)
    }
}

pub trait CallableSignature<'a> {
    type Callable;
    type DynCallable;
    type Fn: CallableBuildFn;
    type StaticFn: StaticCallableBuildFn;
    type DynFn: CallableBuildFn + 'static;
    type Ret: CallableRet;
    fn wrap_raw_callable(callable: RawCallable) -> Self::Callable;
    fn create_dyn_callable(device: Device, init_once: bool, f: Self::DynFn) -> Self::DynCallable;
}

pub trait KernelSignature<'a> {
    type Fn: KernelBuildFn;
    type Kernel;

    fn wrap_raw_kernel(kernel: crate::runtime::RawKernel) -> Self::Kernel;
}
macro_rules! impl_callable_signature {
    ()=>{
        impl<'a, R: CallableRet +'static> CallableSignature<'a> for fn()->R {
            type Fn = &'a dyn Fn() ->R;
            type DynFn = Box<dyn Fn() ->R>;
            type StaticFn = fn() -> R;
            type Callable = Callable<fn()->R>;
            type DynCallable = DynCallable<fn()->R>;
            type Ret = R;
            fn wrap_raw_callable(callable: RawCallable) -> Self::Callable{
                Callable {
                    inner: callable,
                    _marker:PhantomData,
                }
            }
            fn create_dyn_callable(device:Device, init_once:bool, f: Self::DynFn) -> Self::DynCallable {
                DynCallable::new(device, init_once, Box::new(move |arg, builder| {
                    let raw_callable = CallableBuildFn::build_callable(&f, Some(arg), builder);
                    Self::wrap_raw_callable(raw_callable)
                }))
            }
        }
    };
    ($first:ident  $($rest:ident)*) => {
        impl<'a, R:CallableRet +'static, $first:CallableParameter +'static, $($rest: CallableParameter +'static),*> CallableSignature<'a> for fn($first, $($rest,)*)->R {
            type Fn = &'a dyn Fn($first, $($rest),*)->R;
            type DynFn = Box<dyn Fn($first, $($rest),*)->R>;
            type Callable = Callable<fn($first, $($rest,)*)->R>;
            type StaticFn = fn($first, $($rest,)*)->R;
            type DynCallable = DynCallable<fn($first, $($rest,)*)->R>;
            type Ret = R;
            fn wrap_raw_callable(callable: RawCallable) -> Self::Callable{
                Callable {
                    inner: callable,
                    _marker:PhantomData,
                }
            }
            fn create_dyn_callable(device:Device, init_once:bool, f: Self::DynFn) -> Self::DynCallable {
                DynCallable::new(device, init_once, Box::new(move |arg, builder| {
                    let raw_callable = CallableBuildFn::build_callable(&f, Some(arg), builder);
                    Self::wrap_raw_callable(raw_callable)
                }))
            }
        }
        impl_callable_signature!($($rest)*);
    };
}
impl_callable_signature!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);
macro_rules! impl_kernel_signature {
    ()=>{
        impl<'a> KernelSignature<'a> for fn() {
            type Fn = &'a dyn Fn();
            type Kernel = Kernel<fn()>;
            fn wrap_raw_kernel(kernel: crate::runtime::RawKernel) -> Self::Kernel {
                Self::Kernel{
                    inner:kernel,
                    _marker:PhantomData,
                }
            }
        }
    };
    ($first:ident  $($rest:ident)*) => {
        impl<'a, $first:KernelArg +'static, $($rest: KernelArg +'static),*> KernelSignature<'a> for fn($first, $($rest,)*) {
            type Fn = &'a dyn Fn($first::Parameter, $($rest::Parameter),*);
            type Kernel = Kernel<fn($first, $($rest,)*)>;
            fn wrap_raw_kernel(kernel: crate::runtime::RawKernel) -> Self::Kernel {
                Self::Kernel{
                    inner:kernel,
                    _marker:PhantomData,
                }
            }
        }
        impl_kernel_signature!($($rest)*);
    };
}
impl_kernel_signature!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);

macro_rules! impl_callable_build_for_fn {
    ()=>{
        impl<R:CallableRet +'static> CallableBuildFn for &dyn Fn()->R {
            fn build_callable(&self, _args: Option<Rc<dyn Any>>, builder: &mut KernelBuilder)->RawCallable {
                builder.build_callable( |_| {
                    self()
                })
            }
        }
        impl<R:CallableRet +'static> CallableBuildFn for fn()->R {
            fn build_callable(&self, _args: Option<Rc<dyn Any>>, builder: &mut KernelBuilder)->RawCallable {
                builder.build_callable( |_| {
                    self()
                })
            }
        }
        impl<R:CallableRet +'static> CallableBuildFn for Box<dyn Fn()->R> {
            fn build_callable(&self, _args: Option<Rc<dyn Any>>, builder: &mut KernelBuilder)->RawCallable {
                builder.build_callable( |_| {
                    self()
                })
            }
        }
        impl <R:CallableRet +'static> StaticCallableBuildFn  for fn()->R {}
    };
    ($first:ident  $($rest:ident)*) => {
        impl<R:CallableRet +'static, $first:CallableParameter, $($rest: CallableParameter),*> CallableBuildFn for &dyn Fn($first, $($rest,)*)->R {
            #[allow(non_snake_case)]
            fn build_callable(&self, args: Option<Rc<dyn Any>>, builder: &mut KernelBuilder)->RawCallable {
                builder.build_callable( |builder| {
                    if let Some(args) = args {
                        let ($first, $($rest,)*) = args.downcast_ref::<($first, $($rest,)*)>().cloned().unwrap();
                        let $first = $first::def_param(Some(Rc::new($first)), builder);
                        $(let $rest = $rest::def_param(Some(Rc::new($rest)), builder);)*
                        self($first, $($rest,)*)
                    } else {
                        let $first = $first::def_param(None, builder);
                        $(let $rest = $rest::def_param(None, builder);)*
                        self($first, $($rest,)*)
                    }
                })
            }
        }
        impl<R:CallableRet +'static, $first:CallableParameter, $($rest: CallableParameter),*> CallableBuildFn for Box<dyn Fn($first, $($rest,)*)->R> {
            #[allow(non_snake_case)]
            fn build_callable(&self, args: Option<Rc<dyn Any>>, builder: &mut KernelBuilder)->RawCallable {
                builder.build_callable( |builder| {
                    if let Some(args) = args {
                        let ($first, $($rest,)*) = args.downcast_ref::<($first, $($rest,)*)>().cloned().unwrap();
                        let $first = $first::def_param(Some(Rc::new($first)), builder);
                        $(let $rest = $rest::def_param(Some(Rc::new($rest)), builder);)*
                        self($first, $($rest,)*)
                    } else {
                        let $first = $first::def_param(None, builder);
                        $(let $rest = $rest::def_param(None, builder);)*
                        self($first, $($rest,)*)
                    }
                })
            }
        }
        impl<R:CallableRet +'static, $first:CallableParameter, $($rest: CallableParameter),*> CallableBuildFn for fn($first, $($rest,)*)->R {
            #[allow(non_snake_case)]
            fn build_callable(&self, args: Option<Rc<dyn Any>>, builder: &mut KernelBuilder)->RawCallable {
                builder.build_callable( |builder| {
                    if let Some(args) = args {
                        let ($first, $($rest,)*) = args.downcast_ref::<($first, $($rest,)*)>().cloned().unwrap();
                        let $first = $first::def_param(Some(Rc::new($first)), builder);
                        $(let $rest = $rest::def_param(Some(Rc::new($rest)), builder);)*
                        self($first, $($rest,)*)
                    } else {
                        let $first = $first::def_param(None, builder);
                        $(let $rest = $rest::def_param(None, builder);)*
                        self($first, $($rest,)*)
                    }
                })
            }
        }
        impl<R:CallableRet +'static, $first:CallableParameter, $($rest: CallableParameter),*> StaticCallableBuildFn for fn($first, $($rest,)*)->R {}
        impl_callable_build_for_fn!($($rest)*);
    };
}
impl_callable_build_for_fn!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);
macro_rules! impl_kernel_build_for_fn {
    ()=>{
        impl KernelBuildFn for &dyn Fn() {
            fn build_kernel(&self, builder: &mut KernelBuilder, options:KernelBuildOptions) -> crate::runtime::RawKernel {
                builder.build_kernel(options, |_| {
                    self()
                })
            }
        }
    };
    ($first:ident  $($rest:ident)*) => {
        impl<$first:KernelParameter, $($rest: KernelParameter),*> KernelBuildFn for &dyn Fn($first, $($rest,)*) {
            #[allow(non_snake_case)]
            fn build_kernel(&self, builder: &mut KernelBuilder, options:KernelBuildOptions) -> crate::runtime::RawKernel {
                builder.build_kernel(options, |builder| {
                    let $first = $first::def_param(builder);
                    $(let $rest = $rest::def_param(builder);)*
                    self($first, $($rest,)*)
                })
            }
        }
        impl_kernel_build_for_fn!($($rest)*);
    };
}
impl_kernel_build_for_fn!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);
