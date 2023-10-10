use crate::lang::{pop_recorder, push_recorder, soa::SoaMetadata, KERNEL_ID};

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
// impl CallableParameter for ByteBufferVar {
//     fn def_param(_: Option<Rc<dyn Any>>, builder: &mut KernelBuilder) -> Self {
//         builder.byte_buffer()
//     }
//     fn encode(&self, encoder: &mut CallableArgEncoder) {
//         encoder.byte_buffer(self)
//     }
// }
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
    type Arg = rtx::Accel;
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.accel()
    }
}

pub trait KernelParameter {
    type Arg: KernelArg<Parameter = Self> + 'static;
    fn def_param(builder: &mut KernelBuilder) -> Self;
}

impl<T: Value> KernelParameter for Expr<T> {
    type Arg = T;
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.uniform::<T>()
    }
}

impl<T: Value> KernelParameter for BufferVar<T> {
    type Arg = Buffer<T>;
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.buffer()
    }
}
impl<T: SoaValue> KernelParameter for SoaBufferVar<T> {
    type Arg = SoaBuffer<T>;
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.soa_buffer()
    }
}
// impl KernelParameter for ByteBufferVar {
//     type Arg = ByteBuffer;
//     fn def_param(builder: &mut KernelBuilder) -> Self {
//         builder.byte_buffer()
//     }
// }

impl<T: IoTexel> KernelParameter for Tex2dVar<T> {
    type Arg = Tex2d<T>;
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.tex2d()
    }
}

impl<T: IoTexel> KernelParameter for Tex3dVar<T> {
    type Arg = Tex3d<T>;
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.tex3d()
    }
}

impl KernelParameter for BindlessArrayVar {
    type Arg = BindlessArray;
    fn def_param(builder: &mut KernelBuilder) -> Self {
        builder.bindless_array()
    }
}

macro_rules! impl_kernel_param_for_tuple {
    ($first:ident  $($rest:ident)*) => {
        impl<$first:KernelParameter, $($rest: KernelParameter),*> KernelParameter for ($first, $($rest,)*) {
            type Arg = ($first::Arg, $($rest::Arg),*);
            #[allow(non_snake_case)]
            fn def_param(builder: &mut KernelBuilder) -> Self {
                ($first::def_param(builder), $($rest::def_param(builder)),*)
            }
        }
        impl_kernel_param_for_tuple!($($rest)*);
    };
    ()=>{
        impl KernelParameter for () {
            type Arg = ();
            fn def_param(_: &mut KernelBuilder) -> Self {
            }
        }
    }
}
impl_kernel_param_for_tuple!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);
impl KernelBuilder {
    pub fn new(device: Option<crate::runtime::Device>, is_kernel: bool) -> Self {
        let kernel_id = RECORDER.with(|r| {
            let r = r.borrow();
            if is_kernel {
                assert!(r.is_none(), "Cannot record a kernel inside another kernel");
                KERNEL_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            } else {
                let r = r.as_ref();
                if let Some(r) = r {
                    r.borrow().kernel_id
                } else {
                    KERNEL_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                }
            }
        });
        push_recorder(kernel_id);
        with_recorder(|r| {
            r.device = device.as_ref().map(|d| WeakDevice::new(d));
            r.pools = CArc::new(ModulePools::new());
            r.scopes.clear();
            r.building_kernel = is_kernel;
            let pools = r.pools.clone();
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
        FromNode::from_node(node.into())
    }
    pub fn var<T: Value>(&mut self) -> Var<T> {
        let node = self.arg(T::type_(), false);
        FromNode::from_node(node.into())
    }
    pub fn uniform<T: Value>(&mut self) -> Expr<T> {
        let node = new_node(
            __module_pools(),
            Node::new(CArc::new(Instruction::Uniform), T::type_()),
        );
        self.args.push(node);
        FromNode::from_node(node.into())
    }
    // pub fn byte_buffer(&mut self) -> ByteBufferVar {
    //     let node = new_node(
    //         __module_pools(),
    //         Node::new(CArc::new(Instruction::Buffer), Type::void()),
    //     );
    //     self.args.push(node);
    //     ByteBufferVar { node, handle: None }
    // }
    pub fn buffer<T: Value>(&mut self) -> BufferVar<T> {
        let node = new_node(
            __module_pools(),
            Node::new(CArc::new(Instruction::Buffer), T::type_()),
        );
        self.args.push(node);
        BufferVar {
            node: node.into(),
            marker: PhantomData,
            handle: None,
        }
    }
    pub fn soa_buffer<T: SoaValue>(&mut self) -> SoaBufferVar<T> {
        let storage = self.buffer::<u8>();
        let metadata = self.buffer::<SoaMetadata>();
        SoaBufferVar {
            proxy: T::SoaBuffer::from_soa_storage(storage, metadata.read(0), 0),
        }
    }
    pub fn tex2d<T: IoTexel>(&mut self) -> Tex2dVar<T> {
        let node = new_node(
            __module_pools(),
            Node::new(CArc::new(Instruction::Texture2D), T::type_()),
        );
        self.args.push(node);
        Tex2dVar {
            node: node.into(),
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
            node: node.into(),
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
        BindlessArrayVar {
            node: node.into(),
            handle: None,
        }
    }
    pub fn accel(&mut self) -> rtx::AccelVar {
        let node = new_node(
            __module_pools(),
            Node::new(CArc::new(Instruction::Accel), Type::void()),
        );
        self.args.push(node);
        rtx::AccelVar {
            node: node.into(),
            handle: None,
        }
    }
    fn collect_module_info(&self) -> (ResourceTracker, Vec<CArc<CpuCustomOp>>, Vec<Capture>) {
        with_recorder(|r| {
            let mut resource_tracker = ResourceTracker::new();
            let mut captured: Vec<Capture> = Vec::new();
            let mut captured_resources: Vec<_> = r.captured_resources.values().cloned().collect();
            captured_resources.sort_by_key(|(i, _, _, _)| *i);
            for (j, (i, node, binding, handle)) in captured_resources.into_iter().enumerate() {
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

    /// Don't use this directly
    /// See [`Callable`] for how to create a callable
    #[doc(hidden)]
    pub fn build_callable<R: CallableRet>(
        &mut self,
        body: impl FnOnce(&mut Self) -> R,
    ) -> RawCallable {
        let ret = body(self);
        let ret_type = ret._return();
        let (rt, cpu_custom_ops, captures) = self.collect_module_info();
        let ret = with_recorder(|r| {
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
                pools: r.pools.clone(),
                flags: ModuleFlags::REQUIRES_REV_AD_TRANSFORM
                    | ModuleFlags::REQUIRES_FWD_AD_TRANSFORM,
            };
            let ir_module = luisa_compute_ir::transform::luisa_compute_ir_transform_auto(ir_module);

            let mut args = self.args.clone();
            args.extend(r.captured_vars.values().map(|x| unsafe { x.get_raw() }));

            let module = CallableModule {
                module: ir_module,
                ret_type,
                cpu_custom_ops: CBoxedSlice::new(cpu_custom_ops),
                captures: CBoxedSlice::new(captures),
                args: CBoxedSlice::new(args.clone()),
                pools: r.pools.clone(),
            };
            let module = CallableModuleRef(CArc::new(module));

            RawCallable {
                device: self.device.clone(),
                module,
                resource_tracker: rt,
                captured_args: r
                    .captured_vars
                    .keys()
                    .map(|x| unsafe { x.get_raw() })
                    .collect(),
            }
        });
        pop_recorder();
        ret
    }

    /// Don't use this directly
    /// See [`Kernel`] for how to create a kernel
    #[doc(hidden)]
    pub fn build_kernel<S: KernelSignature>(
        &mut self,
        body: impl FnOnce(&mut Self),
    ) -> crate::runtime::KernelDef<S> {
        body(self);
        let (rt, cpu_custom_ops, captures) = self.collect_module_info();
        let ret = with_recorder(|r| {
            assert_eq!(r.scopes.len(), 1);
            let scope = r.scopes.pop().unwrap();
            let entry = scope.finish();
            assert!(r.captured_vars.is_empty());
            let ir_module = Module {
                entry,
                kind: ModuleKind::Kernel,
                pools: r.pools.clone(),
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
                pools: r.pools.clone(),
            };

            KernelDef {
                inner: RawKernelDef {
                    device: self.device.clone(),
                    resource_tracker: rt,
                    module: CArc::new(module),
                },
                _marker: PhantomData,
            }
        });
        pop_recorder();
        ret
    }
}

/// Build options for kernel compilation
/// * `enable_debug_info`: enable debug info, default true on debug build
/// * `enable_optimization`: enable optimization, default true
/// * `async_compile`: compile the kernel asynchronously
/// * `enable_cache`: enable cache for the compiled kernel
/// * `enable_fast_math`: enable fast math in the compiled kernel
/// * `name`: name of the compiled kernel. On CUDA backend, this is the name of
///   the generated PTX kernel
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
pub trait CallableBuildFn<S: CallableSignature> {
    fn build_callable(&self, args: Option<Rc<dyn Any>>, builder: &mut KernelBuilder)
        -> RawCallable;
}

pub trait StaticCallableBuildFn<S: CallableSignature>: CallableBuildFn<S> {}

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
        let ret = self.node().get();
        __current_scope(|b| {
            b.return_(ret);
        });
        V::type_()
    }
    fn _from_return(node: NodeRef) -> Self {
        Self::from_node(node.into())
    }
}

pub trait CallableSignature {
    type Ret: CallableRet;
}

macro_rules! impl_callable {
    ($($Ts:ident)*) => {
        impl<R:CallableRet +'static, $($Ts: CallableParameter +'static),*> CallableSignature for fn($($Ts,)*)->R {
            type Ret = R;
         }
        impl<R:CallableRet +'static, $($Ts: CallableParameter +'static),*> Callable<fn($($Ts,)*)->R> {
            pub fn new<F:Fn($($Ts,)*)->R>(device: &Device, f:F)->Self where F:CallableBuildFn<fn($($Ts,)*)->R> {
                Self::new_maybe_device(Some(device.clone()), f)
            }
            pub fn new_maybe_device<F:Fn($($Ts,)*)->R>(device: Option<Device>, f:F)->Self where F:CallableBuildFn<fn($($Ts,)*)->R> {
                let mut builder = KernelBuilder::new(device, false);
                let raw_callable = CallableBuildFn::build_callable(&f, None, &mut builder);
                Self{
                    inner: raw_callable,
                    _marker: PhantomData,
                }
            }
            pub fn new_static(f:fn($($Ts,)*)->R)->Self  where fn($($Ts,)*)->R :CallableBuildFn<fn($($Ts,)*)->R> {
                let r_backup = RECORDER.with(|r| {
                    let mut r = r.borrow_mut();
                    let kernel_id = r.as_ref().unwrap().borrow().kernel_id;
                    std::mem::replace(&mut *r, Some(Rc::new(RefCell::new(FnRecorder::new(kernel_id)))))
                });
                let mut builder = KernelBuilder::new(None, false);
                let raw_callable = CallableBuildFn::build_callable(&f, None, &mut builder);
                RECORDER.with(|r| {
                    let mut r = r.borrow_mut();
                    *r = r_backup;
                });
                Self{
                    inner: raw_callable,
                    _marker: PhantomData,
                }
            }
        }
        impl<R:CallableRet +'static, $($Ts: CallableParameter +'static),*> DynCallable<fn($($Ts,)*)->R> {
            pub fn new(device: &Device, f:Box<dyn Fn($($Ts,)*)->R>)->Self where Box<dyn Fn($($Ts,)*)->R> : CallableBuildFn<fn($($Ts,)*)->R> {
                DynCallable::_new(device.clone(), false, Box::new(move |arg, builder| {
                    let raw_callable = CallableBuildFn::build_callable(&f, Some(arg), builder);
                    Callable {
                        inner: raw_callable,
                        _marker: PhantomData,
                    }
                }))
            }
        }
    };
}

impl_callable!();
impl_callable!(T0);
impl_callable!(T0 T1 );
impl_callable!(T0 T1 T2 );
impl_callable!(T0 T1 T2 T3 );
impl_callable!(T0 T1 T2 T3 T4 );
impl_callable!(T0 T1 T2 T3 T4 T5 );
impl_callable!(T0 T1 T2 T3 T4 T5 T6 );
impl_callable!(T0 T1 T2 T3 T4 T5 T6 T7 );
impl_callable!(T0 T1 T2 T3 T4 T5 T6 T7 T8 );
impl_callable!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 );
impl_callable!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 );
impl_callable!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 );
impl_callable!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 );
impl_callable!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 );
impl_callable!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 );
impl_callable!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);

pub trait KernelSignature: Sized {}
pub trait KernelSignature2<'a>: KernelSignature {
    type Fn: KernelBuildFn<'a, Self>;
}
pub trait KernelBuildFn<'a, S: KernelSignature2<'a>> {
    fn build_kernel(&self, builder: &mut KernelBuilder) -> KernelDef<S>;
}

macro_rules! impl_kernel {
    ($($Ts:ident)*) => {
        impl<$($Ts: KernelArg +'static),*> KernelSignature for fn($($Ts,)*) {}
        impl<'a, $($Ts: KernelArg +'static),*> KernelSignature2<'a> for fn($($Ts,)*) {
            type Fn = &'a dyn Fn($($Ts::Parameter,)*);
        }
        impl<'a, $($Ts: KernelArg +'static),*> KernelBuildFn<'a, fn($($Ts,)*)> for &'a dyn Fn($($Ts::Parameter,)*) {
            #[allow(non_snake_case)]
            #[allow(unused_variables)]
            fn build_kernel(&self, builder: &mut KernelBuilder) -> KernelDef<fn($($Ts,)*)> {
                builder.build_kernel(|builder| {
                    $(let $Ts = <$Ts::Parameter as KernelParameter>::def_param(builder);)*
                    (self)($($Ts,)*)
                })
            }
        }
        impl<$($Ts: KernelArg +'static),*> KernelDef<fn($($Ts,)*)>  {
            #[allow(non_snake_case)]
            #[allow(unused_variables)]
            pub fn new_maybe_device(device: Option<&Device>, f:&dyn Fn($($Ts::Parameter,)*))->Self {
                KernelBuildFn::build_kernel(&f, &mut KernelBuilder::new(device.cloned(), true))
            }
            pub fn new(device: &Device, f:&dyn Fn($($Ts::Parameter,)*))->Self {
                Self::new_maybe_device(Some(device), f)
            }
            pub fn new_static(f:fn($($Ts::Parameter,)*))->Self {
                Self::new_maybe_device(None, &f)
            }
        }
        impl<$($Ts: KernelArg +'static),*> Kernel<fn($($Ts,)*)> {
            /// Compile a kernel with given recording function `f`.
            pub fn new(device: &Device, f:&dyn Fn($($Ts::Parameter,)*))->Self {
                let def = KernelDef::<fn($($Ts,)*)>::new(device, f);
                device.compile_kernel_def(&def)
            }

            /// Compile a kernel asynchronously with given recording function `f`.
            /// This function returns immediately after `f` returns
            pub fn new_async(device: &Device, f:&dyn Fn($($Ts::Parameter,)*))->Self {
                let def = KernelDef::<fn($($Ts,)*)>::new(device, f);
                device.compile_kernel_def_async(&def)
            }

             // Compile a kernel with given recording function `f` and build options [`KernelBuildOptions`]
            pub fn new_with_options(device: &Device, options: KernelBuildOptions, f:&dyn Fn($($Ts::Parameter,)*))->Self {
                let def = KernelDef::<fn($($Ts,)*)>::new(device, f);
                device.compile_kernel_def_with_options(&def, options)
            }
        }
    };
}

impl_kernel!();
impl_kernel!(T0);
impl_kernel!(T0 T1 );
impl_kernel!(T0 T1 T2 );
impl_kernel!(T0 T1 T2 T3 );
impl_kernel!(T0 T1 T2 T3 T4 );
impl_kernel!(T0 T1 T2 T3 T4 T5 );
impl_kernel!(T0 T1 T2 T3 T4 T5 T6 );
impl_kernel!(T0 T1 T2 T3 T4 T5 T6 T7 );
impl_kernel!(T0 T1 T2 T3 T4 T5 T6 T7 T8 );
impl_kernel!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 );
impl_kernel!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 );
impl_kernel!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 );
impl_kernel!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 );
impl_kernel!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 );
impl_kernel!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 );
impl_kernel!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);

macro_rules! impl_callable_build_for_fn {
    ($($Ts:ident)*) => {
        impl<T, R:CallableRet +'static, $($Ts: CallableParameter),*> CallableBuildFn<fn($($Ts,)*)->R> for T
            where T: Fn($($Ts,)*)->R {
            #[allow(non_snake_case)]
            #[allow(unused_variables)]
            fn build_callable(&self, args: Option<Rc<dyn Any>>, builder: &mut KernelBuilder)->RawCallable {
                builder.build_callable( |builder| {
                    if let Some(args) = args {
                        let ($($Ts,)*) = args.downcast_ref::<($($Ts,)*)>().cloned().unwrap();
                        $(let $Ts = $Ts::def_param(Some(Rc::new($Ts)), builder);)*
                        self($($Ts,)*)
                    } else {
                        $(let $Ts = $Ts::def_param(None, builder);)*
                        self($($Ts,)*)
                    }
                })
            }
        }
        // impl<R:CallableRet +'static, $first:CallableParameter, $($rest: CallableParameter),*> CallableBuildFn for Box<dyn Fn($first, $($rest,)*)->R> {
        //     #[allow(non_snake_case)]
        //     fn build_callable(&self, args: Option<Rc<dyn Any>>, builder: &mut KernelBuilder)->RawCallable {
        //         builder.build_callable( |builder| {
        //             if let Some(args) = args {
        //                 let ($first, $($rest,)*) = args.downcast_ref::<($first, $($rest,)*)>().cloned().unwrap();
        //                 let $first = $first::def_param(Some(Rc::new($first)), builder);
        //                 $(let $rest = $rest::def_param(Some(Rc::new($rest)), builder);)*
        //                 self($first, $($rest,)*)
        //             } else {
        //                 let $first = $first::def_param(None, builder);
        //                 $(let $rest = $rest::def_param(None, builder);)*
        //                 self($first, $($rest,)*)
        //             }
        //         })
        //     }
        // }
        // impl<R:CallableRet +'static, $first:CallableParameter, $($rest: CallableParameter),*> CallableBuildFn for fn($first, $($rest,)*)->R {
        //     #[allow(non_snake_case)]
        //     fn build_callable(&self, args: Option<Rc<dyn Any>>, builder: &mut KernelBuilder)->RawCallable {
        //         builder.build_callable( |builder| {
        //             if let Some(args) = args {
        //                 let ($first, $($rest,)*) = args.downcast_ref::<($first, $($rest,)*)>().cloned().unwrap();
        //                 let $first = $first::def_param(Some(Rc::new($first)), builder);
        //                 $(let $rest = $rest::def_param(Some(Rc::new($rest)), builder);)*
        //                 self($first, $($rest,)*)
        //             } else {
        //                 let $first = $first::def_param(None, builder);
        //                 $(let $rest = $rest::def_param(None, builder);)*
        //                 self($first, $($rest,)*)
        //             }
        //         })
        //     }
        // }
        impl<R:CallableRet +'static, $($Ts: CallableParameter),*> StaticCallableBuildFn<fn($($Ts,)*)->R> for fn($($Ts,)*)->R
        where fn($($Ts,)*)->R : CallableBuildFn<fn($($Ts,)*)->R> {}
    };
}

impl_callable_build_for_fn!();
impl_callable_build_for_fn!(T0);
impl_callable_build_for_fn!(T0 T1 );
impl_callable_build_for_fn!(T0 T1 T2 );
impl_callable_build_for_fn!(T0 T1 T2 T3 );
impl_callable_build_for_fn!(T0 T1 T2 T3 T4 );
impl_callable_build_for_fn!(T0 T1 T2 T3 T4 T5 );
impl_callable_build_for_fn!(T0 T1 T2 T3 T4 T5 T6 );
impl_callable_build_for_fn!(T0 T1 T2 T3 T4 T5 T6 T7 );
impl_callable_build_for_fn!(T0 T1 T2 T3 T4 T5 T6 T7 T8 );
impl_callable_build_for_fn!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 );
impl_callable_build_for_fn!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 );
impl_callable_build_for_fn!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 );
impl_callable_build_for_fn!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 );
impl_callable_build_for_fn!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 );
impl_callable_build_for_fn!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 );
impl_callable_build_for_fn!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15);
