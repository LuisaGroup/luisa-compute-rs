# luisa-compute-rs
Rust frontend to LuisaCompute and more! Unified API and embedded DSL for high performance computing on stream architectures. 

*Warning:* while the project is already usable, it is not stable and **breaking changes** can happend at any time without notification.

To see the use of `luisa-compute-rs` in a high performance offline rendering system, checkout [our research renderer](https://github.com/shiinamiyuki/akari_render)
## Table of Contents
- [luisa-compute-rs](#luisa-compute-rs)
  - [Table of Contents](#table-of-contents)
  - [Example](#example)
    - [Vecadd](#vecadd)
  - [Overview](#overview)
    - [Embedded Domain-Specific Language](#embedded-domain-specific-language)
    - [Automatic Differentiation](#automatic-differentiation)
    - [CPU Backend](#cpu-backend)
    - [IR Module](#ir-module)
    - [Debuggability](#debuggability)
  - [Usage](#usage)
    - [Building](#building)
    - [`track!` and #[tracked] Macro](#track-and-tracked-macro)
    - [Variables and Expressions](#variables-and-expressions)
    - [Builtin Functions](#builtin-functions)
    - [Control Flow](#control-flow)

    - [Custom Data Types](#custom-data-types)
    - [Polymorphism](#polymorphism)
    - [Autodiff](#autodiff)
    - [Custom Operators](#custom-operators)
    - [Callable](#callable)
    - [Kernel](#kernel)
    - [Debugging](#debugging)
  - [Advanced Usage](#advanced-usage)
  - [Safety](#safety)
    - [API](#api)
    - [Backend](#backend)
  - [Citation](#citation)

## Example
Try `cargo run --release --example path_tracer -- [cpu|cuda|dx|metal]`!

### Vecadd
```rust
use luisa::prelude::*;
use luisa_compute as luisa;

fn main() {
    use luisa::*;
    init_logger();
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() <= 2,
        "Usage: {} <backend>. <backend>: cpu, cuda, dx, metal, remote",
        args[0]
    );

    let ctx = Context::new(current_exe());
    let device = ctx.create_device(if args.len() == 2 {
        args[1].as_str()
    } else {
        "cpu"
    });
    let x = device.create_buffer::<f32>(1024);
    let y = device.create_buffer::<f32>(1024);
    let z = device.create_buffer::<f32>(1024);
    x.view(..).fill_fn(|i| i as f32);
    y.view(..).fill_fn(|i| 1000.0 * i as f32);
    let kernel = Kernel::<fn(Buffer<f32>)>::new(&device, |buf_z| {
        // z is pass by arg
        let buf_x = x.var(); // x and y are captured
        let buf_y = y.var();
        let tid = dispatch_id().x;
        let x = buf_x.read(tid);
        let y = buf_y.read(tid);
        let vx = 0.0f32.var(); // create a local mutable variable
        *vx += x;
        buf_z.write(tid, vx + y);
    });
    kernel.dispatch([1024, 1, 1], &z);
    let z_data = z.view(..).copy_to_vec();
    println!("{:?}", &z_data[0..16]);
}


```
Other examples in [examples](luisa_compute/examples)

| Example | Description |
| ----------- | ----------- |
| [Atomic](luisa_compute/examples/atomic.rs) | Atomic buffer operations |
| [Bindless](luisa_compute/examples/bindless.rs) | Bindless array access |
| [Custom Aggregate](luisa_compute/examples/custom_aggregate.rs) | Use #[derive(Aggregate)] for kernel only data types|
| [Custom Op](luisa_compute/examples/custom_op.rs) | Custom operator for CPU backend |
| [Polymporphism](luisa_compute/examples/polymorphism.rs) | Simple usage of Polymorphic<K, T> |
| [Advanced Polymporphism](luisa_compute/examples/polymorphism_advanced.rs) | Use Polymorphic<K, T> to implement recursive polymorphic call|
| [Ray Tracing](luisa_compute/examples/raytracing.rs) | A simple raytracing kernel with GUI|
| [Path Tracer](luisa_compute/examples/path_tracer.rs) | A path tracer with GUI|

## Overview
### Embedded Domain-Specific Language
We provided an Rust-flavored implementation of LuisaCompute EDSL that tightly integrates with Rust language via traits and proc-macros.

### Automatic Differentiation
We implemented a source-to-source reverse mode automatic differentiation that supports complex control flow.

The autodiff works tightly with builtin functions and the type system. Instead of implementing every function using basic arithmetic operations and apply autodiff on it, all builtin functions are differentiated using efficient VJP formulae.

### CPU Backend
This crate also provides a CPU backend implementation in Rust that will eventually replace the current LLVM backend in LuisaCompute. This backend emphasizes on debuggability, flexibility and as well as safety.

### IR Module
The EDSL and code generation are built atop of an SSA-based IR module. The IR module is in a separate crate and can be used to implement other backends and IR transformation such as autodiff.

### Debuggability
The CPU backend is designed to be debuggable. If needed, it will perform runtime checks to detect errors such as out-of-bound access, bindless array type mismatch, etc. It will display error message containing the **host** stacktrace for pinpointing the error location.

## Usage
### Building
To try out the examples, clone the repo using `git clone --recursive https://github.com/LuisaGroup/luisa-compute-rs.git`.

To use it in your project, add the following to your `Cargo.toml`:
```toml
[dependencies]
luisa_compute = { git= "https://github.com/LuisaGroup/luisa-compute-rs.git"}
```
You need to install `CMake` and `Ninja` to build the backends. More details about prerequistes can be found in [here](https://github.com/LuisaGroup/LuisaCompute#building).

In your project, the following to your files:
```rust
use luisa_compute as luisa;
use luisa::prelude::*;
```

### `track!` and `#[tracked]` Macro
To start writing using DSL, let's first introduce the `track!` macro. `track!( expr )` rewrites `expr` and redirect operators/control flows to DSL's internal traits. It resolves the fundamental issue that Rust is unable to overload `operator=`.

**Every operation involving a DSL object must be enclosed within `track!`**, except `Var<T>::store()` and `Var<T>::load()`

For, example:
```rust
let a = 1.0f32.expr();
let b = 1.0f32.expr();
let c = a + b; // Compile error

let c = track!(a + b); // c is now 2.0

// Or even better,
track!({
  let a = 1.0f32.expr();
  let b = 1.0f32.expr();
  let c = a + b;
});
```
We also offer a `#[tracked]` macro that applies to a function. It transform the body of the function using `track!`.
 ```rust
#[tracked]
fn add(a:Expr<f32>, b:Expr<f32>)->Expr<f32> {
  a + b
}

However, not every kernel can be constructed using `track!` code only. We still need the ability to use native control flow directly in kernel. 

For example, we can use native `for` loops to unroll a DSL loop. We first starts with a native version using DSL loops.
```rust
#[tracked]
fn pow_naive(x:Expr<f32>, i:u32)->Expr<f32> {
  let p = 1.0f32.var();
  for _ in 0..i {
    p *= x;
  }
  **p // converts Var<f32> to Expr<f32>, only required when passing a Var<T> to fn(Expr<T>) and return from fn(...)->Expr<T>
}
```
To unroll the loop, we basically just what the DSL to produce `p*=x` for `i` times. We can use the `escape!(expr)` macro so that it leaves `expr` as is, preserving the native loop. 
```rust
#[tracked]
fn pow_unrolled(x:Expr<f32>, i:u32)->Expr<f32> {
  let p = 1.0f32.var();
  escape!({
    for _ in 0..i {
      track!({
        p *= x;
      });
  });
  **p 
}
```
Of course this can be tedius if you just want to unroll a loop. Thus we provide a `for_unrolled` function that unrolls a loop for you. 
```rust
#[tracked]
fn pow_unrolled(x:Expr<f32>, i:u32)->Expr<f32> {
  let p = 1.0f32.var();
  for_unrolled(0..i, |_|{
      p *= x;
  });
  **p 
}
```



### Variables and Expressions
We support the following primitive types on backend `bool`, `i32`, `u32`, `i64`, `u64`, `f32`. Additional primitive types such as `u8`, `i8`, `i16`, `u16`, and `f64` are supported on some backends.
For each type, there are two EDSL proxy objects `Expr<T>` and `Var<T>`. `Expr<T>` is an immutable object that represents a value. `Var<T>` is also an **immutable** object that represents a variable (mutable value). 

**Warning**: Every DSL object in host code **must** be immutable due to Rust unable to overload `operator =`. Attempting to circumvent this limitation using `Cell` and `RefCell` would likely result in uncompilable kernels/wrong results.
For example:
```rust
// **no good**
let v = Cell::new(0.0f32.expr());
track!(if cond {
  v.set(v.get() + 1.0);
));

// **good**
let v = 0.0f32.var();
track!(if cond {
  *v += 1.0;
));

All operations except load/store should be performed on `Expr<T>`. `Var<T>` can only be used to load/store values.

As in the C++ EDSL, we additionally supports the vector of length 2-4 for all primitives and float square matrices with dimension 2-4 such as 

```rust
luisa_compute::lang::types::vector::alias::{
  Bool2 
  Bool3 
  Bool4 
  Float2 
  Float3 
  Float4
  Int2
  Int3 
  Int4 
  Uint2
  Uint3 
  Uint4
};

luisa_compute::lang::types::vector::{Mat2, Mat3, Mat4};
```

Array types `[T;N]` are also supported. Call `arr.read(i)` and `arr.write(i, value)` on `ArrayVar<T, N>` for element access. `ArrayExpr<T,N>` can be stored to and loaded from `ArrayVar<T, N>`. The limitation is however the array length must be determined during host compile time. If runtime length is required, use `VLArrayVar<T>`. `VLArrayVar<T>::zero(length: usize)` would create a zero initialized array. Similarly you can use `read` and `write` methods as well. To query the length of a `VLArrayVar<T>` in host, use `VLArrayVar<T>::static_len()->usize`. To query the length in kernel, use `VLArrayVar<T>::len()->Expr<u32>`

Most operators are already overloaded with the only exception is comparision. We cannot overload comparision operators as `PartialOrd` cannot return a DSL type. Instead, use `cmpxx` methods such as `cmpgt, cmpeq`, etc. To cast a primitive/vector into another type, use `v.as_::<Type>()`, `v.as_Type()` and `v.as_PrimitiveType()`. For example:
```rust
let iv = Int2::expr(1, 1, 1);
let fv = iv.as_::<Float2>(); //fv is Expr<Float2>
let also_fv = iv.as_float2();
let also_fv = iv.cast_f32(); 
```
To perform a bitwise cast, use the `bitcast` function. `let fv:Expr<f32> = bitcast::<u32, f32>(0u32);`

### Builtin Functions

We have extentded primitive types with methods similar to their host counterpart: `v.sin(), luisa::max(a, b)`, etc. Most methods accepts both a `Expr<T>` or a literal such as `0.0`. However, the `select` function is slightly different as it does not accept literals. You need to use `select(cond, f_var, 1.0f32.expr())`.

### Control Flow
*Note*, you cannot modify outer scope variables inside a control flow block by declaring the variable as `mut`. To modify outer scope variables, use `Var<T>` instead and store the value back to the outer scope.

`if`, `while`, `break`, `continue`, `return` and `loop` are supported via `tracked!` macro. It is also possible to construct these control flows without `track!`. 

The `switch_` statement has to be constructe manually inside a `escape!` block. For example,
```rust
let (x,y) = switch::<(Expr<i32>, Expr<f32>)>(value)
    .case(1, || { ... })
    .case(2, || { ... })
    .default(|| { ... })
    .finish();
```

**Warning**:  due to backend generates C-like source code instead of LLVM IR/PTX/DXIL directly, it is not possible to use `break` inside switch cases.

### Custom Data Types
To add custom data types to the EDSL, simply derive from `Value` macro. Note that `#[repr(C)]` is required for the struct to be compatible with C ABI.
`#[derive(Value)]` would generate two proxies types: `XXExpr` and `XXVar`. Implement your methods on these proxies instead of `Expr<T>` and `Var<T>` directly.

```rust
#[derive(Copy, Clone, Default, Debug, Value)]
#[repr(C)]
#[value_new(pub)]
pub struct MyVec2 {
    pub x: f32,
    pub y: f32,
}

impl MyVec2Expr {
  // pass arguments using `AsExpr` so that they accept both Var and Expr
  #[tracked]
  pub fn dot(&self, other: impl AsExpr<Value=MyVec2>) {
    self.x * other.x + self.y * other.y
  }
}
impl MyVec2Var {
  #[tracked]
  pub fn set_to_one(&self) {
    // you can access the current `Var<Self>` using `self_`
    self.self_ = MyVec2::new_expr(1.0, 1.0);
  }
}

track!({
  let v = MyVec2::var_zeroed();
  let sum = v.x +*v.y; 
  *v.x += 1.0;
  let v = MyVec2::from_comps_expr(MyVec2Comps{x:1.0f32.expr(), y:1.0f32.expr()});
  let v = MyVec2::new_expr(1.0f32, 2.0f32); // only if #[value_new] is present
});

// You can also control the order of arguments in `#[value_new]`
#[derive(Copy, Clone, Default, Debug, Value)]
#[repr(C)]
#[value_new(pub y, x)]
pub struct Foo {
    pub x: f32,
    pub y: i32,
}
let v = MyVec2::new_expr(1.0fi32, 2.0f32);
// v.x == 2.0
// v.y == 1
```

### Polymorphism
We prvoide a powerful `Polymorphic<DevirtualizationKey, dyn Trait>` construct as in the C++ DSL. See examples for more detail
```rust
trait Area {
    fn area(&self) -> Float32;
}
#[derive(Value, Clone, Copy)]
#[repr(C)]
pub struct Circle {
    radius: f32,
}
impl Area for CircleExpr {
    fn area(&self) -> Float32 {
        PI * self.radius * self.radius
    }
}
impl_polymorphic!(Area, Circle);

let circles = device.create_buffer(..);
let mut poly_area: Polymorphic<(), dyn Area> = Polymorphic::new();
poly_area.register((), &circles);
let area = poly_area.dispatch(tag, index, |obj|{
    obj.area()
});
```

### Autodiff
Autodiff code should be enclosed in the `autodiff` call. The `requires_grad` call is used to mark the variables that need to be differentiated. Any type including user defined ones can receive gradients. The `backward` call triggers the backward pass. Subsequent calls to `gradient` will return the gradient of the variable passed in. User can also supply custom gradients with `backward_with_grad`.

Note: Only one backward call is allowed in a single autodiff block. The autodiff block does not return any value. To store any side effects, use of local variables or buffers is required.

```rust
autodiff(||{
    let v: Expr<Vec3> = buf_v.read(..);
    let m: Expr<Mat3> = buf_m.read(..);
    requires_grad(v);
    requires_grad(m);
    let z = v.dot(m * v) * 0.5;
    backward(z);
    let dv = gradient(dv);
    let dm = gradient(dm);
    buf_dv.write(.., dv);
    buf_dm.write(.., dm);
});
```

### Custom Operators
LuisaCompute supports injecting arbitrary user code to implement a custom operator. This is handled differently on different backends.
On CPU backends, user can directly pass a closure to the kernel. The closure needs to have a `Fn(&mut T)` signature where it modifies the argument inplace. The EDSL frontend would then wrap the closure into a `T->T` function object.

```rust
#[derive(Clone, Copy, Value, Debug)]
#[repr(C)]
pub struct MyAddArgs {
    pub x: f32,
    pub y: f32,
    pub result: f32,
}
let my_add = CpuFn::new(|args: &mut MyAddArgs| {
    args.result = args.x + args.y;
});

let args = MyAddArgsExpr::new(x, y, Float32::zero());
let result = my_add.call(args);
```

### Callable
Users can define device-only functions using Callables. Callables have similar type signature to kernels: `Callable<fn(Args)->Ret>`. 
The difference is that Callables are not dispatchable and can only be called from other Callables or Kernels. Callables can be created using `Device::create_callable`. To invoke a Callable, use `Callable::call(args...)`. Callables accepts arguments such as resources (`BufferVar<T>`, .etc), expressions and references (pass a `Var<T>` to the callable). For example:
```rust
let add = Callable::<fn(Expr<f32>, Expr<f32>)-> Expr<f32>>::new(&device, track!(|a, b| {
    a + b
}));
let z = add.call(x, y);
let pass_by_ref =  Callable::<fn(Var<f32>)>::new(&device, track!(|a| {
   a += 1.0;
}));
let a = 1.0f32.var();
pass_by_ref.call(a);
cpu_dbg!(*a); // prints 2.0
```
***Note***: You cannot record a callable when recording another kernel or callables. This is because a callable can capture outer variables such as buffers. However, capturing local variables define in another callable is undefined behavior. To avoid this, we disallow recording a callable when recording another callable or kernel. 
```rust
let add = Callable::<fn(Expr<f32>, Expr<f32>)-> Expr<f32>>::new(&device, track!(|a, b| {
    // runtime error!
    let another_add = Callable::<fn(Expr<f32>, Expr<f32>)-> Expr<f32>>::new(&device, track!(|a, b| {
        a + b
    }));
    a + b
}));
```

***However, we acknowledge that recording a callable inside another callable/kernel is a useful feature***. Thus we provide two ways to workaround this limitation:
1. Use static callables. A static callable does not capture any resources and thus can be safely recorded inside any callable/kernel. To create a static callable, use `create_static_callable(fn)`. For example,
```rust
lazy_static! {
    static ref ADD:Callable<fn(Expr<f32>, Expr<f32>)->Expr<f32>> = Callable::<fn(Expr<f32>, Expr<f32>)->Expr<f32>>::new_static(|a, b| {
    track!(a + b)
});
}
ADD.call(x, y);
```

2. Use `DynCallable`. These are callables that defer recording until being called. As a result, it requires you to pass a `'static` closure, avoiding the capture issue. To create a `DynCallable`, use `Device::create_dyn_callable(Box::new(fn))`. The syntax is the same as `create_callable`. Furthermore, `DynCallable` supports `DynExpr` and `DynVar`, which provides some capablitiy of implementing template/overloading inside EDSL.

```rust
let add = Callable::<fn(Expr<f32>, Expr<f32>)-> Expr<f32>>::new(&device, track!(|a, b| {
    // no error!
    let another_add = DynCallable::<fn(Expr<f32>, Expr<f32>)-> Expr<f32>>::new(&device, track!(Box::new(|a, b| {
        a + b
    })));
    a + b
}));
```

### Kernel
A kernel can be written in a closure or a function. The closure/function should have a `Fn(/*args*/)->()` signature, where the args are taking the `Var` type of resources, such as `BufferVar<T>`, `Tex2D<T>`, etc.


```rust
let kernel = device.create_kernel::<fn(Arg0, Arg1, ...)>(&|/*args*/| {
    /*body*/
});
kernel.dispatch([/*dispatch size*/], &arg0, &arg1, ...);
```
There are two ways to pass arguments to a kernel: by arguments or by capture.
```rust
let captured:Buffer<f32> = device.create_buffer(...);
let kernel = device.create_kernel::<fn(BufferVar<f32>>(arg| {
    let v = arg.read(..);
    let u = captured.var().read(..);
}));
```
User can pass a maximum of 16 arguments to kernel and unlimited number of captured variables. If more than 16 arguments are needed, user can pack them into a struct and pass the struct as a single argument.
```rust
#[derive(BindGroup)]
pub struct BufferPair {
    a:Buffer<f32>,
    b:Buffer<f32>
}
let kernel = device.create_kernel::<fn(BufferPair)>(&|| {
    // ...
});
let a = device.create_buffer(...);
let b = device.create_buffer(...);
let pair = BufferPair{a,b};
kernel.dispatch([...], &packed);
let BufferPair{a, b} = packed; // unpack if you need to use them later
```
### Debugging
We provide logging through the `log` crate. Users can either setup their own logger or use the `init_logger()` and `init_logger_verbose()` for handy initialization.
For `debug` builds,  oob checks are automatically inserted so that an assertion failure would occur if oob access is detected. On CPU backend, it will be accompanied by an informative message such as `assertion failed: i.cmplt(self.len()) at xx.rs:yy:zz`. Setting the environment variable `LUISA_BACKTRACE=1` would display a stacktrace containing the *DSL* code that records the kernel. For other backends, assertion with message is still *WIP*.

For `release` builds however, these checks are disabled by default for performance reasons. To enable them, set environment variable `LUISA_DEBUG=1` prior to launching the application.

## Advanced Usage
Note that the IR module has a public interface. If needed, user can implement their own DSL syntax sugar. Every EDSL object implements either `Aggregate` or `FromNode` trait, which allows any EDSL type to be destructured into its underlying IR nodes and reconstructed from them.

TODO

## Safety
### API
Host-side safety: The API aims to be 100% safe on host side. However, the safety of async operations are gauranteed via staticly know sync points (such as `Stream::submit_and_sync`). If fully dynamic async operations are needed, user need to manually lift the liftime and use unsafe code accordingly.

Device-side safety: Due to the async nature of device-side operations. It is both very difficult to propose a safe **host** API that captures **device** resource lifetime. While device-side safety isn't guaranteed at compile time, on `cpu` backend runtime checks will catch any illegal memory access/racing condition during execution. However, for other backends such check is either too expensive or impractical and memory errors would result in undefined behavior instead.

### Backend
Safety checks such as OOB is generally not available for GPU backends. As it is difficult to produce meaningful debug message in event of a crash. However, the Rust backend provided in the crate contains full safety checks and is recommended for debugging.

## Citation
When using luisa-compute-rs in an academic project, we encourage you to cite
```bibtex
@misc{LuisaComputeRust
    author = {Xiaochun Tong, et al},
    year = {2023},
    note = {https://github.com/LuisaGroup/luisa-compute-rs},
    title = {Rust frontend to LuisaCompute}
}
```
