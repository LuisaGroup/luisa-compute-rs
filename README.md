# luisa-compute-rs
Rust frontend to LuisaCompute and more!
## Table of Contents
* [Overview](#overview)
    + [Embedded Domain-Specific Language](#embedded-domain-specific-language)
    + [Automatic Differentiation](#automatic-differentiation)
    + [A CPU backend](#cpu-backend)
    + [IR Module for EDSL](#ir-module)
    + [Debuggability](#debuggability)
* [Usage](#usage)
    + [Variables and Expressions](#variables-and-expressions)
    + [Builtin Functions](#builtin-functions)
    + [Control Flow](#control-flow)
    + [Custom Data Types](#custom-data-types)
    + [Polymorphism](#polymorphism)
    + [Autodiff](#autodiff)
    + [Custom Operators](#custom-operators)
    + [Callable](#callable)
    + [Kernel](#kernel)
* [Advanced Usage](#advanced-usage)
* [Safety](#safety)
* [Citation](#citation)

## Example
Try `cargo run --release --example path_tracer`!
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
    let kernel = device.create_kernel::<(Buffer<f32>,)>(&|buf_z| {
        // z is pass by arg
        let buf_x = x.var(); // x and y are captured
        let buf_y = y.var();
        let tid = dispatch_id().x();
        let x = buf_x.read(tid);
        let y = buf_y.read(tid);
        let vx = var!(f32); // create a local mutable variable
        *vx.get_mut() += x;
        buf_z.write(tid, vx.load() + y);
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
To get started, add the following to your `Cargo.toml`:
```toml
[dependencies]
luisa_compute = { git= "https://github.com/LuisaGroup/luisa-compute-rs.git"}
```
Then added the following to your files:
```rust
use luisa_compute as luisa;
use luisa::prelude::*;
```
### Variables and Expressions
There are six basic types in EDSL. `bool`, `i32`, `u32`, `i64`, `u64`, `f32`. (`f64` support might be added to CPU backend).
For each type, there are two EDSL proxy objects `Expr<T>` and `Var<T>`. `Expr<T>` is an immutable object that represents a value. `Var<T>` is a mutable object that represents a variable. To load values from `Var<T>`, use `*var` and to obtain a mutable reference for assignment, use `v.get_mut()`. E.g. `*v.get_mut() = f(*u)`.

*Note*: Every DSL object in host code **must** be immutable due to Rust unable to overload. For example:
```rust
// **no good**
let mut v = const_(0.0f32);
if_!(cond, {
    v += 1.0;
});

// also **not good**
let v = Cell::new(const_(0.0f32));
if_!(cond, {
    v.set(v.get() + 1.0);
});

// **good**
let v = var!(f32);
if_!(cond, {
    *v.get_mut() += 1.0;
});
```
*Note*: You should not store the referene obtained by `v.get_mut()` for repeated use, as the assigned value is only updated when `v.get_mut()` is dropped. For example,:
```rust
let v = var!(f32);
let bad = v.get_mut();
*bad = 1.0;
let u = *v;
drop(bad);
cpu_dbg!(u); // prints 0.0
cpu_dbg!(*v); // prints now 1.0
```
All operations except load/store should be performed on `Expr<T>`. `Var<T>` can only be used to load/store values. While `Expr<T>` and `Var<T>` are sufficent in most cases, it cannot be placed in an `impl` block. To do so, the exact name of these proxies are needed.
```rust
Expr<Bool> == Bool, Var<Bool> == BoolVar
Expr<f32> == Float32, Var<f32> == Float32Var
Expr<i32> == Int32, Var<i32> == Int32Var
Expr<u32> == UInt32, Var<u32> == UInt32Var
Expr<i64> == Int64, Var<i64> == Int64Var
Expr<u64> == UInt64, Var<u64> == UInt64Var
```

As in the C++ EDSL, we additionally supports the following vector/matrix types. Their proxy types are `XXXExpr` and `XXXVar`:

```rust
Bool2 // bool2 in C++
Bool3 // bool3 in C++
Bool4 // bool4 in C++
Vec2 // float2 in C++
Vec3 // float3 in C++
Vec4 // float4 in C++
Int2 // int2 in C++
Int3 // int3 in C++
Int4 // int4 in C++
Uint2 // uint2 in C++
Uint3 // uint3 in C++
Uint4 // uint4 in C++
Mat2 // float2x2 in C++
Mat3 // float3x3 in C++
Mat4 // float4x4 in C++
```
Array types `[T;N]` are also supported and their proxy types are `ArrayExpr<T, N>` and `ArrayVar<T, N>`. Call `arr.read(i)` and `arr.write(i, value)` on `ArrayVar<T, N>` for element access. `ArrayExpr<T,N>` can be stored to and loaded from `ArrayVar<T, N>`. The limitation is however the array length must be determined during host compile time. If runtime length is required, use `VLArrayVar<T>`. `VLArrayVar<T>::zero(length: usize` would create a zero initialized array. Similarly you can use `read` and `write` methods as well. To query the length of a `VLArrayVar<T>` in host, use ``VLArrayVar<T>::static_len()->usize`. To query the length in kernel, use ``VLArrayVar<T>::len()->Expr<u32>`

Most operators are already overloaded with the only exception is comparision. We cannot overload comparision operators as `PartialOrd` cannot return a DSL type. Instead, use `cmpxx` methods such as `cmpgt, cmpeq`, etc. To cast a primitive/vector into another type, use `v.type()`. For example:
```rust
let iv = make_int2(1,1,1);
let fv = iv.float(); //fv is Expr<Float2>
let bv = fv.bool(); // bv is Expr<Bool2>
```
To perform a bitwise cast, use the `bitcast` function. `let fv:Expr<f32> = bitcast::<u32, f32>(const_(0u32));`

### Builtin Functions

We have extentded primitive types with methods similar to their host counterpart: `v.sin(), v.max(u)`, etc. Most methods accepts both a `Expr<T>` or a literal like `0.0`. However, the `select` function is slightly different as it do not accept literals. You need to use `select(cond, f_var, const_(1.0f32))`.


### Control Flow
*Note*, you cannot modify outer scope variables inside a control flow block by declaring the variable as `mut`. To modify outer scope variables, use `Var<T>` instead and call *var.get_mut() = value` to store the value back to the outer scope.

If, while, break, continue are supported. Note that `if` and `switch` works similar to native Rust `if` and `match` in that values can be returned at the end of the block.


```rust
if_!(cond, { /* then */});
if_!(cond, { /* then */}, { /* else */});
if_!(cond, { value_a }, { value_b })
while_!(cond, { /* body */});
for_range(start..end, |i| { /* body */});
/* For loops in C-style are mapped to generic loops
for(init; cond; update) { body } is mapped to:
init;
generic_loop(cond, body, update)
*/
generic_loop(|| -> Expr<bool>{ /*cond*/ }, || { /* body */}, || { /* update after each iteration */})
break_();
continue_();
let (x,y) = switch::<(Expr<i32>, Expr<f32>)>(value)
    .case(1, || { ... })
    .case(2, || { ... })
    .default(|| { ... })
    .finish();
```

### Custom Data Types
To add custom data types to the EDSL, simply derive from `luisa::Value` macro. Note that `#[repr(C)]` is required for the struct to be compatible with C ABI. The proxy types are `XXXExpr` and `XXXVar`:

```rust
#[derive(Copy, Clone, Default, Debug, Value)]
#[repr(C)]
pub struct MyVec2 {
    pub x: f32,
    pub y: f32,
}

let v = var!(MyVec2);
let sum = *v.x() + *v.y(); 
*v.x().get_mut() += 1.0;
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
        PI * self.radius() * self.radius()
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
let add = device.create_callable::<fn(Expr<f32>, Expr<f32>)-> Expr<f32>>(&|a, b| {
    a + b
});
let z = add.call(x, y);
let pass_by_ref = device.create_callable::<fn(Var<f32>)>(&|a| {
    *a.get_mut() += 1.0;
});
let a = var!(f32, 1.0);
pass_by_ref.call(a);
cpu_dbg!(*a); // prints 2.0
```
***Note***: You cannot record a callable when recording another kernel or callables. This is because a callable can capture outer variables such as buffers. However, capturing local variables define in another callable is undefined behavior. To avoid this, we disallow recording a callable when recording another callable or kernel. 
```rust
let add = device.create_callable::<fn(Expr<f32>, Expr<f32>)-> Expr<f32>>(&|a, b| {
    // runtime error!
    let another_add = device.create_callable::<fn(Expr<f32>, Expr<f32>)-> Expr<f32>>(&|a, b| {
        a + b
    });
    a + b
});
```

***However, we acknowledge that recording a callable inside another callable/kernel is a useful feature***. Thus we provide two ways to workaround this limitation:
1. Use static callables. A static callable does not capture any resources and thus can be safely recorded inside any callable/kernel. To create a static callable, use `create_static_callable(fn)`. For example,
```rust
lazy_static! {
    static ref ADD:Callable<fn(Expr<f32>, Expr<f32>)->Expr<f32>> = create_static_callable::<fn(Expr<f32>, Expr<f32>)->Expr<f32>>(|a, b| {
    a + b
});
}
ADD.call(x, y);
```

2. Use `DynCallable`. These are callables that defer recording until being called. As a result, it requires you to pass a `'static` closure, avoiding the capture issue. To create a `DynCallable`, use `Device::create_dyn_callable(Box::new(fn))`. The syntax is the same as `create_callable`. Furthermore, `DynCallable` supports `DynExpr` and `DynVar`, which provides some capablitiy of implementing template/overloading inside EDSL.

```rust
let add = device.create_callable::<fn(Expr<f32>, Expr<f32>)->Expr<f32>>(&|a, b| {
    // no error!
    let another_add = device.create_dyn_callable::<fn(Expr<f32>, Expr<f32>)->Expr<f32>>(Box::new(|a, b| {
        a + b
    }));
    a + b
});
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
    author = {Xiaochun Tong},
    year = {2023},
    note = {https://github.com/LuisaGroup/luisa-compute-rs},
    title = {Rust frontend to LuisaCompute}
}
```
