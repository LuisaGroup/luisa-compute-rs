# luisa-compute-rs 
Rust binding to LuisaCompute (WIP)
## Table of Contents
* Example
* Usage

## Example

```rust
use luisa_compute_rs as luisa;
use luisa::prelude::*;
#[derive(Value)]
struct Vec2 {
    x: f32,
    y: f32,
}
#[proxy]
impl Vec2 {
    fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y
    }
}
#[kernel]
fn dot(u: BufferVar<Vec2>, v:  BufferVar<Vec2>, out:BufferVar<f32>) {
    let tid = luisa::dispatch_id().x;
    out.write(tid, u.read(tid).dot(v.read(tid)))
}

```

## Usage
### Variables
```rust
// all variables should not be mut
// otherwise the compute graph will be recorded incorrectly
let v = local(Float::new(0.0));
let shared_v = shared(Float::new(0.0));
// rust cannot overload = so...
v.store(1.0);
// other operators work fine!
v += 1.0;
let u = local(Float2::new(0.0, 0.0));
u.x.store(1.0);


```
### Functions
```rust

#[callable] // optional
pub fn foo(p: Vec2, q:Vec2) -> f32 {
    let mut d = p.x * q.x + p.y * q.y;
    flow!(if d < 0.0 => {
        d = 0.0;
    })
    d
}

let d = foo(p, q); // inlined
let d = call!(foo, p, q); // translate to function call

```

### Structs & Methods
```rust
#[derive(Value)]
struct Ray {
    o: Vec3,
    d: Vec3,
}
#[proxy]
impl Ray::Proxy {
    #[callable]
    fn at(&self, t: Var<f32>) -> Var<Vec3> {
        self.o + self.d * t
    }
}
```

### Kernel
```rust
#[kernel]
fn foo(u: BufferVar<Vec2>, v: BufferVar<Vec2>, out: BufferVar<f32>) {
    let tid = dispatch_id().x;
    out.write(tid, u.read(tid).dot(v.read(tid)));
}
```

### Control Flow
```rust
flow!(if cond {
    // do something
} else {
    // do something else
})

flow!(for i in 0..10 {
    // do something
})

flow!(while cond {
    // do something
})

flow!(match x => { // x must be a Var<i32>
    0 => {
        // do something
    }
    1 => {
        // do something else
    }
    _ => {
        // do something else
    }
})
```

### Polymorphism
```rust
#[polymorphic]
pub trait Area {
    fn area(&self) -> Var<f32>;
}
// Since a trait cannot be generic paraeter, we need to use a macro
type AreaArrayBuilder = polymorphic_array_builder!(Area);
type AreaArray = polymorphic_array!(Area);
type AreaProxy = polymorphic_proxy!(Area); // this is normally not needed

#[derive(Value)]
pub struct Circle {
    r: f32,
}
#[proxy]
impl Area for Circle::Proxy {
    fn area(&self) -> Var<f32> {
        self.r * self.r * PI
    }
}

#[kernel]
fn compute_areas(array: AreaArray::Var, out:BufferVar<f32>) {
    let tid = dispatch_id().x;
    out.write(tid, array.read(tid).area());
}

let mut area_objects = AreaArrayBuilder::new();
area_objects.push(Circle { r: 1.0 });
let area_objects = area_objects.build();


```

### Autodiff
```rust 
autodiff! {
    let x = Var::new(1.0);
    let y = Var::new(2.0);
    requires_grad!(x, y);
    let z = x * x + y * y;
    backward!(z);  // only one backward pass is allowed
    let (dx, dy) = grad!(x, y);
    ...
}

```


### Command 
```rust
let command_buffer = stream.command_buffer();
cmd_submit!(command_buffer, 
    raytrace_shader(framebuffer, accel, resolution)
        .dispatch(resolution),
    accumulate_shader(accum_image, framebuffer)
        .dispatch(resolution),
)

stream


```
