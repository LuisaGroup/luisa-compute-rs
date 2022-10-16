# luisa-compute-rs
Rust binding to LuisaCompute


## Example

```rust
use luisa_compute_rs as luisa;
#[derive(luisa::Value)]
struct Vec2 {
    x: f32,
    y: f32,
}
#[luisa::impl_proxy]
impl Vec2 {
    fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y
    }
}
#[luisa::kernel]
fn dot(u: BufferVar<Vec2>, v:  BufferVar<Vec2>, out:BufferVar<f32>) {
    let tid = luisa::dispatch_id().x;
    out.write(tid, u.read(tid).dot(v.read(tid)))
}

fn main() {
    let ctx = luisa::Context::new();
    let device = ctx.create_device("cuda", json!({}));
}
```