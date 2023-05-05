use std::env::current_exe;
use std::f32::consts::PI;

use luisa::prelude::*;
use luisa::Value;
use luisa::{impl_polymorphic, lang::*};
use luisa_compute as luisa;

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum ShaderDevirtualizationKey {
    ConstShader,
    SinShader,
    AddShader(
        Box<ShaderDevirtualizationKey>,
        Box<ShaderDevirtualizationKey>,
    ),
}
#[derive(Clone, Copy)]
pub struct ShaderEvalContext<'a> {
    pub poly_shader: &'a Polymorphic<ShaderDevirtualizationKey, dyn ShaderNode>,
    pub key: &'a ShaderDevirtualizationKey,
}
pub trait ShaderNode {
    fn evaluate(&self, sp: Expr<f32>, ctx: &ShaderEvalContext<'_>) -> Expr<f32>;
}

#[derive(Value, Clone, Copy)]
#[repr(C)]
pub struct ConstShader {
    value: f32,
}
impl ShaderNode for ConstShaderExpr {
    fn evaluate(&self, _: Expr<f32>, _ctx: &ShaderEvalContext<'_>) -> Expr<f32> {
        self.value()
    }
}
impl_polymorphic!(ShaderNode, ConstShader);
#[derive(Value, Clone, Copy)]
#[repr(C)]
pub struct SinShader {
    _pad: u32,
}
impl ShaderNode for SinShaderExpr {
    fn evaluate(&self, x: Expr<f32>, _ctx: &ShaderEvalContext<'_>) -> Expr<f32> {
        x.sin()
    }
}
impl_polymorphic!(ShaderNode, SinShader);
#[derive(Value, Clone, Copy)]
#[repr(C)]
pub struct AddShader {
    pub shader_a: TagIndex,
    pub shader_b: TagIndex,
}
fn eval_recursive_shader(
    shader: PolymorphicRef<'_, ShaderDevirtualizationKey, dyn ShaderNode>,
    x: Expr<f32>,
    ctx: &ShaderEvalContext<'_>,
) -> Expr<f32> {
    let tag = shader.tag_from_key(ctx.key).unwrap();
    shader.unwrap(tag, |key, shader| {
        assert_eq!(key, ctx.key);
        shader.evaluate(x, ctx)
    })
}
impl ShaderNode for AddShaderExpr {
    fn evaluate(&self, x: Expr<f32>, ctx: &ShaderEvalContext<'_>) -> Expr<f32> {
        let key = ctx.key;
        match key {
            ShaderDevirtualizationKey::AddShader(a, b) => {
                let shader_a = ctx.poly_shader.get(self.shader_a());
                let shader_b = ctx.poly_shader.get(self.shader_b());
                let value_a = eval_recursive_shader(
                    shader_a,
                    x,
                    &ShaderEvalContext {
                        poly_shader: ctx.poly_shader,
                        key: a.as_ref(),
                    },
                );
                let value_b = eval_recursive_shader(
                    shader_b,
                    x,
                    &ShaderEvalContext {
                        poly_shader: ctx.poly_shader,
                        key: b.as_ref(),
                    },
                );
                value_a + value_b
            }
            _ => unreachable!(),
        }
    }
}
impl_polymorphic!(ShaderNode, AddShader);
fn main() -> luisa::Result<()> {
    use luisa::*;
    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_device("cpu")?;
    let mut builder =
        PolymorphicBuilder::<ShaderDevirtualizationKey, dyn ShaderNode>::new(device.clone());
    // build shader = sin(x) + (1.0 + 2.0)
    let shader_const_1 = builder.push(
        ShaderDevirtualizationKey::ConstShader,
        ConstShader { value: 1.0 },
    );
    let shader_const_2 = builder.push(
        ShaderDevirtualizationKey::ConstShader,
        ConstShader { value: 2.0 },
    );
    let shader_sin = builder.push(ShaderDevirtualizationKey::SinShader, SinShader { _pad: 0 });
    let shader_add_1_2_key = ShaderDevirtualizationKey::AddShader(
        Box::new(ShaderDevirtualizationKey::ConstShader),
        Box::new(ShaderDevirtualizationKey::ConstShader),
    );
    let shader_add_1_2 = builder.push(
        shader_add_1_2_key.clone(),
        AddShader {
            shader_a: shader_const_1,
            shader_b: shader_const_2,
        },
    );
    let shader_final_key = ShaderDevirtualizationKey::AddShader(
        Box::new(ShaderDevirtualizationKey::SinShader),
        Box::new(shader_add_1_2_key),
    );
    let shader_final = builder.push(
        shader_final_key.clone(),
        AddShader {
            shader_a: shader_sin,
            shader_b: shader_add_1_2,
        },
    );
    let poly_shader = builder.build()?;
    let result = device.create_buffer::<f32>(100).unwrap();
    let kernel = device
        .create_kernel::<()>(&|| {
            let i = dispatch_id().x();
            let x = i.float() / 100.0 * PI;
            let ctx = ShaderEvalContext {
                poly_shader: &poly_shader,
                key: &shader_final_key,
            };
            let tag_index = TagIndexExpr::new(shader_final.tag, shader_final.index);
            let v = poly_shader
                .get(tag_index)
                .dispatch(|_, _, shader| shader.evaluate(x, &ctx));
            result.var().write(i, v);
        })
        .unwrap();
    kernel.dispatch([100, 1, 1]).unwrap();
    let result = result.copy_to_vec();
    for i in 0..100 {
        let x = i as f32 / 100.0 * PI;
        let v = x.sin() + (1.0 + 2.0);
        assert!((result[i] - v).abs() < 1e-5);
    }
    println!("OK");
    Ok(())
}
