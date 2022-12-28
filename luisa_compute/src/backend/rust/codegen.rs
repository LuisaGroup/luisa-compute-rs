use ir::Type;
use luisa_compute_ir::ir::{self, NodeRef, Primitive, SwitchCase, VectorElementType};
use luisa_compute_ir::Gc;
use std::collections::HashMap;
use std::ffi::CString;
use std::fmt::Write;

use super::sha256;
use super::shader::{MATH_LIB_SRC, SHADER_LIB_SRC};
pub(crate) struct TypeGen {
    cache: HashMap<Gc<Type>, String>,
    struct_typedefs: String,
}

impl TypeGen {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
            struct_typedefs: String::new(),
        }
    }
    fn to_rust_type_(&mut self, t: Gc<Type>) -> String {
        match t.as_ref() {
            Type::Primitive(t) => match t {
                ir::Primitive::Bool => "bool".to_string(),
                ir::Primitive::Int32 => "i32".to_string(),
                ir::Primitive::Uint32 => "u32".to_string(),
                ir::Primitive::Int64 => "i64".to_string(),
                ir::Primitive::Uint64 => "u64".to_string(),
                ir::Primitive::Float32 => "f32".to_string(),
                ir::Primitive::Float64 => "f64".to_string(),
                // crate::ir::Primitive::USize => format!("i{}", std::mem::size_of::<usize>() * 8),
            },
            Type::Void => "()".to_string(),
            Type::Struct(st) => {
                let field_types: Vec<String> = st
                    .fields
                    .as_ref()
                    .iter()
                    .map(|f| self.to_rust_type(*f))
                    .collect();
                let field_types_str = field_types.join(", ");
                let hash = sha256(&format!("{}_alignas({})", field_types_str, st.alignment));
                let hash = hash.replace("-", "x_");
                let name = format!("s_{}", hash);

                self.cache.insert(t, name.clone());
                let mut tmp = String::new();
                writeln!(
                    tmp,
                    "#[repr(C, align({}))]#[derive(Copy, Clone)]\nstruct {} {{",
                    st.alignment, name
                )
                .unwrap();
                for (i, field) in st.fields.as_ref().iter().enumerate() {
                    let field_name = format!("f{}", i);
                    let field_type = self.to_rust_type(*field);
                    writeln!(tmp, "    {}: {},", field_name, field_type).unwrap();
                }
                writeln!(tmp, "}}").unwrap();
                self.struct_typedefs.push_str(&tmp);
                name
            }
            Type::Vector(vt) => {
                let n = vt.length;
                match vt.element {
                    VectorElementType::Scalar(s) => match s {
                        Primitive::Bool => format!("BVec{}", n),
                        Primitive::Int32 => format!("IVec{}", n),
                        Primitive::Uint32 => format!("UVec{}", n),
                        Primitive::Int64 => format!("LVec{}", n),
                        Primitive::Uint64 => format!("ULVec{}", n),
                        Primitive::Float32 => format!("Vec{}", n),
                        Primitive::Float64 => format!("DVec{}", n),
                    },
                    _ => todo!(),
                }
            }
            Type::Matrix(mt) => {
                let n = mt.dimension;
                match mt.element {
                    VectorElementType::Scalar(s) => match s {
                        Primitive::Float32 => format!("Mat{}", n),
                        // Primitive::Float64 => format!("DMat{}", n),
                        _ => unreachable!(),
                    },
                    _ => todo!(),
                }
            }
            _ => todo!("{:?}", t),
        }
    }
    fn to_rust_type(&mut self, t: Gc<Type>) -> String {
        if let Some(t) = self.cache.get(&t) {
            return t.clone();
        } else {
            let t_ = self.to_rust_type_(t);
            self.cache.insert(t, t_.clone());
            return t_;
        }
    }
}

pub struct CodeGen {
    type_gen: TypeGen,
    node_to_var: HashMap<NodeRef, String>,
    body: String,
    captures: HashMap<NodeRef, usize>,
    args: HashMap<NodeRef, usize>,
}
impl CodeGen {
    fn gen_node(&mut self, node: NodeRef) -> String {
        if let Some(var) = self.node_to_var.get(&node) {
            return var.clone();
        } else {
            let var = self.gen_node_(node);
            self.node_to_var.insert(node, var.clone());
            return var;
        }
    }
    fn gep_field_name(node: NodeRef, i: i32) -> String {
        let node_ty = node.type_();
        match node_ty.as_ref() {
            Type::Struct(_) => {
                format!("f{}", i)
            }
            Type::Vector(_) => match i {
                0 => "x".to_string(),
                1 => "y".to_string(),
                2 => "z".to_string(),
                3 => "w".to_string(),
                _ => unreachable!(),
            },
            Type::Matrix(_) => {
                format!("cols[{}]", i)
            }
            _ => todo!(),
        }
    }
    fn gen_node_(&mut self, node: NodeRef) -> String {
        let inst = node.get().instruction;
        let node_ty = node.type_();
        let node_ty_s = self.type_gen.to_rust_type(node_ty);
        let var = format!("v{}", self.node_to_var.len());
        self.node_to_var.insert(node, var.clone());

        match inst.as_ref() {
            ir::Instruction::Buffer => {
                if let Some(i) = self.captures.get(&node) {
                    writeln!(
                        &mut self.body,
                        "let {}: BufferView<_> = k_args.captures({}).as_buffer().unwrap();",
                        var, i
                    )
                    .unwrap();
                } else if let Some(i) = self.args.get(&node) {
                    writeln!(
                        &mut self.body,
                        "let {}: BufferView<_> = k_args.args({}).as_buffer().unwrap();",
                        var, i
                    )
                    .unwrap();
                } else {
                    panic!("unknown buffer");
                }
            }
            ir::Instruction::Bindless => todo!(),
            ir::Instruction::Texture2D => todo!(),
            ir::Instruction::Texture3D => todo!(),
            ir::Instruction::Accel => todo!(),
            ir::Instruction::Shared => todo!(),
            ir::Instruction::Uniform => todo!(),
            ir::Instruction::Local { init } => {
                let init_s = self.gen_node(*init);
                writeln!(
                    &mut self.body,
                    "let mut {}: {} = {};",
                    var, node_ty_s, init_s
                )
                .unwrap();
            }
            ir::Instruction::Argument { by_value } => todo!(),
            ir::Instruction::UserData(_) => todo!(),
            ir::Instruction::Invalid => todo!(),
            ir::Instruction::Const(c) => match c {
                ir::Const::Zero(_) => writeln!(
                    &mut self.body,
                    "let {}: {} = unsafe{{ std::mem::zeroed() }};",
                    var, node_ty_s
                )
                .unwrap(),
                ir::Const::One(_) => todo!(),
                ir::Const::Bool(v) => {
                    writeln!(&mut self.body, "let {}: {} = {};", var, node_ty_s, v).unwrap();
                }
                ir::Const::Int32(v) => {
                    writeln!(&mut self.body, "let {}: {} = {}i32;", var, node_ty_s, v).unwrap();
                }
                ir::Const::Uint32(v) => {
                    writeln!(&mut self.body, "let {}: {} = {}u32;", var, node_ty_s, v).unwrap();
                }
                ir::Const::Int64(v) => {
                    writeln!(&mut self.body, "let {}: {} = {}i64;", var, node_ty_s, v).unwrap();
                }
                ir::Const::Uint64(v) => {
                    writeln!(&mut self.body, "let {}: {} = {}u64;", var, node_ty_s, v).unwrap();
                }
                ir::Const::Float32(v) => {
                    writeln!(&mut self.body, "let {}: {} = {}f32;", var, node_ty_s, v).unwrap();
                }
                ir::Const::Float64(v) => {
                    writeln!(&mut self.body, "let {}: {} = {}f64;", var, node_ty_s, v).unwrap();
                }
                ir::Const::Generic(data, _) => {
                    let data = data
                        .as_ref()
                        .iter()
                        .map(|x| format!("{:?}u8", x))
                        .collect::<Vec<_>>()
                        .join(", ");
                    writeln!(
                        &mut self.body,
                        "let {0}: {1} = unsafe{{ const DATA:&'static [u8] = &[{2}];\
                        let ptr = DATA.as_ptr() as * const {1}; std::ptr::read(ptr) }};",
                        var, node_ty_s, data
                    )
                    .unwrap();
                }
            },
            ir::Instruction::Update { var, value } => {
                let var_s = self.gen_node(*var);
                let value_s = self.gen_node(*value);
                if var.is_local() {
                    writeln!(&mut self.body, "{} = {};", var_s, value_s).unwrap();
                } else {
                    // should be gep
                    writeln!(
                        &mut self.body,
                        "unsafe{{ *(&mut *{}) = {} }};",
                        var_s, value_s
                    )
                    .unwrap();
                }
            }
            ir::Instruction::Call(f, args) => {
                let args_v = args
                    .as_ref()
                    .iter()
                    .map(|arg| self.gen_node(*arg))
                    .collect::<Vec<_>>();
                match f {
                    ir::Func::ZeroInitializer => todo!(),
                    ir::Func::Assume => todo!(),
                    ir::Func::Unreachable => {
                        writeln!(&mut self.body, "panic(\"unreachable\");").unwrap();
                    }
                    ir::Func::Assert => {
                        writeln!(&mut self.body, "lc_assert({});", args_v[0]).unwrap()
                    }
                    ir::Func::ThreadId => {
                        writeln!(
                            &mut self.body,
                            "let {}: {} = UVec3::new(k_args.thread_id[0], k_args.thread_id[1], k_args.thread_id[2]);",
                            var, node_ty_s
                        ).unwrap();
                    }
                    ir::Func::BlockId => {
                        writeln!(
                            &mut self.body,
                            "let {}: {} = UVec3::new(k_args.block_id[0], k_args.block_id[1], k_args.block_id[2]);",
                            var, node_ty_s
                        ).unwrap();
                    }
                    ir::Func::DispatchId => {
                        writeln!(
                            &mut self.body,
                            "let {}: {} = UVec3::new(k_args.dispatch_id[0], k_args.dispatch_id[1], k_args.dispatch_id[2]);",
                            var, node_ty_s
                        ).unwrap();
                    }
                    ir::Func::DispatchSize => {
                        writeln!(
                            &mut self.body,
                            "let {}: {} = UVec3::new(k_args.dispatch_size[0], k_args.dispatch_size[1], k_args.dispatch_size[2]);",
                            var, node_ty_s
                        ).unwrap();
                    }
                    ir::Func::RequiresGradient => todo!(),
                    ir::Func::Gradient => todo!(),
                    ir::Func::GradientMarker => todo!(),
                    ir::Func::InstanceToWorldMatrix => todo!(),
                    ir::Func::TraceClosest => todo!(),
                    ir::Func::TraceAny => todo!(),
                    ir::Func::SetInstanceTransform => todo!(),
                    ir::Func::SetInstanceVisibility => todo!(),
                    ir::Func::Load => {
                        writeln!(
                            &mut self.body,
                            "let {}: {} = {};",
                            var, node_ty_s, args_v[0]
                        )
                        .unwrap();
                    }
                    ir::Func::Cast => todo!(),
                    ir::Func::Bitcast => todo!(),
                    ir::Func::Add => writeln!(
                        self.body,
                        "let {}: {} = {} + {};",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Sub => writeln!(
                        self.body,
                        "let {}: {} = {} - {};",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Mul => writeln!(
                        self.body,
                        "let {}: {} = {} * {};",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Div => writeln!(
                        self.body,
                        "let {}: {} = {} / {};",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Rem => writeln!(
                        self.body,
                        "let {}: {} = {} % {};",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::BitAnd => writeln!(
                        self.body,
                        "let {}: {} = {} & {};",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::BitOr => writeln!(
                        self.body,
                        "let {}: {} = {} | {};",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::BitXor => writeln!(
                        self.body,
                        "let {}: {} = {} ^ {};",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Shl => writeln!(
                        self.body,
                        "let {}: {} = {} << {};",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Shr => writeln!(
                        self.body,
                        "let {}: {} = {} >> {};",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::RotRight => writeln!(
                        self.body,
                        "let {}: {} = {}.rotate_right({});",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::RotLeft => writeln!(
                        self.body,
                        "let {}: {} = {}.rotate_left({});",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Eq => writeln!(
                        self.body,
                        "let {}: {} = {}.cmpeq({});",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Ne => writeln!(
                        self.body,
                        "let {}: {} = {}.cmpne({});",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Lt => writeln!(
                        self.body,
                        "let {}: {} = {}.cmplt({});",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Le => writeln!(
                        self.body,
                        "let {}: {} = {}.cmple({});",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Gt => writeln!(
                        self.body,
                        "let {}: {} = {}.cmpgt({});",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Ge => writeln!(
                        self.body,
                        "let {}: {} = {}.cmpge({});",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::OuterProduct => writeln!(
                        self.body,
                        "let {}: {} = {}.outer({});",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::MatCompMul => writeln!(
                        self.body,
                        "let {}: {} = {}.comp_mul({});",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Neg => {
                        writeln!(self.body, "let {}: {} = -{};", var, node_ty_s, args_v[0]).unwrap()
                    }
                    ir::Func::Not => {
                        writeln!(self.body, "let {}: {} = !{};", var, node_ty_s, args_v[0]).unwrap()
                    }
                    ir::Func::BitNot => {
                        writeln!(self.body, "let {}: {} = ~{};", var, node_ty_s, args_v[0]).unwrap()
                    }
                    ir::Func::All => todo!(),
                    ir::Func::Any => todo!(),
                    ir::Func::Select => todo!(),
                    ir::Func::Clamp => todo!(),
                    ir::Func::Lerp => todo!(),
                    ir::Func::Step => todo!(),
                    ir::Func::Abs => writeln!(
                        self.body,
                        "let {}: {} = {}.abs();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Min => writeln!(
                        self.body,
                        "let {}: {} = {}.min({});",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Max => writeln!(
                        self.body,
                        "let {}: {} = {}.max({});",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::ReduceSum => writeln!(
                        self.body,
                        "let {}: {} = {}.reduce_sum();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::ReduceProd => writeln!(
                        self.body,
                        "let {}: {} = {}.reduce_prod();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::ReduceMin => writeln!(
                        self.body,
                        "let {}: {} = {}.reduce_min();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::ReduceMax => writeln!(
                        self.body,
                        "let {}: {} = {}.reduce_max();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Clz => todo!(),
                    ir::Func::Ctz => todo!(),
                    ir::Func::PopCount => todo!(),
                    ir::Func::Reverse => todo!(),
                    ir::Func::IsInf => writeln!(
                        self.body,
                        "let {}: {} = {}.is_infinite();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::IsNan => writeln!(
                        self.body,
                        "let {}: {} = {}.is_nan();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Acos => writeln!(
                        self.body,
                        "let {}: {} = {}.acos();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Acosh => writeln!(
                        self.body,
                        "let {}: {} = {}.acosh();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Asin => writeln!(
                        self.body,
                        "let {}: {} = {}.asin();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Asinh => writeln!(
                        self.body,
                        "let {}: {} = {}.asinh();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Atan => writeln!(
                        self.body,
                        "let {}: {} = {}.atan();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Atan2 => writeln!(
                        self.body,
                        "let {}: {} = {}.atan2({});",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Atanh => writeln!(
                        self.body,
                        "let {}: {} = {}.atanh();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Cos => writeln!(
                        self.body,
                        "let {}: {} = {}.cos();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Cosh => writeln!(
                        self.body,
                        "let {}: {} = {}.cosh();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Sin => writeln!(
                        self.body,
                        "let {}: {} = {}.sin();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Sinh => writeln!(
                        self.body,
                        "let {}: {} = {}.sinh();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Tan => writeln!(
                        self.body,
                        "let {}: {} = {}.tan();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Tanh => writeln!(
                        self.body,
                        "let {}: {} = {}.tanh();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Exp => writeln!(
                        self.body,
                        "let {}: {} = {}.exp();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Exp2 => writeln!(
                        self.body,
                        "let {}: {} = {}.exp2();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Exp10 => writeln!(
                        self.body,
                        "const {} {} = math::exp10({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Log => writeln!(
                        self.body,
                        "let {}: {} = {}.ln();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Log2 => writeln!(
                        self.body,
                        "let {}: {} = {}.log2();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Log10 => writeln!(
                        self.body,
                        "let {}: {} = {}.log10();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Powi => writeln!(
                        self.body,
                        "let {}: {} = {}.powi({});",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Powf => writeln!(
                        self.body,
                        "let {}: {} = {}.powf({});",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Sqrt => writeln!(
                        self.body,
                        "let {}: {} = {}.sqrt();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Rsqrt => writeln!(
                        self.body,
                        "let {}: {} = {}.rsqrt();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Ceil => writeln!(
                        self.body,
                        "let {}: {} = {}.ceil();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Floor => writeln!(
                        self.body,
                        "let {}: {} = {}.floor();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Fract => writeln!(
                        self.body,
                        "let {}: {} = {}.fract();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Trunc => writeln!(
                        self.body,
                        "let {}: {} = {}.trunc();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Round => writeln!(
                        self.body,
                        "let {}: {} = {}.round();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Fma => writeln!(
                        self.body,
                        "let {}: {} = {}.mul_add({}, {});",
                        var, node_ty_s, args_v[0], args_v[1], args_v[2]
                    )
                    .unwrap(),
                    ir::Func::Copysign => writeln!(
                        self.body,
                        "let {}: {} = {}.copysign({});",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Cross => writeln!(
                        self.body,
                        "let {}: {} = {}.cross({});",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Dot => writeln!(
                        self.body,
                        "let {}: {} = {}.dot({});",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Length => writeln!(
                        self.body,
                        "let {}: {} = {}.length();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::LengthSquared => writeln!(
                        self.body,
                        "let {}: {} = {}.length_squared();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Normalize => writeln!(
                        self.body,
                        "let {}: {} = {}.normalize();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Faceforward => writeln!(
                        self.body,
                        "let {}: {} = {}.faceforward({}, {});",
                        var, node_ty_s, args_v[0], args_v[1], args_v[2]
                    )
                    .unwrap(),
                    ir::Func::Determinant => writeln!(
                        self.body,
                        "let {}: {} = {}.determinant();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Transpose => writeln!(
                        self.body,
                        "let {}: {} = {}.transpose();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Inverse => writeln!(
                        self.body,
                        "let {}: {} = {}.inverse();",
                        var, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::SynchronizeBlock => todo!(),
                    ir::Func::AtomicExchange => todo!(),
                    ir::Func::AtomicCompareExchange => todo!(),
                    ir::Func::AtomicFetchAdd => todo!(),
                    ir::Func::AtomicFetchSub => todo!(),
                    ir::Func::AtomicFetchAnd => todo!(),
                    ir::Func::AtomicFetchOr => todo!(),
                    ir::Func::AtomicFetchXor => todo!(),
                    ir::Func::AtomicFetchMin => todo!(),
                    ir::Func::AtomicFetchMax => todo!(),
                    ir::Func::BufferRead => writeln!(
                        self.body,
                        "let {}: {} = {}.read({} as usize);",
                        var, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::BufferWrite => writeln!(
                        self.body,
                        "{}.write({} as usize, {});",
                        args_v[0], args_v[1], args_v[2]
                    )
                    .unwrap(),
                    ir::Func::BufferSize => todo!(),
                    ir::Func::TextureRead => todo!(),
                    ir::Func::TextureWrite => todo!(),
                    ir::Func::BindlessTexture2dSample => todo!(),
                    ir::Func::BindlessTexture2dSampleLevel => todo!(),
                    ir::Func::BindlessTexture2dSampleGrad => todo!(),
                    ir::Func::BindlessTexture3dSample => todo!(),
                    ir::Func::BindlessTexture3dSampleLevel => todo!(),
                    ir::Func::BindlessTexture3dSampleGrad => todo!(),
                    ir::Func::BindlessTexture2dRead => todo!(),
                    ir::Func::BindlessTexture3dRead => todo!(),
                    ir::Func::BindlessTexture2dReadLevel => todo!(),
                    ir::Func::BindlessTexture3dReadLevel => todo!(),
                    ir::Func::BindlessTexture2dSize => todo!(),
                    ir::Func::BindlessTexture3dSize => todo!(),
                    ir::Func::BindlessTexture2dSizeLevel => todo!(),
                    ir::Func::BindlessTexture3dSizeLevel => todo!(),
                    ir::Func::BindlessBufferRead => todo!(),
                    ir::Func::BindlessBufferSize => todo!(),
                    ir::Func::Vec => writeln!(
                        self.body,
                        "let {}: {} = {}::splat({});",
                        var, node_ty_s, node_ty_s, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Vec2 => writeln!(
                        self.body,
                        "let {}: {} = {}::new({}, {});",
                        var, node_ty_s, node_ty_s, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Vec3 => writeln!(
                        self.body,
                        "let {}: {} = {}::new({}, {}, {});",
                        var, node_ty_s, node_ty_s, args_v[0], args_v[1], args_v[2]
                    )
                    .unwrap(),
                    ir::Func::Vec4 => writeln!(
                        self.body,
                        "let {}: {} = {}::new({}, {}, {}, {});",
                        var, node_ty_s, node_ty_s, args_v[0], args_v[1], args_v[2], args_v[3]
                    )
                    .unwrap(),
                    ir::Func::Permute => todo!(),
                    ir::Func::ExtractElement => {
                        let i = args.as_ref()[1].get_i32();
                        let field_name = Self::gep_field_name(args.as_ref()[0], i);
                        writeln!(
                            self.body,
                            "let {}: {} = {}.{};",
                            var, node_ty_s, args_v[0], field_name
                        )
                        .unwrap();
                    }
                    ir::Func::InsertElement => {
                        let i = args.as_ref()[2].get_i32();
                        let field_name = Self::gep_field_name(args.as_ref()[0], i);
                        writeln!(self.body, "{}.{};", args_v[0], field_name).unwrap();
                    }
                    ir::Func::GetElementPtr => {
                        let i = args.as_ref()[1].get_i32();
                        let field_name = Self::gep_field_name(args.as_ref()[0], i);
                        writeln!(
                            self.body,
                            "let {}: {}* = &mut {}.{} as * mut _;",
                            var, node_ty_s, args_v[0], field_name
                        )
                        .unwrap();
                    }
                    ir::Func::Struct => {
                        let mut fields = String::new();
                        for (i, arg) in args_v.iter().enumerate() {
                            let field_name = Self::gep_field_name(node, i as i32);
                            fields.push_str(&format!("{}: {}, ", field_name, arg));
                        }
                        writeln!(
                            self.body,
                            "let {}: {} = {} {{ {} }};",
                            var, node_ty_s, node_ty_s, fields
                        )
                        .unwrap();
                    }
                    ir::Func::Mat => todo!(),
                    ir::Func::Matrix2 => todo!(),
                    ir::Func::Matrix3 => todo!(),
                    ir::Func::Matrix4 => todo!(),
                    ir::Func::Callable(_) => todo!(),
                    ir::Func::CpuCustomOp(_) => todo!(),
                }
            }
            ir::Instruction::Phi(_) => todo!(),
            ir::Instruction::Return(v) => {
                let v = self.gen_node(*v);
                writeln!(self.body, "return {};", v).unwrap();
            }
            ir::Instruction::Loop { body, cond } => {
                writeln!(self.body, "loop {{").unwrap();
                self.gen_block(*body);
                let cond = self.gen_node(*cond);
                writeln!(self.body, "if (!{}) break;", cond).unwrap();
                writeln!(self.body, "}}").unwrap();
            }
            ir::Instruction::GenericLoop {
                prepare,
                cond,
                body,
                update,
            } => {
                self.gen_block(*prepare);
                let cond = self.gen_node(*cond);
                writeln!(self.body, "if(!{}) break;", cond).unwrap();
                self.gen_block(*body);
                self.gen_block(*update);
            }
            ir::Instruction::Break => writeln!(self.body, "break;").unwrap(),
            ir::Instruction::Continue => writeln!(self.body, "continue;").unwrap(),
            ir::Instruction::If {
                cond,
                true_branch,
                false_branch,
            } => {
                let cond = self.gen_node(*cond);
                writeln!(self.body, "if ({}) {{", cond).unwrap();
                self.gen_block(*true_branch);
                writeln!(self.body, "}} else {{").unwrap();
                self.gen_block(*false_branch);
                writeln!(self.body, "}}").unwrap();
            }
            ir::Instruction::Switch {
                value,
                default,
                cases,
            } => {
                let value = self.gen_node(*value);
                writeln!(self.body, "match {} {{", value).unwrap();
                for SwitchCase { value, block } in cases.as_ref() {
                    let value = self.gen_node(*value);
                    writeln!(self.body, "{} => {{", value).unwrap();
                    self.gen_block(*block);
                    writeln!(self.body, "}}").unwrap();
                }
                writeln!(self.body, "_ => {{").unwrap();
                self.gen_block(*default);
                writeln!(self.body, "}}").unwrap();
                writeln!(self.body, "}}").unwrap();
            }
            ir::Instruction::Comment(comment) => {
                let comment = CString::new(comment.as_ref()).unwrap();
                writeln!(&mut self.body, "/* {} */", comment.to_string_lossy()).unwrap();
            }
            ir::Instruction::Debug(_) => {}
        }
        var
    }
    fn gen_block(&mut self, block: Gc<ir::BasicBlock>) {
        writeln!(&mut self.body, "{{").unwrap();
        for n in block.nodes() {
            self.gen_node(n);
        }
        writeln!(&mut self.body, "}}").unwrap();
    }
    pub fn run(kernel: &ir::KernelModule) -> String {
        let mut gen = Self::new();
        for (i, capture) in kernel.captures.as_ref().iter().enumerate() {
            gen.captures.insert(capture.node, i);
        }
        for (i, arg) in kernel.args.as_ref().iter().enumerate() {
            gen.args.insert(*arg, i);
        }
        gen.gen_block(kernel.module.entry);
        let prelude = r#"#[no_mangle] pub extern "C" fn kernel_fn(k_args:&KernelFnArgs) {"#;
        format!(
            "#![allow(unused_variables)]{}{}\n{}\n{}\n{} }}",
            SHADER_LIB_SRC, MATH_LIB_SRC, gen.type_gen.struct_typedefs, prelude, gen.body
        )
    }
    fn new() -> Self {
        Self {
            body: String::new(),
            type_gen: TypeGen::new(),
            node_to_var: HashMap::new(),
            captures: HashMap::new(),
            args: HashMap::new(),
        }
    }
}
