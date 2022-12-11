use ir::Type;
use luisa_compute_ir::ir::{self, NodeRef};
use std::collections::HashMap;
use std::fmt::Write;

use super::sha256;
pub(crate) struct TypeGen {
    cache: HashMap<&'static Type, String>,
    struct_typedefs: String,
}
pub(crate) type PushConstants = Vec<u8>;
impl TypeGen {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
            struct_typedefs: String::new(),
        }
    }
    fn to_c_type_(&mut self, t: &'static Type) -> String {
        match t {
            Type::Primitive(t) => match t {
                ir::Primitive::Bool => "bool".to_string(),
                ir::Primitive::Int32 => "int32_t".to_string(),
                ir::Primitive::Uint32 => "uint32_t".to_string(),
                ir::Primitive::Int64 => "int64_t".to_string(),
                ir::Primitive::Uint64 => "uint64_t".to_string(),
                ir::Primitive::Float32 => "float".to_string(),
                ir::Primitive::Float64 => "double".to_string(),
                // crate::ir::Primitive::USize => format!("i{}", std::mem::size_of::<usize>() * 8),
            },
            Type::Void => "void".to_string(),
            Type::Struct(st) => {
                let field_types: Vec<String> = st
                    .fields
                    .as_ref()
                    .iter()
                    .map(|f| self.to_c_type(f))
                    .collect();
                let field_types_str = field_types.join(", ");
                let hash = sha256(&format!("{}_alignas({})", field_types_str, st.alignment));
                let name = format!("s_{}", hash);

                self.cache.insert(t, name.clone());
                let mut tmp = String::new();
                writeln!(tmp, "struct alignas({}) {} {{", st.alignment, name).unwrap();
                for (i, field) in st.fields.as_ref().iter().enumerate() {
                    let field_name = format!("field{}", i);
                    let field_type = self.to_c_type(field);
                    writeln!(tmp, "    {} {};", field_type, field_name).unwrap();
                }
                writeln!(tmp, "}};").unwrap();
                self.struct_typedefs.push_str(&tmp);
                name
            }
            _ => todo!("{:?}", t),
        }
    }
    fn to_c_type(&mut self, t: &'static Type) -> String {
        if let Some(t) = self.cache.get(t) {
            return t.clone();
        } else {
            let t_ = self.to_c_type_(&t);
            self.cache.insert(t, t_.clone());
            return t_;
        }
    }
}

struct CodeGen {
    type_gen: TypeGen,
    node_to_var: HashMap<NodeRef, String>,
    body: String,
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
    fn gen_node_(&mut self, node: NodeRef) -> String {
        let inst = node.get().instruction;
        match inst.as_ref() {
            ir::Instruction::Buffer => todo!(),
            ir::Instruction::Bindless => todo!(),
            ir::Instruction::Texture2D => todo!(),
            ir::Instruction::Texture3D => todo!(),
            ir::Instruction::Accel => todo!(),
            ir::Instruction::Shared => todo!(),
            ir::Instruction::Uniform => todo!(),
            ir::Instruction::Local { init } => todo!(),
            ir::Instruction::Argument { by_value } => todo!(),
            ir::Instruction::UserData(_) => todo!(),
            ir::Instruction::Invalid => todo!(),
            ir::Instruction::Const(_) => todo!(),
            ir::Instruction::Update { var, value } => todo!(),
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
                    ir::Func::ThreadId => todo!(),
                    ir::Func::BlockId => todo!(),
                    ir::Func::DispatchId => todo!(),
                    ir::Func::DispatchSize => todo!(),
                    ir::Func::RequiresGradient => todo!(),
                    ir::Func::Gradient => todo!(),
                    ir::Func::GradientMarker => todo!(),
                    ir::Func::InstanceToWorldMatrix => todo!(),
                    ir::Func::TraceClosest => todo!(),
                    ir::Func::TraceAny => todo!(),
                    ir::Func::SetInstanceTransform => todo!(),
                    ir::Func::SetInstanceVisibility => todo!(),
                    ir::Func::Load => todo!(),
                    ir::Func::Cast => todo!(),
                    ir::Func::Bitcast => todo!(),
                    ir::Func::Add => todo!(),
                    ir::Func::Sub => todo!(),
                    ir::Func::Mul => todo!(),
                    ir::Func::Div => todo!(),
                    ir::Func::Rem => todo!(),
                    ir::Func::BitAnd => todo!(),
                    ir::Func::BitOr => todo!(),
                    ir::Func::BitXor => todo!(),
                    ir::Func::Shl => todo!(),
                    ir::Func::Shr => todo!(),
                    ir::Func::RotRight => todo!(),
                    ir::Func::RotLeft => todo!(),
                    ir::Func::Eq => todo!(),
                    ir::Func::Ne => todo!(),
                    ir::Func::Lt => todo!(),
                    ir::Func::Le => todo!(),
                    ir::Func::Gt => todo!(),
                    ir::Func::Ge => todo!(),
                    ir::Func::MatCompMul => todo!(),
                    ir::Func::MatCompDiv => todo!(),
                    ir::Func::Neg => todo!(),
                    ir::Func::Not => todo!(),
                    ir::Func::BitNot => todo!(),
                    ir::Func::All => todo!(),
                    ir::Func::Any => todo!(),
                    ir::Func::Select => todo!(),
                    ir::Func::Clamp => todo!(),
                    ir::Func::Lerp => todo!(),
                    ir::Func::Step => todo!(),
                    ir::Func::Abs => todo!(),
                    ir::Func::Min => todo!(),
                    ir::Func::Max => todo!(),
                    ir::Func::ReduceSum => todo!(),
                    ir::Func::ReduceProd => todo!(),
                    ir::Func::ReduceMin => todo!(),
                    ir::Func::ReduceMax => todo!(),
                    ir::Func::Clz => todo!(),
                    ir::Func::Ctz => todo!(),
                    ir::Func::PopCount => todo!(),
                    ir::Func::Reverse => todo!(),
                    ir::Func::IsInf => todo!(),
                    ir::Func::IsNan => todo!(),
                    ir::Func::Acos => todo!(),
                    ir::Func::Acosh => todo!(),
                    ir::Func::Asin => todo!(),
                    ir::Func::Asinh => todo!(),
                    ir::Func::Atan => todo!(),
                    ir::Func::Atan2 => todo!(),
                    ir::Func::Atanh => todo!(),
                    ir::Func::Cos => todo!(),
                    ir::Func::Cosh => todo!(),
                    ir::Func::Sin => todo!(),
                    ir::Func::Sinh => todo!(),
                    ir::Func::Tan => todo!(),
                    ir::Func::Tanh => todo!(),
                    ir::Func::Exp => todo!(),
                    ir::Func::Exp2 => todo!(),
                    ir::Func::Exp10 => todo!(),
                    ir::Func::Log => todo!(),
                    ir::Func::Log2 => todo!(),
                    ir::Func::Log10 => todo!(),
                    ir::Func::Powi => todo!(),
                    ir::Func::Powf => todo!(),
                    ir::Func::Sqrt => todo!(),
                    ir::Func::Rsqrt => todo!(),
                    ir::Func::Ceil => todo!(),
                    ir::Func::Floor => todo!(),
                    ir::Func::Fract => todo!(),
                    ir::Func::Trunc => todo!(),
                    ir::Func::Round => todo!(),
                    ir::Func::Fma => todo!(),
                    ir::Func::Copysign => todo!(),
                    ir::Func::Cross => todo!(),
                    ir::Func::Dot => todo!(),
                    ir::Func::Length => todo!(),
                    ir::Func::LengthSquared => todo!(),
                    ir::Func::Normalize => todo!(),
                    ir::Func::Faceforward => todo!(),
                    ir::Func::Determinant => todo!(),
                    ir::Func::Transpose => todo!(),
                    ir::Func::Inverse => todo!(),
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
                    ir::Func::BufferRead => todo!(),
                    ir::Func::BufferWrite => todo!(),
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
                    ir::Func::Vec => todo!(),
                    ir::Func::Vec2 => todo!(),
                    ir::Func::Vec3 => todo!(),
                    ir::Func::Vec4 => todo!(),
                    ir::Func::Permute => todo!(),
                    ir::Func::ExtractElement => todo!(),
                    ir::Func::InsertElement => todo!(),
                    ir::Func::GetElementPtr => todo!(),
                    ir::Func::Struct => todo!(),
                    ir::Func::Mat => todo!(),
                    ir::Func::Matrix2 => todo!(),
                    ir::Func::Matrix3 => todo!(),
                    ir::Func::Matrix4 => todo!(),
                    ir::Func::Callable(_) => todo!(),
                    ir::Func::CpuCustomOp(_) => todo!(),
                }
            }
            ir::Instruction::Phi(_) => todo!(),
            ir::Instruction::Return(_) => todo!(),
            ir::Instruction::Loop { body, cond } => todo!(),
            ir::Instruction::GenericLoop {
                prepare,
                cond,
                body,
                update,
            } => todo!(),
            ir::Instruction::Break => todo!(),
            ir::Instruction::Continue => todo!(),
            ir::Instruction::If {
                cond,
                true_branch,
                false_branch,
            } => todo!(),
            ir::Instruction::Switch {
                value,
                default,
                cases,
            } => todo!(),
            ir::Instruction::Comment(_) => todo!(),
            ir::Instruction::Debug(_) => todo!(),
        }
        todo!()
    }
}
