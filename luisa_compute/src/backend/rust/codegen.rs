use ir::Type;
use luisa_compute_ir::ir::{self, NodeRef};
use luisa_compute_ir::Gc;
use std::collections::HashMap;
use std::ffi::CString;
use std::fmt::Write;

use super::sha256;
pub(crate) struct TypeGen {
    cache: HashMap<Gc<Type>, String>,
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
    fn to_c_type_(&mut self, t: Gc<Type>) -> String {
        match t.as_ref() {
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
                    .map(|f| self.to_c_type(*f))
                    .collect();
                let field_types_str = field_types.join(", ");
                let hash = sha256(&format!("{}_alignas({})", field_types_str, st.alignment));
                let name = format!("s_{}", hash);

                self.cache.insert(t, name.clone());
                let mut tmp = String::new();
                writeln!(tmp, "struct alignas({}) {} {{", st.alignment, name).unwrap();
                for (i, field) in st.fields.as_ref().iter().enumerate() {
                    let field_name = format!("f{}", i);
                    let field_type = self.to_c_type(*field);
                    writeln!(tmp, "    {} {};", field_type, field_name).unwrap();
                }
                writeln!(tmp, "}};").unwrap();
                self.struct_typedefs.push_str(&tmp);
                name
            }
            _ => todo!("{:?}", t),
        }
    }
    fn to_c_type(&mut self, t: Gc<Type>) -> String {
        if let Some(t) = self.cache.get(&t) {
            return t.clone();
        } else {
            let t_ = self.to_c_type_(t);
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
        let node_ty = node.type_();
        let node_ty_s = self.type_gen.to_c_type(node_ty);
        let var = format!("v{}", self.node_to_var.len());
        self.node_to_var.insert(node, var.clone());

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
                    ir::Func::Add => writeln!(
                        self.body,
                        "{} {} = {} + {};",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Sub => writeln!(
                        self.body,
                        "{} {} = {} - {};",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Mul => writeln!(
                        self.body,
                        "{} {} = {} * {};",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Div => writeln!(
                        self.body,
                        "{} {} = {} / {};",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Rem => writeln!(
                        self.body,
                        "{} {} = rem({}, {});",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::BitAnd => writeln!(
                        self.body,
                        "{} {} = {} & {};",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::BitOr => writeln!(
                        self.body,
                        "{} {} = {} | {};",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::BitXor => writeln!(
                        self.body,
                        "{} {} = {} ^ {};",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Shl => writeln!(
                        self.body,
                        "{} {} = {} << {};",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Shr => writeln!(
                        self.body,
                        "{} {} = {} >> {};",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::RotRight => writeln!(
                        self.body,
                        "{} {} = rot_right({}, {});",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::RotLeft => writeln!(
                        self.body,
                        "{} {} = rot_left({}, {});",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Eq => writeln!(
                        self.body,
                        "{} {} = {} == {};",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Ne => writeln!(
                        self.body,
                        "{} {} = {} != {};",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Lt => writeln!(
                        self.body,
                        "{} {} = {} < {};",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Le => writeln!(
                        self.body,
                        "{} {} = {} <= {};",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Gt => writeln!(
                        self.body,
                        "{} {} = {} > {};",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Ge => writeln!(
                        self.body,
                        "{} {} = {} >= {};",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::MatCompMul => todo!(),
                    ir::Func::MatCompDiv => todo!(),
                    ir::Func::Neg => {
                        writeln!(self.body, "{} {} = -{};", node_ty_s, var, args_v[0]).unwrap()
                    }
                    ir::Func::Not => {
                        writeln!(self.body, "{} {} = !{};", node_ty_s, var, args_v[0]).unwrap()
                    }
                    ir::Func::BitNot => {
                        writeln!(self.body, "{} {} = ~{};", node_ty_s, var, args_v[0]).unwrap()
                    }
                    ir::Func::All => todo!(),
                    ir::Func::Any => todo!(),
                    ir::Func::Select => todo!(),
                    ir::Func::Clamp => todo!(),
                    ir::Func::Lerp => todo!(),
                    ir::Func::Step => todo!(),
                    ir::Func::Abs => writeln!(
                        self.body,
                        "const {} {} = math::abs({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Min => writeln!(
                        self.body,
                        "const {} {} = math::min({}, {});",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Max => writeln!(
                        self.body,
                        "const {} {} = math::max({}, {});",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::ReduceSum => todo!(),
                    ir::Func::ReduceProd => todo!(),
                    ir::Func::ReduceMin => todo!(),
                    ir::Func::ReduceMax => todo!(),
                    ir::Func::Clz => todo!(),
                    ir::Func::Ctz => todo!(),
                    ir::Func::PopCount => todo!(),
                    ir::Func::Reverse => todo!(),
                    ir::Func::IsInf => writeln!(
                        self.body,
                        "const {} {} = math::isinf({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::IsNan => writeln!(
                        self.body,
                        "const {} {} = math::isnan({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Acos => writeln!(
                        self.body,
                        "const {} {} = math::acos({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Acosh => writeln!(
                        self.body,
                        "const {} {} = math::acosh({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Asin => writeln!(
                        self.body,
                        "const {} {} = math::asin({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Asinh => writeln!(
                        self.body,
                        "const {} {} = math::asinh({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Atan => writeln!(
                        self.body,
                        "const {} {} = math::atan({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Atan2 => writeln!(
                        self.body,
                        "const {} {} = math::atan2({}, {});",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Atanh => writeln!(
                        self.body,
                        "const {} {} = math::atanh({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Cos => writeln!(
                        self.body,
                        "const {} {} = math::cos({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Cosh => writeln!(
                        self.body,
                        "const {} {} = math::cosh({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Sin => writeln!(
                        self.body,
                        "const {} {} = math::sin({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Sinh => writeln!(
                        self.body,
                        "const {} {} = math::sinh({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Tan => writeln!(
                        self.body,
                        "const {} {} = math::tan({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Tanh => writeln!(
                        self.body,
                        "const {} {} = math::tanh({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Exp => writeln!(
                        self.body,
                        "const {} {} = math::exp({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Exp2 => writeln!(
                        self.body,
                        "const {} {} = math::exp2({});",
                        node_ty_s, var, args_v[0]
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
                        "const {} {} = math::log({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Log2 => writeln!(
                        self.body,
                        "const {} {} = math::log2({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Log10 => writeln!(
                        self.body,
                        "const {} {} = math::log10({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Powi => writeln!(
                        self.body,
                        "const {} {} = math::powi({}, {});",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Powf => writeln!(
                        self.body,
                        "const {} {} = math::powf({}, {});",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
                    ir::Func::Sqrt => writeln!(
                        self.body,
                        "const {} {} = math::sqrt({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Rsqrt => writeln!(
                        self.body,
                        "const {} {} = math::rsqrt({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Ceil => writeln!(
                        self.body,
                        "const {} {} = math::ceil({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Floor => writeln!(
                        self.body,
                        "const {} {} = math::floor({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Fract => writeln!(
                        self.body,
                        "const {} {} = math::fract({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Trunc => writeln!(
                        self.body,
                        "const {} {} = math::trunc({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Round => writeln!(
                        self.body,
                        "const {} {} = math::round({});",
                        node_ty_s, var, args_v[0]
                    )
                    .unwrap(),
                    ir::Func::Fma => writeln!(
                        self.body,
                        "const {} {} = math::fma({}, {}, {});",
                        node_ty_s, var, args_v[0], args_v[1], args_v[2]
                    )
                    .unwrap(),
                    ir::Func::Copysign => writeln!(
                        self.body,
                        "const {} {} = math::copysign({}, {});",
                        node_ty_s, var, args_v[0], args_v[1]
                    )
                    .unwrap(),
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
                    ir::Func::ExtractElement => {
                        let i = args.as_ref()[1].get_i32();
                        writeln!(
                            self.body,
                            "const {} {} = {}.f{};",
                            node_ty_s, var, args_v[0], i
                        )
                        .unwrap();
                    }
                    ir::Func::InsertElement => {
                        let i = args.as_ref()[2].get_i32();
                        writeln!(
                            self.body,
                            "{}.f{};",
                            args_v[0], i
                        )
                        .unwrap();
                    },
                    ir::Func::GetElementPtr => {
                        let i = args.as_ref()[1].get_i32();
                        writeln!(self.body, "{}* {} = &{}.f{};", node_ty_s, var, args_v[0], i)
                            .unwrap();
                    }
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
            ir::Instruction::Comment(comment) => {
                let comment = CString::new(comment.as_ref()).unwrap();
                writeln!(&mut self.body, "/* {} */", comment.to_string_lossy()).unwrap();
            }
            ir::Instruction::Debug(_) => {}
        }
        todo!()
    }
}
