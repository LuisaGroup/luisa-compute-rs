use ir::Type;
use luisa_compute_ir::ir;
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
                let hash = sha256(&field_types_str);
                let name = format!("s_{}", hash);

                self.cache.insert(t, name.clone());
                let mut tmp = String::new();
                writeln!(tmp, "struct {} {{", name).unwrap();
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
    
}