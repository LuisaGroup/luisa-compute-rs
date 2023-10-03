use luisa_compute_ir::ir::Type;
use luisa_compute_ir::CArc;

use crate::prelude::*;
/** A buffer with SOA layout.
 */
pub struct SoaBuffer<T: Value> {
    storage: ByteBuffer,
    view: SoaBufferView<'static, T>,
    _marker: std::marker::PhantomData<T>,
}

pub struct SoaBufferView<'a, T: Value> {
    inner: ByteBufferView<'a>,
    _marker: std::marker::PhantomData<T>,
}

fn compute_number_of_32bits_buffers(ty: &Type) -> usize {
    match ty {
        Type::Void => unreachable!(),
        Type::UserData => unreachable!(),
        Type::Primitive(_) => todo!(),
        Type::Vector(_) => todo!(),
        Type::Matrix(_) => todo!(),
        Type::Struct(_) => todo!(),
        Type::Array(_) => todo!(),
        Type::Opaque(_) => todo!(),
    }
}

// impl<T> IndexRead for SoaBuffer<T>
// where
//     T: Value,
// {
//     type Element = T;
//     fn read<I: super::index::IntoIndex>(&self, i: I) -> Expr<Self::Element> {
//         let i = i.to_u64();
//         todo!()
//     }
// }
