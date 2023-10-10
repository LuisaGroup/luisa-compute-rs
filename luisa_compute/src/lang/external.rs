use std::sync::Arc;

use luisa_compute_ir::ir::CpuCustomOp;

use crate::internal_prelude::*;
pub struct CpuFn<T: Value> {
    op: CArc<CpuCustomOp>,
    _marker: PhantomData<T>,
}

/*
Interestingly, Box::into_raw(Box<Closure>) does not give a valid pointer.
*/
struct ClosureContainer<T> {
    f: Arc<dyn Fn(&mut T) + 'static + Send + Sync>,
}

/// A custom function that can be called inside a cpu kernel
impl<T: Value> CpuFn<T> {
    pub fn new<F: Fn(&mut T) + 'static + Send + Sync>(f: F) -> Self {
        let f_ptr = Box::into_raw(Box::new(ClosureContainer::<T> { f: Arc::new(f) }));
        let op = CpuCustomOp {
            data: f_ptr as *mut u8,
            func: _trampoline::<T, F>,
            destructor: _drop::<F>,
            arg_type: T::type_(),
        };
        Self {
            op: CArc::new(op),
            _marker: PhantomData,
        }
    }
    pub fn call(&self, arg: impl AsExpr<Value = T>) -> Expr<T> {
        with_recorder(|r| {
            assert_eq!(
                r.device
                    .as_ref()
                    .unwrap()
                    .upgrade()
                    .unwrap()
                    .inner
                    .query("device_name")
                    .unwrap(),
                "cpu",
                "CpuFn can only be used in cpu backend"
            );
            let addr = CArc::as_ptr(&self.op) as u64;
            if let Some((_, op)) = r.cpu_custom_ops.get(&addr) {
                assert_eq!(CArc::as_ptr(op), CArc::as_ptr(&self.op));
            } else {
                let i = r.cpu_custom_ops.len();
                r.cpu_custom_ops.insert(addr, (i, self.op.clone()));
            }
        });
        let arg = arg.as_expr().node().get();
        Expr::<T>::from_node(__current_scope(|b| {
            b.call(
                Func::CpuCustomOp(self.op.clone()),
                &[arg],
                T::type_(),
            )
        }).into())
    }
}

extern "C" fn _trampoline<T, F: FnMut(&mut T)>(data: *mut u8, args: *mut u8) {
    unsafe {
        let container = &*(data as *const ClosureContainer<T>);
        let f = &container.f;
        let args = &mut *(args as *mut T);
        f(args);
    }
}

extern "C" fn _drop<T>(data: *mut u8) {
    unsafe {
        let _ = Box::from_raw(data as *mut T);
    }
}
