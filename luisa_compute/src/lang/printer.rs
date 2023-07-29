use crate::prelude::*;
use crate::*;
use std::cell::RefCell;
use std::fmt::Debug;

struct PrinterItem {
    level: log::Level,
    log_fn: Vec<Box<dyn Fn(usize, &[u32]) -> Box<dyn Any>>>,
    count: usize,
    count_per_arg: Vec<usize>,
}
pub struct Printer {
    inner: Arc<PrinterData>,
}
struct PrinterData {
    items: RefCell<Vec<PrinterItem>>,
    count: Buffer<u32>,
    data: Buffer<u32>,
}
pub struct PrinterArgs {
    pack_fn: Vec<Box<dyn FnOnce(Expr<u32>, &BufferVar<u32>)>>,
    log_fn: Vec<Box<dyn Fn(usize, &[u32]) -> Box<dyn Any>>>,
    count_per_arg: Vec<usize>,
    count: usize,
}
impl PrinterArgs {
    pub fn append<E: ExprProxy + 'static>(&mut self, v: E)
    where
        E::Value: Debug,
    {
        let n = packed_size::<E::Value>();
        self.count_per_arg.push(n);
        let count = self.count;
        self.pack_fn.push(Box::new(move |offset, data| {
            pack_to(v, data, offset + count as u32);
        }));
        self.log_fn.push(Box::new(move |offset, data| unsafe {
            let data: E::Value = *(data.as_ptr().add(offset) as *const E::Value);
            Box::new(data)
        }));
        self.count += n;
    }
    pub fn new() -> Self {
        Self {
            pack_fn: vec![],
            count_per_arg: vec![],
            log_fn: vec![],
            count: 0,
        }
    }
}
#[macro_export]
macro_rules! info {
    () => {};
}
impl Printer {
    pub fn _log(&self, level: log::Level, args: PrinterArgs) {
        let inner = &self.inner;
        let offset = inner.count.var().atomic_fetch_add(0, 1 + args.count as u32);
        let mut items = inner.items.borrow_mut();
        let item_id = items.len() as u32;

        let data = inner.data.var();
        data.write(offset, item_id);
        let mut cnt = 0;
        for (i, pack_fn) in args.pack_fn.into_iter().enumerate() {
            pack_fn(offset + 1 + cnt, &data);
            cnt += args.count_per_arg[i] as u32;
        }
        items.push(PrinterItem {
            level,
            log_fn: args.log_fn,
            count: args.count,
            count_per_arg: args.count_per_arg,
        });
    }
    // pub fn print(&self) -> PrinterPrint {
    //     PrinterPrint {
    //         inner: self,
    //     }
    // }
}
// pub struct PrinterPrint<'a> {
//     inner: &'a Printer,
// }
// impl<'a> Scope<'a> {
//     pub fn print(&self, printer: &Printer) -> &Self {
//         let data = printer.inner.clone();
//         self.submit_with_callback([], move ||{

//         })
//     }
// }
