use parking_lot::RwLock;
use std::fmt::Debug;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

#[doc(hidden)]
pub use log as _log;

use crate::internal_prelude::*;

use crate::lang::types::TypeTag;
use crate::lang::{pack_to, packed_size};

pub use crate::{lc_debug, lc_error, lc_info, lc_warn};

pub type LogFn = Box<dyn Fn(&[*const u32]) + Send + Sync>;
struct PrinterItem {
    log_fn: LogFn,
    count: usize,
    count_per_arg: Vec<usize>,
}
pub struct Printer {
    inner: Arc<PrinterData>,
}
struct PrinterData {
    items: RwLock<Vec<PrinterItem>>,
    data: Buffer<u32>,
    host_data: Vec<u32>,
    dirty: AtomicBool,
}
pub struct PrinterArgs {
    pack_fn: Vec<Box<dyn Fn(Expr<u32>, &BufferVar<u32>)>>,
    count_per_arg: Vec<usize>,
    count: usize,
}
impl PrinterArgs {
    pub fn append<V: Value + Debug>(&mut self, v: Expr<V>) {
        let n = packed_size::<V>();
        self.count_per_arg.push(n);
        self.pack_fn.push(Box::new(move |offset, data| {
            let v = (&v).clone();
            pack_to(v, data, offset);
        }));
        self.count += n;
    }
    pub fn new() -> Self {
        Self {
            pack_fn: vec![],
            count_per_arg: vec![],
            count: 0,
        }
    }
}
#[macro_export]
macro_rules! lc_log {
    ($printer:expr, $level:expr, $fmt:literal, $($arg:tt) *) => {
        // {
        //     let log_fn = Box::new(move |args: &[*const u32]| -> () {
        //         let mut i = 0;
        //         log::log!($level, $fmt , $(
        //             {
        //                 let ret = $crate::lang::printer::_unpack_from_expr(args[i], $arg);
        //                 i += 1;
        //                 ret
        //             }
        //         ), *);
        //     });
        //     let mut printer_args = $crate::lang::PrinterArgs::new();
        //     $(
        //         printer_args.append($arg);
        //     )*
        //     $printer._log($level, printer_args, log_fn);
        // }
        $crate::_log!(
            $printer,
            $level,
            $fmt,
            $($arg)*
        )
    };
}
#[macro_export]
macro_rules! lc_info {
    ($printer:expr, $fmt:literal,$($arg:tt) *) => {
        $crate::lc_log!($printer, log::Level::Info, $fmt, $($arg) *);
    };
}
#[macro_export]
macro_rules! lc_debug {
    ($printer:expr, $fmt:literal, $($arg:tt)*) => {
        $crate::lc_log!($printer, log::Level::Debug, $fmt, $($arg)*);
    };
}
#[macro_export]
macro_rules! lc_warn {
    ($printer:expr, $fmt:literal, $($arg:tt)*) => {
        $crate::lc_log!($printer, log::Level::Warn, $fmt, $($arg)*);
    };
}
#[macro_export]
macro_rules! lc_error {
    ($printer:expr, $fmt:literal, $($arg:tt)*) => {
        $crate::lc_log!($printer, log::Level::Error, $fmt, $($arg)*);
    };
}
pub fn _unpack_from_expr<V: Value>(data: *const u32, _: TypeTag<V>) -> V {
    unsafe { std::ptr::read_unaligned(data as *const V) }
}
impl Printer {
    pub fn new(device: &Device, size: usize) -> Self {
        let data = device.create_buffer(size);
        data.view(0..2).copy_from(&[0, 2]);
        let host_data = vec![0; size];
        Self {
            inner: Arc::new(PrinterData {
                items: RwLock::new(vec![]),
                data,
                host_data,
                dirty: AtomicBool::new(false),
            }),
        }
    }
    pub fn _log(&self, _level: log::Level, args: PrinterArgs, log_fn: LogFn) {
        let inner = &self.inner;
        let data = inner.data.var();
        let offset = data.atomic_fetch_add(1, 1 + args.count as u32);

        let mut items = inner.items.write();
        let item_id = items.len() as u32;

        if_!(
            offset
                .lt(data.len().cast::<u32>())
                .bitand((offset.add(1 + args.count as u32)).le(data.len().cast::<u32>())),
            {
                data.atomic_fetch_add(0, 1);
                data.write(offset, item_id);
                let mut cnt = 0;
                for (i, pack_fn) in args.pack_fn.iter().enumerate() {
                    pack_fn(offset.add(1 + cnt), &data);
                    cnt += args.count_per_arg[i] as u32;
                }
            }
        );

        items.push(PrinterItem {
            log_fn,
            count: args.count + 1,
            count_per_arg: args.count_per_arg,
        });
    }
    pub fn reset(&self) -> PrinterReset {
        PrinterReset { inner: self }
    }
    pub fn print(&self) -> PrinterPrint {
        PrinterPrint { inner: self }
    }
}
pub struct PrinterPrint<'a> {
    inner: &'a Printer,
}
pub struct PrinterReset<'a> {
    inner: &'a Printer,
}
impl<'a> std::ops::Shl<PrinterPrint<'a>> for &'a Scope<'a> {
    type Output = Self;
    fn shl(self, rhs: PrinterPrint<'a>) -> Self::Output {
        self.print(rhs.inner)
    }
}
impl<'a> std::ops::Shl<PrinterReset<'a>> for &'a Scope<'a> {
    type Output = Self;
    fn shl(self, rhs: PrinterReset<'a>) -> Self::Output {
        self.reset_printer(rhs.inner)
    }
}
impl<'a> Scope<'a> {
    pub fn reset_printer(&self, printer: &Printer) -> &Self {
        printer
            .inner
            .dirty
            .store(false, std::sync::atomic::Ordering::Relaxed);
        self.submit([printer.inner.data.view(0..2).copy_from_async(&[0, 2])])
    }
    pub fn print(&self, printer: &Printer) -> &Self {
        assert!(
            !printer
                .inner
                .dirty
                .load(std::sync::atomic::Ordering::Relaxed),
            "must reset printer before printing again!"
        );
        let data = printer.inner.clone();
        let host_data = data.host_data.as_ptr() as *mut u32;
        let host_data = unsafe { std::slice::from_raw_parts_mut(host_data, data.host_data.len()) };
        let cmd = data.data.copy_to_async(host_data);
        let data: Arc<PrinterData> = printer.inner.clone();
        self.submit_with_callback([cmd], move || {
            data.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
            let host_data = &data.host_data;
            let items = data.items.read();
            let mut i = 2;
            let item_count = host_data[0] as usize;
            for _j in 0..item_count {
                if i >= host_data.len() {
                    break;
                }
                let item_idx = host_data[i];
                let item = &items[item_idx as usize];
                let mut offset = 0;
                let unpacked = item
                    .count_per_arg
                    .iter()
                    .map(|c| {
                        let x = &host_data[i + 1 + offset] as *const u32;
                        offset += *c;
                        x
                    })
                    .collect::<Vec<_>>();
                (item.log_fn)(&unpacked);
                i += item.count;
            }
        });
        self
    }
}
