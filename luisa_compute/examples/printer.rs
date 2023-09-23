use std::env::current_exe;

use luisa::prelude::*;
use luisa::printer::*;

use luisa_compute as luisa;

fn main() {
    luisa::init_logger();
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() <= 2,
        "Usage: {} <backend>. <backend>: cpu, cuda, dx, metal, remote",
        args[0]
    );

    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_device(if args.len() == 2 {
        args[1].as_str()
    } else {
        "cpu"
    });
    let printer = Printer::new(&device, 65536);
    let kernel = Kernel::<fn()>::new(&device, track!(|| {
        let id = dispatch_id().xy();
        if id.x == id.y {
            lc_info!(printer, "id = {:?}", id);
        } else {
            lc_info!(printer, "not equal!, id = [{} {}]", id.x, id.y);
        }
    }));
    device.default_stream().with_scope(|s| {
        s.reset_printer(&printer);
        s.submit([kernel.dispatch_async([4, 4, 1])]);
        s.print(&printer);
    });
}
