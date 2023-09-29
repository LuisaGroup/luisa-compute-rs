use std::env::current_exe;

use luisa::prelude::*;
use luisa_compute as luisa;

fn main() {
    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_device("cpu");
    let x = device.create_buffer::<f32>(128);
    let y = device.create_buffer::<f32>(128);
    let x_data = (0..x.len()).map(|i| i as f32).collect::<Vec<_>>();
    let stream = device.default_stream();

    // this should not compile
    // stream.with_scope(|s| {
    //     let cmd = {
    //         let tmp_data = (0..y.len()).map(|i| i as f32).collect::<Vec<_>>();
    //         y.copy_from_async(&tmp_data)
    //     };
    //     s.submit([cmd]);
    // });

    // also should not compile
    // {
    //     let s: Scope<'static> = stream.scope();
    //     let cmd = {
    //         let tmp_data = (0..y.len()).map(|i| i as f32).collect::<Vec<_>>();
    //         y.copy_from_async(&tmp_data)
    //     };
    //     s.submit([cmd]);
    // }

    {
        let s = stream.scope();
        {
            let tmp_data = (0..y.len()).map(|i| i as f32).collect::<Vec<_>>();
            // nested lifetime should also be fine
            {
                let tmp_data = (0..y.len()).map(|i| i as f32).collect::<Vec<_>>();
                s.submit([y.copy_from_async(&tmp_data)]);
            };
            s.submit([y.copy_from_async(&tmp_data)]);
        };

        s.submit([x.copy_from_async(&x_data)]);
    }

    stream.with_scope(|s| {
        let tmp_data = (0..y.len()).map(|i| i as f32).collect::<Vec<_>>();
        s.submit([x.copy_from_async(&x_data), y.copy_from_async(&tmp_data)]);
    });
}
