use std::env::current_exe;
use std::path::PathBuf;

use image::io::Reader as ImageReader;
use luisa::lang::types::vector::alias::*;
use luisa::prelude::*;
use luisa_compute as luisa;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() <= 2,
        "Usage: {} <backend>. <backend>: cpu, cuda, dx, metal, remote",
        args[0]
    );

    luisa::init_logger();
    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_device(if args.len() == 2 {
        args[1].as_str()
    } else {
        "cpu"
    });
    let x = device.create_buffer::<f32>(1024);
    let y = device.create_buffer::<f32>(1024);
    let z = device.create_buffer::<f32>(1024);
    x.view(..).fill_fn(|i| i as f32);
    y.view(..).fill_fn(|i| 1000.0 * i as f32);
    let mut file_path = PathBuf::from(
        PathBuf::from(file!())
            .canonicalize()
            .unwrap()
            .parent()
            .unwrap(),
    );
    file_path.push("logo.png");
    let mip_levels = 4;

    let img = {
        let img = ImageReader::open(file_path).unwrap().decode().unwrap();
        let tex: Tex2d<Float4> =
            device.create_tex2d(PixelStorage::Float4, img.width(), img.height(), mip_levels);
        for i in 0..mip_levels {
            let mip = img
                .resize(
                    img.width() >> i,
                    img.height() >> i,
                    image::imageops::FilterType::Triangle,
                )
                .to_rgba32f();
            let buffer = mip
                .pixels()
                .map(|p| Float4::new(p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32))
                .collect::<Vec<_>>();

            tex.view(i).copy_from(&buffer);
        }
        tex
    };
    let bindless = device.create_bindless_array(2);
    bindless.emplace_buffer_async(0, &x);
    bindless.emplace_buffer_async(1, &y);
    bindless.emplace_tex2d_async(0, &img, Sampler::default());
    bindless.update();
    let kernel = Kernel::<fn(Buffer<f32>)>::new(
        &device,
        &track!(|buf_z| {
            let bindless = bindless.var();
            let tid = dispatch_id().x;
            let buf_x = bindless.buffer::<f32>(0_u32.expr());
            let buf_y = bindless.buffer::<f32>(1_u32.expr());
            let x = buf_x.read(tid).as_::<u32>().as_::<f32>();
            let y = buf_y.read(tid);
            buf_z.write(tid, x + y);
        }),
    );
    kernel.dispatch([1024, 1, 1], &z);
    let mut z_data = vec![0.0; 1024];
    z.view(..).copy_to(&mut z_data);
    println!("{:?}", &z_data[0..16]);
}
