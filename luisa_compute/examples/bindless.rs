use std::path::PathBuf;

use image::io::Reader as ImageReader;
use luisa::prelude::*;
use luisa_compute as luisa;
fn main() {
    init();
    init_logger();
    let device = create_cpu_device().unwrap();
    let x = device.create_buffer::<f32>(1024).unwrap();
    let y = device.create_buffer::<f32>(1024).unwrap();
    let z = device.create_buffer::<f32>(1024).unwrap();
    x.view(..).fill_fn(|i| i as f32);
    y.view(..).fill_fn(|i| 1000.0 * i as f32);
    let mut file_path = PathBuf::from(PathBuf::from(file!()).canonicalize().unwrap().parent().unwrap());
    file_path.push("logo.png");
    let mip_levels = 4;

    let img = {
        let img = ImageReader::open(file_path).unwrap().decode().unwrap();
        let tex: Tex2d<Float4> = device
            .create_tex2d(PixelStorage::Float4, img.width(), img.height(), mip_levels)
            .unwrap();
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
    let bindless = device.create_bindless_array(2).unwrap();
    bindless.emplace_buffer_async(0, &x);
    bindless.emplace_buffer_async(1, &y);
    bindless.emplace_tex2d_async(0, &img, Sampler::default());
    bindless.update();
    let kernel = device
        .create_kernel::<(BufferView<f32>,)>(&|buf_z| {
            let bindless = bindless.var();
            let tid = dispatch_id().x();
            let buf_x = bindless.buffer::<f32>(Uint::from(0));
            let buf_y = bindless.buffer::<f32>(Uint::from(1));
            let x = buf_x.read(tid).uint().float();
            let y = buf_y.read(tid);
            buf_z.write(tid, x + y);
        })
        .unwrap();
    kernel.dispatch([1024, 1, 1], &z).unwrap();
    let mut z_data = vec![0.0; 1024];
    z.view(..).copy_to(&mut z_data);
    println!("{:?}", &z_data[0..16]);
}
