use luisa::prelude::*;
use luisa_compute as luisa;


fn main() {
    init();
    init_logger();
    let device = create_device("cpu").unwrap();
    let shader = device.create_shader::<(BindlessArray, )>(&|a:BindlessArrayVar|{
        let tex = a.tex2d(0);
        let color = tex.read(make_uint2(0, 0));
    }).unwrap();
}
