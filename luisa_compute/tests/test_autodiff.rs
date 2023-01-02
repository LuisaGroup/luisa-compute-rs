use std::ops::Range;

use luisa::prelude::*;
use luisa_compute as luisa;
use rand::prelude::*;
use rayon::{
    prelude::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

fn finite_difference(inputs: &[Float32], f: impl Fn(&[Float32]) -> Float32) -> Vec<Float32> {
    let eps = 1e-4;

    let mut outputs = vec![];
    for i in 0..inputs.len() {
        let mut inputs_add = inputs.to_vec();
        inputs_add[i] += eps;
        let mut inputs_sub = inputs.to_vec();
        inputs_sub[i] -= eps;
        outputs.push((f(&inputs_add) - f(&inputs_sub)) / Float32::from(2.0 * eps));
    }
    outputs
}

fn test_helper<F: Fn(&[Float32]) -> Float32>(
    range: Range<f32>,
    repeats: usize,
    n_inputs: usize,
    f: F,
) {
    let device = create_cpu_device().unwrap();
    let inputs = (0..n_inputs)
        .map(|_| device.create_buffer::<f32>(repeats).unwrap())
        .collect::<Vec<_>>();
    let grad_fd = (0..n_inputs)
        .map(|_| device.create_buffer::<f32>(repeats).unwrap())
        .collect::<Vec<_>>();
    let grad_ad = (0..n_inputs)
        .map(|_| device.create_buffer::<f32>(repeats).unwrap())
        .collect::<Vec<_>>();
    let tic = std::time::Instant::now();
    let tmp: Vec<Vec<f32>> = (0..n_inputs)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            let mut tmp: Vec<f32> = vec![];
            for _ in 0..repeats {
                tmp.push(rng.gen_range(range.clone()));
            }
            tmp
        })
        .collect::<Vec<_>>();
    for i in 0..n_inputs {
        inputs[i].view(..).copy_from(&tmp[i]);
    }

    // let mut rng = rand::thread_rng();
    // for i in 0..n_inputs {
    //     let mut tmp: Vec<f32> = vec![];
    //     for _ in 0..repeats {
    //         tmp.push(rng.gen_range(range.clone()));
    //     }
    //     inputs[i].view(..).copy_from(&tmp);
    // }
    println!("init time: {:?}", tic.elapsed());
    let kernel = device
        .create_kernel(wrap_fn!(0, || {
            let input_vars = inputs.iter().map(|input| input.var()).collect::<Vec<_>>();
            let grad_fd_vars = grad_fd.iter().map(|grad| grad.var()).collect::<Vec<_>>();
            let grad_ad_vars = grad_ad.iter().map(|grad| grad.var()).collect::<Vec<_>>();
            let tid = dispatch_id().x();
            let inputs = input_vars
                .iter()
                .map(|input| input.read(tid))
                .collect::<Vec<_>>();
            autodiff(|| {
                for input in &inputs {
                    requires_grad(*input);
                }
                let output = f(&inputs);
                backward(output);
                for i in 0..n_inputs {
                    grad_ad_vars[i].write(tid, gradient(inputs[i]));
                }
            });
            let fd = finite_difference(&inputs, &f);
            for i in 0..n_inputs {
                grad_fd_vars[i].write(tid, fd[i]);
            }
        }))
        .unwrap();
    let tic = std::time::Instant::now();
    kernel.dispatch([repeats as u32, 1, 1]).unwrap();
    println!("kernel time: {:?}", tic.elapsed());
    let grad_ad_datas = grad_ad
        .iter()
        .map(|grad| {
            let mut data = vec![0.0; repeats];
            grad.view(..).copy_to(&mut data);
            data
        })
        .collect::<Vec<_>>();
    let grad_fd_datas = grad_fd
        .iter()
        .map(|grad| {
            let mut data = vec![0.0; repeats];
            grad.view(..).copy_to(&mut data);
            data
        })
        .collect::<Vec<_>>();
    let input_datas = inputs
        .iter()
        .map(|input| {
            let mut data = vec![0.0; repeats];
            input.view(..).copy_to(&mut data);
            data
        })
        .collect::<Vec<_>>();
    let kernel_dir = kernel.cache_dir().unwrap();
    let mut rel_errors = vec![];
    let mut abs_errors = vec![];
    for r in 0..repeats {
        for i in 0..n_inputs {
            let rel_error = (grad_ad_datas[i][r] - grad_fd_datas[i][r]).abs()
                / (grad_ad_datas[i][r].abs() + 1e-6);
            let abs_error = (grad_ad_datas[i][r] - grad_fd_datas[i][r]).abs();
            assert!(
                abs_error < 5e-2 || rel_error < 5e-2,
                "inputs:{:?} fd: {}, ad: {}, kernel: {:?}",
                (0..n_inputs)
                    .map(|i| input_datas[i][r])
                    .collect::<Vec<f32>>(),
                grad_fd_datas[i][r],
                grad_ad_datas[i][r],
                kernel_dir,
            );
            rel_errors.push(rel_error);
            abs_errors.push(abs_error);
        }
    }
    rel_errors.par_sort_by(|a, b| a.partial_cmp(b).unwrap());
    abs_errors.par_sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_rel_error = rel_errors.iter().sum::<f32>() / rel_errors.len() as f32;
    let mean_abs_error = abs_errors.iter().sum::<f32>() / abs_errors.len() as f32;
    let ninety_ninth_rel_error = rel_errors[(rel_errors.len() * 99) / 100];
    let ninety_ninth_abs_error = abs_errors[(abs_errors.len() * 99) / 100];
    let max_rel_error = *rel_errors
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max_abs_error = *abs_errors
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    assert!(
        ninety_ninth_abs_error <= max_abs_error,
        "{} {}",
        ninety_ninth_abs_error,
        max_abs_error
    );
    assert!(
        ninety_ninth_rel_error <= max_rel_error,
        "{} {}",
        ninety_ninth_rel_error,
        max_rel_error
    );
    assert!(
        mean_rel_error < 3e-3 || mean_abs_error < 1e-2 && ninety_ninth_abs_error < 5e-2,
        "mean rel: {}, mean abs: {}, max rel:{} 99% max rel:{}, max abs:{}, 99% max abs:{}, kernel: {:?}",
        mean_rel_error,
        mean_abs_error,
        max_rel_error, max_abs_error,
        ninety_ninth_rel_error,
        ninety_ninth_abs_error,
        kernel_dir,
    );
}
use std::sync::Once;

static START: Once = Once::new();
fn init_once() {
    START.call_once(|| {
        init();
    });
}
macro_rules! test_1 {
    ($name:ident, $range:expr, $e:expr) => {
        #[test]
        fn $name() {
            init_once();
            test_helper($range, 1024 * 1024, 1, |inputs| {
                let x = inputs[0];
                ($e)(x)
            });
        }
    };
}

test_1!(test_sin, -10.0..10.0, |x: Float32| x.sin());
test_1!(test_cos, -10.0..10.0, |x: Float32| x.cos());
test_1!(test_sincos, -10.0..10.0, |x: Float32| x.cos() * x.sin());
test_1!(test_sqrt, 0.1..10.0, |x: Float32| x.sqrt());
test_1!(test_rsqrt, 0.1..10.0, |x: Float32| x.rsqrt());
test_1!(test_exp, -10.0..3.0, |x: Float32| x.exp());
test_1!(test_exp2, -10.0..3.0, |x: Float32| x.exp2());
test_1!(test_ln, 0.1..10.0, |x: Float32| x.ln());
test_1!(test_log2, 0.1..10.0, |x: Float32| x.log2());
test_1!(test_log10, 0.1..10.0, |x: Float32| x.log10());
test_1!(test_abs, 0.1..10.0, |x: Float32| x.abs());
test_1!(test_abs2, -10.0..-0.1, |x: Float32| x.abs());
test_1!(test_asin, -0.9..0.9, |x: Float32| x.asin());
test_1!(test_acos, -0.9..0.9, |x: Float32| x.acos());
test_1!(test_atan, -10.0..10.0, |x: Float32| x.atan());
test_1!(test_sinh, -10.0..10.0, |x: Float32| x.sinh());
test_1!(test_cosh, -10.0..10.0, |x: Float32| x.cosh());
test_1!(test_tanh, -10.0..10.0, |x: Float32| x.tanh());

#[test]
fn test_vec3_reduce_add_manual() {
    init_once();
    test_helper(-10.0..10.0, 1024 * 1024, 3, |inputs| {
        let x = inputs[0];
        let y = inputs[1];
        let z = inputs[2];
        let v = make_float3(x, y, z);
        v.x() + v.y() + v.z()
    });
}

#[test]
fn test_vec3_reduce_prod_manual() {
    init_once();
    test_helper(-10.0..10.0, 1024 * 1024, 3, |inputs| {
        let x = inputs[0];
        let y = inputs[1];
        let z = inputs[2];
        let v = make_float3(x, y, z);
        v.x() * v.y() * v.z()
    });
}
#[test]
fn test_vec3_reduce_add() {
    init_once();
    test_helper(-10.0..10.0, 1024 * 1024, 3, |inputs| {
        let x = inputs[0];
        let y = inputs[1];
        let z = inputs[2];
        let v = make_float3(x, y, z);
        v.reduce_sum()
    });
}
#[test]
fn test_vec3_reduce_mul() {
    init_once();
    test_helper(-10.0..10.0, 1024 * 1024, 3, |inputs| {
        let x = inputs[0];
        let y = inputs[1];
        let z = inputs[2];
        let v = make_float3(x, y, z);
        v.reduce_prod()
    });
}
#[test]
fn test_vec3_length() {
    init_once();
    test_helper(-10.0..10.0, 1024 * 1024, 3, |inputs| {
        let x = inputs[0];
        let y = inputs[1];
        let z = inputs[2];
        let v = make_float3(x, y, z);
        v.length()
    });
}
#[test]
fn test_vec3_length_squared() {
    init_once();
    test_helper(-2.0..2.0, 1024 * 1024, 3, |inputs| {
        let x = inputs[0];
        let y = inputs[1];
        let z = inputs[2];
        let v = make_float3(x, y, z);
        v.length_squared()
    });
}
#[test]
fn test_vec3_normalize() {
    init_once();
    test_helper(-10.0..10.0, 1024 * 1024, 3, |inputs| {
        let x = inputs[0];
        let y = inputs[1];
        let z = inputs[2];
        let v = make_float3(x, y, z);
        v.normalize().x()
    });
    test_helper(-10.0..10.0, 1024 * 1024, 3, |inputs| {
        let x = inputs[0];
        let y = inputs[1];
        let z = inputs[2];
        let v = make_float3(x, y, z);
        v.normalize().y()
    });
    test_helper(-10.0..10.0, 1024 * 1024, 3, |inputs| {
        let x = inputs[0];
        let y = inputs[1];
        let z = inputs[2];
        let v = make_float3(x, y, z);
        v.normalize().z()
    });
}

#[test]
fn test_vec3_cross_x() {
    init_once();
    test_helper(-10.0..10.0, 1024 * 1024, 6, |inputs| {
        let ax = inputs[0];
        let ay = inputs[1];
        let az = inputs[2];
        let a = make_float3(ax, ay, az);
        let bx = inputs[3];
        let by = inputs[4];
        let bz = inputs[5];
        let b = make_float3(bx, by, bz);
        let v = a.cross(b);
        v.x()
    });
}
#[test]
fn test_vec3_cross_y() {
    init_once();
    test_helper(-10.0..10.0, 1024 * 1024, 6, |inputs| {
        let ax = inputs[0];
        let ay = inputs[1];
        let az = inputs[2];
        let a = make_float3(ax, ay, az);
        let bx = inputs[3];
        let by = inputs[4];
        let bz = inputs[5];
        let b = make_float3(bx, by, bz);
        let v = a.cross(b);
        v.y()
    });
}

#[test]
fn test_vec3_cross_z() {
    init_once();
    test_helper(-10.0..10.0, 1024 * 1024, 6, |inputs| {
        let ax = inputs[0];
        let ay = inputs[1];
        let az = inputs[2];
        let a = make_float3(ax, ay, az);
        let bx = inputs[3];
        let by = inputs[4];
        let bz = inputs[5];
        let b = make_float3(bx, by, bz);
        let v = a.cross(b);
        v.z()
    });
}
#[test]
fn test_vec3_distance() {
    init_once();
    test_helper(-10.0..10.0, 1024 * 1024, 6, |inputs| {
        let ax = inputs[0];
        let ay = inputs[1];
        let az = inputs[2];
        let a = make_float3(ax, ay, az);
        let bx = inputs[3];
        let by = inputs[4];
        let bz = inputs[5];
        let b = make_float3(bx, by, bz);
        a.distance(b)
    });
}
// #[test]
// fn test_vec3_reduce_min(){
//     init_once();
//     test_helper(0.1..1.0, 1024 * 1024, 3, |inputs| {
//         let x = inputs[0];
//         let y = inputs[1];
//         let z = inputs[2];
//         let v = make_float3(x, y, z);
//         v.reduce_min()
//     });
// }

// #[test]
// fn test_vec3_reduce_max(){
//     init_once();
//     test_helper(0.1..1.0, 1024 * 1024, 3, |inputs| {
//         let x = inputs[0];
//         let y = inputs[1];
//         let z = inputs[2];
//         let v = make_float3(x, y, z);
//         v.reduce_max()
//     });
// }
