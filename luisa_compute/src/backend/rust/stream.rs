use std::{
    collections::VecDeque,
    ptr::null,
    sync::{atomic::AtomicUsize, Arc},
    thread::{self, JoinHandle, Thread},
};

use parking_lot::{Condvar, Mutex};
use rayon;

use crate::backend::rust::shader_impl::BufferView;

use super::{
    resource::BufferImpl,
    shader::ShaderImpl,
    shader_impl::{KernelFnArg, KernelFnArgs},
};
struct StreamContext {
    queue: Mutex<VecDeque<Arc<dyn Fn() + Send + Sync>>>,
    new_work: Condvar,
    sync: Condvar,
    work_count: AtomicUsize,
    finished_count: AtomicUsize,
}
pub(super) struct StreamImpl {
    shared_pool: Arc<rayon::ThreadPool>,
    private_thread: Arc<JoinHandle<()>>,
    ctx: Arc<StreamContext>,
}

impl StreamImpl {
    pub(super) fn new(shared_pool: Arc<rayon::ThreadPool>) -> Self {
        let ctx = Arc::new(StreamContext {
            queue: Mutex::new(VecDeque::new()),
            new_work: Condvar::new(),
            sync: Condvar::new(),
            work_count: AtomicUsize::new(0),
            finished_count: AtomicUsize::new(0),
        });
        let private_thread = {
            let ctx = ctx.clone();
            Arc::new(thread::spawn(move || {
                let mut guard = ctx.queue.lock();
                loop {
                    while guard.is_empty() {
                        ctx.new_work.wait(&mut guard);
                    }
                    // println!("new work");
                    loop {
                        if guard.is_empty() {
                            break;
                        }
                        // println!("get work");
                        let work = guard.pop_front().unwrap();
                        drop(guard);
                        // println!("do work");
                        work();
                        ctx.finished_count
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        // println!("work done");
                        guard = ctx.queue.lock();
                        if guard.is_empty() {
                            // println!("notify");
                            ctx.sync.notify_one();
                            break;
                        }
                    }
                }
            }))
        };
        Self {
            shared_pool,
            private_thread,
            ctx,
        }
    }
    pub(super) fn synchronize(&self) {
        let mut guard = self.ctx.queue.lock();
        while self
            .ctx
            .work_count
            .load(std::sync::atomic::Ordering::Relaxed)
            > self
                .ctx
                .finished_count
                .load(std::sync::atomic::Ordering::Relaxed)
        {
            self.ctx.sync.wait(&mut guard);
        }
    }
    pub(super) fn enqueue(&self, work: impl Fn() + Send + Sync + 'static) {
        let mut guard = self.ctx.queue.lock();
        guard.push_back(Arc::new(work));
        self.ctx
            .work_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.ctx.new_work.notify_one();
    }
    pub(super) fn parallel_for(
        &self,
        kernel: impl Fn(usize) + Send + Sync + 'static,
        block: usize,
        count: usize,
    ) {
        let kernel = Arc::new(kernel);
        let counter = Arc::new(AtomicUsize::new(0));
        let pool = self.shared_pool.clone();
        let nthreads = pool.current_num_threads();
        pool.scope(|s| {
            for _ in 0..nthreads {
                s.spawn(|_| loop {
                    let index = counter.fetch_add(block, std::sync::atomic::Ordering::Relaxed);
                    if index >= count {
                        break;
                    }

                    for i in index..(index + block).min(count) {
                        kernel(i);
                    }
                })
            }
        });
    }
    pub(super) fn dispatch(&self, command_list: &[luisa_compute_api_types::Command]) {
        unsafe {
            for cmd in command_list {
                match cmd {
                    luisa_compute_api_types::Command::BufferUpload(cmd) => {
                        let buffer = &*(cmd.buffer.0 as *mut BufferImpl);
                        let offset = cmd.offset;
                        let size = cmd.size;
                        let data = cmd.data;

                        std::ptr::copy_nonoverlapping(data, buffer.data.add(offset), size);
                    }
                    luisa_compute_api_types::Command::BufferDownload(cmd) => {
                        let buffer = &*(cmd.buffer.0 as *mut BufferImpl);
                        let offset = cmd.offset;
                        let size = cmd.size;
                        let data = cmd.data;
                        std::ptr::copy_nonoverlapping(buffer.data.add(offset), data, size);
                    }
                    luisa_compute_api_types::Command::BufferCopy(_) => todo!(),
                    luisa_compute_api_types::Command::BufferToTextureCopy(_) => todo!(),
                    luisa_compute_api_types::Command::TextureToBufferCopy(_) => todo!(),
                    luisa_compute_api_types::Command::TextureUpload(_) => todo!(),
                    luisa_compute_api_types::Command::TextureDownload(_) => todo!(),
                    luisa_compute_api_types::Command::ShaderDispatch(cmd) => {
                        let shader = &*(cmd.shader.0 as *mut ShaderImpl);
                        let dispatch_size = cmd.dispatch_size;
                        // FIXME: multidimensional dispatch
                        let count = dispatch_size[0] as usize
                            * dispatch_size[1] as usize
                            * dispatch_size[2] as usize;
                        let block = 256;
                        let kernel = shader.fn_ptr();
                        let mut args: Vec<KernelFnArg> = Vec::new();
                        for i in 0..cmd.args_count {
                            let arg = *cmd.args.add(i);
                            match arg {
                                luisa_compute_api_types::Argument::Buffer(buffer_arg) => {
                                    let buffer = &*(buffer_arg.buffer.0 as *mut BufferImpl);
                                    let offset = buffer_arg.offset;
                                    let size = buffer_arg.size;
                                    assert!(offset + size <= buffer.size);
                                    args.push(KernelFnArg::Buffer(BufferView {
                                        data: buffer.data.add(offset),
                                        size,
                                        _marker: std::marker::PhantomData,
                                        len: 0,
                                    }));
                                }
                                luisa_compute_api_types::Argument::Texture(_) => todo!(),
                                luisa_compute_api_types::Argument::Uniform(_) => todo!(),
                                luisa_compute_api_types::Argument::Accel(_) => todo!(),
                                luisa_compute_api_types::Argument::BindlessArray(_) => todo!(),
                            }
                        }
                        let args = Arc::new(args);
                        let kernel_args = KernelFnArgs {
                            captured: null(),
                            args: (*args).as_ptr(),
                            dispatch_id: [0, 0, 0],
                            thread_id: [0, 0, 0],
                            dispatch_size,
                            block_size: [1, 1, 1], // FIXME
                            args_count: args.len(),
                        };

                        self.parallel_for(
                            move |i| {
                                let mut args = kernel_args;
                                let dispatch_z =
                                    i / (args.dispatch_size[0] * args.dispatch_size[1]) as usize;
                                let dispatch_y = (i
                                    % (args.dispatch_size[0] * args.dispatch_size[1]) as usize)
                                    / args.dispatch_size[1] as usize;
                                let dispatch_x = i % args.dispatch_size[0] as usize;
                                args.dispatch_id =
                                    [dispatch_x as u32, dispatch_y as u32, dispatch_z as u32];
                                kernel(&args);
                            },
                            block,
                            count,
                        );
                    }
                    luisa_compute_api_types::Command::MeshBuild(_) => todo!(),
                    luisa_compute_api_types::Command::AccelBuild(_) => todo!(),
                    luisa_compute_api_types::Command::BindlessArrayUpdate(_) => todo!(),
                }
            }
        }
    }
}
