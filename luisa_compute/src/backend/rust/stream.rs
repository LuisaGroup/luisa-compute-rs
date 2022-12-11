use std::{
    collections::VecDeque,
    sync::{atomic::AtomicUsize, Arc},
    thread::{self, JoinHandle, Thread},
};

use parking_lot::{Condvar, Mutex};
use rayon;
struct StreamContext {
    queue: Mutex<VecDeque<Arc<dyn Fn() + Send + Sync>>>,
    new_work: Condvar,
    sync: Condvar,
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
        });
        let private_thread = {
            let ctx = ctx.clone();
            Arc::new(thread::spawn(move || {
                let mut guard = ctx.queue.lock();
                loop {
                    while guard.is_empty() {
                        ctx.new_work.wait(&mut guard);
                    }
                    loop {
                        if guard.is_empty() {
                            ctx.sync.notify_one();
                            break;
                        }
                        let work = guard.pop_front().unwrap();
                        drop(guard);
                        work();
                        guard = ctx.queue.lock();
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
        while !guard.is_empty() {
            self.ctx.sync.wait(&mut guard);
        }
    }
    pub(super) fn enqueue(&self, work: impl Fn() + Send + Sync + 'static) {
        let mut guard = self.ctx.queue.lock();
        guard.push_back(Arc::new(work));
        self.ctx.new_work.notify_one();
    }
    pub(super) fn parallel_for(
        &self,
        kernel: Arc<dyn Fn(usize) + Send + Sync>,
        block: usize,
        count: usize,
    ) {
        let counter = Arc::new(AtomicUsize::new(0));
        let pool = self.shared_pool.clone();
        self.enqueue(move || {
            let nthreads = pool.current_num_threads();
            pool.scope(|s| {
                for _ in 0..nthreads {
                    s.spawn(|_| loop {
                        let index = counter.fetch_add(block, std::sync::atomic::Ordering::Relaxed);
                        if index >= count {
                            break;
                        }
                        while index < count {
                            for i in index..(index + block).min(count) {
                                kernel(i);
                            }
                        }
                    })
                }
            })
        });
    }
}
