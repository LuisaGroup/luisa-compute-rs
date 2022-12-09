use std::alloc::Layout;

pub(super) struct BufferImpl {
    pub(super) data: *mut u8,
    pub(super) size: usize,
    pub(super) align: usize,
}

impl BufferImpl {
    pub(super) fn new(size: usize, align: usize) -> Self {
        let layout = Layout::from_size_align(size, align).unwrap();
        let data = unsafe { std::alloc::alloc_zeroed(layout) };
        Self { data, size, align }
    }
}

impl Drop for BufferImpl {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.size, self.align).unwrap();
        unsafe { std::alloc::dealloc(self.data, layout) };
    }
}
