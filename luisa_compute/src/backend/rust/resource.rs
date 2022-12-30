use std::alloc::Layout;

#[repr(C)]
pub struct BufferImpl {
    pub data: *mut u8,
    pub size: usize,
    pub align: usize,
}
#[repr(C)]
pub struct BindlessArrayImpl {}

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
