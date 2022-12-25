#[derive(Clone, Copy)]
#[repr(C)]
pub struct KernelFnArgs {
    pub captured: *const KernelFnArg,
    pub args: *const KernelFnArg,
    pub args_count: usize,
    pub dispatch_id: [u32; 3],
    pub thread_id: [u32; 3],
    pub dispatch_size: [u32; 3],
    pub block_size: [u32; 3],
}
unsafe impl Send for KernelFnArgs {}
unsafe impl Sync for KernelFnArgs {}
impl KernelFnArgs {
    pub fn captured(&self, i: usize) -> &KernelFnArg {
        unsafe { &*self.captured.add(i) }
    }
    pub fn args(&self, i: usize) -> &KernelFnArg {
        assert!(i < self.args_count, "index out of bounds: {} >= {}", i, self.args_count);
        unsafe { &*self.args.add(i) }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct BufferView<T> {
    pub data: *mut u8,
    pub size: usize,
    pub _marker: std::marker::PhantomData<T>,
    pub len: usize,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct BindlessArrayImpl {}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum KernelFnArg {
    Buffer(BufferView<()>),
    BindlessArray(BindlessArrayImpl),
}

impl<T: Copy> BufferView<T> {
    #[inline]
    pub fn read(&self, i: usize) -> T {
        assert!(i < self.len, "index out of bounds: {} >= {}", i, self.len);
        unsafe { std::ptr::read((self.data as *const T).add(i)) }
    }
    #[inline]
    pub fn write(&self, i: usize, value: T) {
        assert!(i < self.len, "index out of bounds: {} >= {}", i, self.len);
        unsafe { std::ptr::write((self.data as *mut T).add(i), value) }
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.size / std::mem::size_of::<T>()
    }
}

impl KernelFnArg {
    #[inline]
    pub fn as_buffer<T>(&self) -> Option<BufferView<T>> {
        match self {
            Self::Buffer(b) => Some(BufferView {
                data: b.data,
                size: b.size,
                _marker: std::marker::PhantomData,
                len: b.size / std::mem::size_of::<T>(),
            }),
            _ => None,
        }
    }
    #[inline]
    pub fn as_bindless_array(&self) -> Option<BindlessArrayImpl> {
        match self {
            Self::BindlessArray(b) => Some(*b),
            _ => None,
        }
    }
}
