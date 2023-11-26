use std::ffi::c_char;
// union CWrapperFunctionResultDataUnion {
//     char *ValuePtr;
//     char Value[sizeof(ValuePtr)];
// };

// typedef struct {
//     CWrapperFunctionResultDataUnion Data;
//     size_t Size;
//   } CWrapperFunctionResult;
#[allow(non_snake_case)]
#[repr(C)]
pub union CWrapperFunctionResultDataUnion {
    pub ValuePtr: *const c_char,
    pub Value: [c_char; 8],
}
#[allow(non_snake_case)]
#[repr(C)]
pub struct CWrapperFunctionResult {
    pub Data: CWrapperFunctionResultDataUnion,
    pub Size: usize,
}

extern "C" {
    pub fn llvm_orc_registerEHFrameSectionWrapper(_data: *const c_char, _size: u64);

    pub fn llvm_orc_deregisterEHFrameSectionWrapper(_data: *const c_char, _size: u64);
}
