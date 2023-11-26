#include <cstddef>
namespace dummy_orc_eh
{
    union CWrapperFunctionResultDataUnion
    {
        char *ValuePtr;
        char Value[sizeof(ValuePtr)];
    };

    typedef struct
    {
        CWrapperFunctionResultDataUnion Data;
        size_t Size;
    } CWrapperFunctionResult;
    extern "C"
    {
        CWrapperFunctionResult llvm_orc_registerEHFrameSectionWrapper(const char *Data, size_t Size)
        {
            CWrapperFunctionResult r;
            r.Data.ValuePtr = nullptr;
            r.Size = 0;
            return r;
        }
        CWrapperFunctionResult llvm_orc_deregisterEHFrameSectionWrapper(const char *Data, size_t Size)
        {
            CWrapperFunctionResult r;
            r.Data.ValuePtr = nullptr;
            r.Size = 0;
            return r;
        }
    }
}