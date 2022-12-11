#define panic(msg, ...) \
    {\
        std::fprintf(stderr, "panic at %s:%d", __FILE__, __LINE__, msg, ##__VA_ARGS__);\
        std::abort();\
    }
#define lc_assert(cond, msg, ...) \
    if (!(cond)) {\
        std::fprintf(stderr, "assertion failed at %s:%d: " msg, __FILE__, __LINE__, ##__VA_ARGS__);\
        std::abort();\
    }