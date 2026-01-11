#include "bnrv.h"

// For BitNet SIMD Below 8 elements, the arguments are the data themselves
.global __bitnetadd4
__bitnetadd4:
    BNSUM(a0, a0, a1) // BNADD4 a0 = a(a0) * w(a1)
    ret

.global __bitnetadd8
__bitnetadd8:
    BNSTORE(x0, a2) // BNSTORE a2 -> buffer
    BNSUM(a0, a0, a1) // BNADD8 a0 = a(a0) * w(a1)
    ret

// For BitNet SIMD Above 8 elements, the arguments are the data pointers
.global __bitnetadd16
__bitnetadd16:
    BNSTORE(x0, a1) // BNSTORE a1 -> buffer
    li a5, 0 // result buffer

    // Iteration 1
    lw t0, 0(a0)
    lw t1, 4(a0)
    BNSUM(t2, t0, t1)
    add a5, a5, t2

    // Iteration 2
    lw t0, 8(a0)
    lw t1, 12(a0)
    BNSUM(t2, t0, t1)
    add a5, a5, t2

    add a0, a5, x0
    ret

.global __bitnetadd32
__bitnetadd32:
    BNSTORE(a2, a1) // BNSTORE a2 ## a1 -> buffer
    li a5, 0 // result buffer
    
    // Iteration 1
    lw t0, 0(a0)
    lw t1, 4(a0)
    BNSUM(t2, t0, t1)
    add a5, a5, t2

    // Iteration 2
    lw t0, 8(a0)
    lw t1, 12(a0)
    BNSUM(t2, t0, t1)
    add a5, a5, t2

    // Iteration 3
    lw t0, 16(a0)
    lw t1, 20(a0)
    BNSUM(t2, t0, t1)
    add a5, a5, t2

    // Iteration 4
    lw t0, 24(a0)
    lw t1, 28(a0)
    BNSUM(t2, t0, t1)
    add a5, a5, t2
    add a0, a5, x0
    ret