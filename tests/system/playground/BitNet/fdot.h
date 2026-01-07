/*
 * FDOT Extension Intrinsics
 * Custom RISC-V instructions for FP32 dot product acceleration.
 */

#ifndef FDOT_H
#define FDOT_H

/* FDOT4: 4-element FP32 dot product with accumulation.
 * F[rd] += dot4(mem[rs1], mem[rs2])
 * R-type encoding: opcode=CUSTOM-1 (0x2B), funct3=0, funct7=0
 */
static inline float fdot4_acc(float acc, float *v1, float *v2)
{
    asm volatile(".insn r 0x2B, 0, 0, %0, %1, %2"
                 : "+f"(acc)
                 : "r"(v1), "r"(v2));
    return acc;
}

#endif /* FDOT_H */
