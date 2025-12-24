#include <math.h>
#include <stdint.h>
#include "bnrv.h"
#include "sim_stdlib.h"

// Profiler Variables
uint64_t addsub4_cycles = 0;
uint64_t matmul_cycles = 0;
uint64_t rmsnorm_cycles = 0;
uint64_t quant_cycles = 0;

void qmatmul(int8_t *input, int32_t *output, uint8_t *weight, int n, int d)
{
    uint64_t start = get_cycles();
    for (int i = 0; i < d; i++) {
        output[i] = 0;
#if USE_SIMD == 0
        for (int j = 0; j < n; j++) {
            uint8_t w = weight[(i * n + j) >> 2];
            uint8_t w_shift = (w >> ((j & 0b11) << 1)) & 0b11;
            output[i] +=
                w_shift == 1 ? input[j] : (w_shift == 3 ? -input[j] : 0);
        }
#else
        for (int j = 0; j < n; j += USE_SIMD) {
            int addr = (i * n + j) >> 2;
#if USE_SIMD == 4
            output[i] += __bitnetadd4(*(int8x4_t *) (input + j), weight[addr]);
#elif USE_SIMD == 8
            output[i] += __bitnetadd8(*(int8x4_t *) (input + j),
                                      *(int8x4_t *) (input + j + 4),
                                      *(uint16_t *) (weight + addr));
#elif USE_SIMD == 16
            output[i] +=
                __bitnetadd16(input + j, *(int2x16_t *) (weight + addr));
#elif USE_SIMD == 32
            output[i] +=
                __bitnetadd32(input + j, *(int1x32_t *) (weight + addr),
                              *(int1x32_t *) (weight + addr + 4));
#endif
        }
#endif
    }
    matmul_cycles += get_cycles() - start;
}

void dequantize(int32_t *a, float *af, float s, int d)
{
    uint64_t start = get_cycles();

    for (int i = 0; i < d; i++) {
        af[i] = a[i] * s;
    }

    quant_cycles += get_cycles() - start;
}


// BitNet Linear Forwarding Blocks
void bit_rmsnorm(float *a_out, float *a, int n)
{
    uint64_t start = get_cycles();
    float scale = 0;
    for (int i = 0; i < n; i++) {
        scale += a[i] * a[i];
    }
    scale /= n;
    scale = 1.0f / sqrtf(scale);
    scale /= n;
    for (int i = 0; i < n; i++) {
        a_out[i] = a[i] * scale;
    }
    rmsnorm_cycles += get_cycles() - start;
}

float act_scale(float *a, int n)
{
    uint64_t start = get_cycles();
    float max = -1;
    for (int i = 0; i < n; i++) {
        if (fabs(a[i]) > max) {
            max = fabs(a[i]);
        }
    }
    quant_cycles += get_cycles() - start;
    return max / 127.0;
}

void act_quantize(float *a, int8_t *qa, float s, int n)
{
    uint64_t start = get_cycles();
    float scale = 1.0 / s;
    for (int i = 0; i < n; i++) {
        qa[i] = (int8_t) round(a[i] * scale);
    }
    quant_cycles += get_cycles() - start;
}