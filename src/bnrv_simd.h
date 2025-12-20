/*
 * rv32emu is freely redistributable under the MIT License. See the file
 * "LICENSE" for information on usage and redistribution of this file.
 */

#pragma once

#include <stdint.h>
#include "common.h"

/* SIMD capability detection */
#if defined(__SSSE3__)
#define BNRV_USE_SSSE3 1
#include <immintrin.h>
#else
#define BNRV_USE_SSSE3 0
#endif

#if defined(__ARM_NEON)
#define BNRV_USE_NEON 1
#include <arm_neon.h>
#else
#define BNRV_USE_NEON 0
#endif

#define BNRV_USE_SIMD (BNRV_USE_SSSE3 || BNRV_USE_NEON)

/* Map 2-bit weights [00, 01, 10, 11] to coefficients [0, +1, 0, -1].
 * Only the first four entries are used; the rest are padding for alignment.
 */
static const __ALIGNED(16) int8_t bitnet_coeff_map[16] = {
    0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/* SSSE3 Implementation */
#if BNRV_USE_SSSE3

FORCE_INLINE int32_t bitnetadd4_ssse3(uint32_t activations, uint32_t weights)
{
    /* Load LUT into a SIMD register */
    __m128i lut = _mm_loadu_si128((const __m128i *) bitnet_coeff_map);

    /* Load activations: 4 signed bytes in the low 32 bits */
    __m128i a = _mm_cvtsi32_si128((int) activations);

    /* Extract 2-bit weight indices */
    __m128i w = _mm_cvtsi32_si128((int) weights);
    __m128i m = _mm_set1_epi8(0x03);

    __m128i idx0 = _mm_and_si128(w, m);
    __m128i idx1 = _mm_and_si128(_mm_srli_epi32(w, 2), m);
    __m128i idx2 = _mm_and_si128(_mm_srli_epi32(w, 4), m);
    __m128i idx3 = _mm_and_si128(_mm_srli_epi32(w, 6), m);

    /* Pack indices into the low four bytes */
    __m128i indices = _mm_unpacklo_epi8(_mm_unpacklo_epi8(idx0, idx1),
                                        _mm_unpacklo_epi8(idx2, idx3));

    /* In-register LUT lookup */
    __m128i coeffs = _mm_shuffle_epi8(lut, indices);

    /* Multiply activations by coefficients.
     * maddubs performs (unsigned * signed) -> signed int16.
     * We use abs/sign to treat both inputs as signed.
     */
    __m128i prod =
        _mm_maddubs_epi16(_mm_abs_epi8(coeffs), _mm_sign_epi8(a, coeffs));

    /* Horizontal sum of the four lanes */
    __m128i sum = _mm_hadd_epi16(prod, prod);
    return (int16_t) _mm_extract_epi16(sum, 0);
}

#endif /* BNRV_USE_SSSE3 */

/* ARM NEON Implementation */
#if BNRV_USE_NEON

FORCE_INLINE int32_t bitnetadd4_neon(uint32_t activations, uint32_t weights)
{
    /* Load LUT */
    int8x8_t lut = vld1_s8(bitnet_coeff_map);

    /* Load activations: 4 signed bytes in low 32 bits, upper 32 bits are 0 */
    int8x8_t a = vcreate_s8((uint64_t) activations);

    /* Extract 2-bit weight indices from low byte of weights */
    uint8x8_t w = vdup_n_u8((uint8_t) weights);
    static const uint8_t shifts_arr[8] = {0, 2, 4, 6, 0, 0, 0, 0};
    uint8x8_t shifts = vld1_u8(shifts_arr);
    uint8x8_t indices = vand_u8(vshl_u8(w, vneg_s8(vreinterpret_s8_u8(shifts))),
                                vdup_n_u8(0x03));

    /* In-register LUT lookup */
    int8x8_t coeffs = vtbl1_s8(lut, vreinterpret_s8_u8(indices));

    /* Multiply int8 -> int16 */
    int16x8_t prod = vmull_s8(a, coeffs);

    /* Horizontal sum */
#if defined(__aarch64__)
    return vaddvq_s16(prod);
#else
    int16x4_t p0 = vget_low_s16(prod);
    int16x4_t p1 = vget_high_s16(prod);
    int16x4_t s = vadd_s16(p0, p1);
    s = vpadd_s16(s, s);
    s = vpadd_s16(s, s);
    return vget_lane_s16(s, 0);
#endif
}

#endif /* BNRV_USE_NEON */

/* Unified API (compile-time dispatch) */
#if BNRV_USE_SSSE3
#define bitnetadd4(a, w) bitnetadd4_ssse3(a, w)
#elif BNRV_USE_NEON
#define bitnetadd4(a, w) bitnetadd4_neon(a, w)
#else
#error "bnrv_simd.h included but no SIMD support available"
#endif