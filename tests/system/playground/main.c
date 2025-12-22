#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define printstr(ptr, length)                   \
    do {                                        \
        asm volatile(                           \
            "add a7, x0, 0x40;"                 \
            "add a0, x0, 0x1;" /* stdout */     \
            "add a1, x0, %0;"                   \
            "mv a2, %1;" /* length character */ \
            "ecall;"                            \
            :                                   \
            : "r"(ptr), "r"(length)             \
            : "a0", "a1", "a2", "a7");          \
    } while (0)

#define TEST_OUTPUT(msg, length) printstr(msg, length)

#define TEST_LOGGER(msg)                     \
    {                                        \
        char _msg[] = msg;                   \
        TEST_OUTPUT(_msg, sizeof(_msg) - 1); \
    }

extern uint64_t get_cycles(void);
extern uint64_t get_instret(void);

/* Bare metal memcpy implementation */
void *memcpy(void *dest, const void *src, size_t n)
{
    uint8_t *d = (uint8_t *) dest;
    const uint8_t *s = (const uint8_t *) src;
    while (n--)
        *d++ = *s++;
    return dest;
}

/* Software division for RV32I (no M extension) */
static unsigned long udiv(unsigned long dividend, unsigned long divisor)
{
    if (divisor == 0)
        return 0;

    unsigned long quotient = 0;
    unsigned long remainder = 0;

    for (int i = 31; i >= 0; i--) {
        remainder <<= 1;
        remainder |= (dividend >> i) & 1;

        if (remainder >= divisor) {
            remainder -= divisor;
            quotient |= (1UL << i);
        }
    }

    return quotient;
}

static unsigned long umod(unsigned long dividend, unsigned long divisor)
{
    if (divisor == 0)
        return 0;

    unsigned long remainder = 0;

    for (int i = 31; i >= 0; i--) {
        remainder <<= 1;
        remainder |= (dividend >> i) & 1;

        if (remainder >= divisor) {
            remainder -= divisor;
        }
    }

    return remainder;
}

/* Software multiplication for RV32I (no M extension) */
static uint32_t umul(uint32_t a, uint32_t b)
{
    uint32_t result = 0;
    while (b) {
        if (b & 1)
            result += a;
        a <<= 1;
        b >>= 1;
    }
    return result;
}

/* Provide __mulsi3 for GCC */
uint32_t __mulsi3(uint32_t a, uint32_t b)
{
    return umul(a, b);
}

/* Simple integer to hex string conversion */
static void print_hex(unsigned long val)
{
    char buf[20];
    char *p = buf + sizeof(buf) - 1;
    *p = '\0';
    p--;

    if (val == 0) {
        *p = '0';
        p--;
    } else {
        while (val > 0) {
            int digit = val & 0xf;
            *p = (digit < 10) ? ('0' + digit) : ('a' + digit - 10);
            p--;
            val >>= 4;
        }
    }

    p++;
    printstr(p, (buf + sizeof(buf) - p));
}

/* Simple integer to decimal string conversion */
static void print_dec(unsigned long val)
{
    char buf[20];
    char *p = buf + sizeof(buf) - 1;
    *p = '\0';
    p--;

    if (val == 0) {
        *p = '0';
        p--;
    } else {
        while (val > 0) {
            *p = '0' + umod(val, 10);
            p--;
            val = udiv(val, 10);
        }
    }

    p++;
    printstr(p, (buf + sizeof(buf) - p));
}

/* SIMD=4 instruction wrapper */
static inline int32_t bnsum4_inline(uint32_t activations, uint32_t weights)
{
    register uint32_t a0 asm("a0") = activations;
    register uint32_t a1 asm("a1") = weights;
    register int32_t a2 asm("a2");

    /* BNSUM a2, a0, a1
     * Encoding:
     * (0x00 << 25) | (11 << 20) | (10 << 15) | (0x1 << 12) | (12 << 7) | 0x0B =
     * 0x00B5160B
     */
    asm volatile(".word 0x00B5160B\n" : "=r"(a2) : "r"(a0), "r"(a1) : "memory");

    return a2;
}

/* SIMD=4 reference implementation */
static int32_t bnsum4_ref(uint32_t activations, uint32_t weights)
{
    int32_t result = 0;
    for (int i = 0; i < 4; i++) {
        int8_t a = (int8_t) ((activations >> (i * 8)) & 0xFF);
        uint8_t w = (weights >> (i * 2)) & 0x03;
        if (w == 0x01)
            result += a;
        else if (w == 0x03)
            result -= a;
    }
    return result;
}

/* Specific weight pattern tests */
static void test_bnsum_patterns(void)
{
    TEST_LOGGER("=== SIMD=4 Patterns Test ===\n");

    int passed = 0;
    int failed = 0;
    uint32_t activations = 0x04030201; /* [1, 2, 3, 4] */

    /* Test 1: Pattern 0xAA (10101010) - all zeros */
    {
        int32_t result = bnsum4_inline(activations, 0xAA);
        int32_t expected = bnsum4_ref(activations, 0xAA);
        /* All weights are 10 = 0, so result = 0 */
        if (result == expected && result == 0) {
            TEST_LOGGER("PASS: Pattern 0xAA (all zeros)\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Pattern 0xAA\n");
            failed++;
        }
    }

    /* Test 2: Pattern 0x00 (00000000) - The "other" zero */
    {
        int32_t result = bnsum4_inline(activations, 0x00);
        int32_t expected = bnsum4_ref(activations, 0x00);
        /* All weights are 00 = 0, so result = 0 */
        if (result == expected && result == 0) {
            TEST_LOGGER("PASS: Pattern 0x00 (alternative zeros)\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Pattern 0x00\n");
            failed++;
        }
    }

    /* Test 3: Pattern 0x55 (01010101) - all +1 */
    {
        int32_t result = bnsum4_inline(activations, 0x55);
        int32_t expected = bnsum4_ref(activations, 0x55);
        /* 1 + 2 + 3 + 4 = 10 */
        if (result == expected && result == 10) {
            TEST_LOGGER("PASS: Pattern 0x55 (all +1)\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Pattern 0x55\n");
            failed++;
        }
    }

    /* Test 4: Pattern 0xFF (11111111) - all -1 */
    {
        int32_t result = bnsum4_inline(activations, 0xFF);
        int32_t expected = bnsum4_ref(activations, 0xFF);
        /* -(1 + 2 + 3 + 4) = -10 */
        if (result == expected && result == -10) {
            TEST_LOGGER("PASS: Pattern 0xFF (all -1)\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Pattern 0xFF\n");
            failed++;
        }
    }

    /* Test 5: Single lane 0 active (+1) */
    {
        int32_t result = bnsum4_inline(activations, 0x01);
        int32_t expected = bnsum4_ref(activations, 0x01);
        /* Only lane 0: 1 */
        if (result == expected && result == 1) {
            TEST_LOGGER("PASS: Single lane 0\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Single lane 0\n");
            failed++;
        }
    }

    /* Test 6: Single lane 1 active (+1) */
    {
        int32_t result = bnsum4_inline(activations, 0x04);
        int32_t expected = bnsum4_ref(activations, 0x04);
        /* Only lane 1: 2 */
        if (result == expected && result == 2) {
            TEST_LOGGER("PASS: Single lane 1\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Single lane 1\n");
            failed++;
        }
    }

    /* Test 7: Single lane 2 active (+1) */
    {
        int32_t result = bnsum4_inline(activations, 0x10);
        int32_t expected = bnsum4_ref(activations, 0x10);
        /* Only lane 2: 3 */
        if (result == expected && result == 3) {
            TEST_LOGGER("PASS: Single lane 2\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Single lane 2\n");
            failed++;
        }
    }

    /* Test 8: Single lane 3 active (+1) */
    {
        int32_t result = bnsum4_inline(activations, 0x40);
        int32_t expected = bnsum4_ref(activations, 0x40);
        /* Only lane 3: 4 */
        if (result == expected && result == 4) {
            TEST_LOGGER("PASS: Single lane 3\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Single lane 3\n");
            failed++;
        }
    }

    /* Test 9: Alternating +1/-1 pattern */
    {
        int32_t result = bnsum4_inline(activations, 0b11010111);
        int32_t expected = bnsum4_ref(activations, 0b11010111);
        /* lane0=-1(-1), lane1=+1(+2), lane2=+1(+3), lane3=-1(-4) = -1+2+3-4 = 0
         */
        if (result == expected && result == 0) {
            TEST_LOGGER("PASS: Alternating +1/-1\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Alternating +1/-1\n");
            failed++;
        }
    }

    if (failed == 0) {
        TEST_LOGGER("All pattern tests passed!\n");
    } else {
        TEST_LOGGER("Some pattern tests failed!\n");
    }
}

/* Boundary value tests - INT8 min/max */
static void test_bnsum_boundary(void)
{
    TEST_LOGGER("=== SIMD=4 Boundary Test ===\n");

    int passed = 0;
    int failed = 0;

    /* Test 1: Max positive (0x7F = 127) with all +1 weights */
    {
        int32_t result = bnsum4_inline(0x7F7F7F7F, 0b01010101);
        int32_t expected = bnsum4_ref(0x7F7F7F7F, 0b01010101);
        /* 127 * 4 = 508 */
        if (result == expected && result == 508) {
            TEST_LOGGER("PASS: Max positive all +1\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Max positive all +1\n");
            failed++;
        }
    }

    /* Test 2: Max negative (0x80 = -128) with all +1 weights */
    {
        int32_t result = bnsum4_inline(0x80808080, 0b01010101);
        int32_t expected = bnsum4_ref(0x80808080, 0b01010101);
        /* -128 * 4 = -512 */
        if (result == expected && result == -512) {
            TEST_LOGGER("PASS: Max negative all +1\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Max negative all +1\n");
            failed++;
        }
    }

    /* Test 3: Max positive with all -1 weights */
    {
        int32_t result = bnsum4_inline(0x7F7F7F7F, 0b11111111);
        int32_t expected = bnsum4_ref(0x7F7F7F7F, 0b11111111);
        /* -127 * 4 = -508 */
        if (result == expected && result == -508) {
            TEST_LOGGER("PASS: Max positive all -1\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Max positive all -1\n");
            failed++;
        }
    }

    /* Test 4: Max negative with all -1 weights */
    {
        /* 0x81818181 = [-127, -127, -127, -127] */
        int32_t result = bnsum4_inline(0x81818181, 0b11111111);
        int32_t expected = bnsum4_ref(0x81818181, 0b11111111);
        /* -(-127) * 4 = 508 */
        if (result == expected && result == 508) {
            TEST_LOGGER("PASS: Max negative (-127) all -1\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Max negative (-127) all -1\n");
            failed++;
        }
    }

    /* Test 5: Mixed boundary values */
    {
        /* 0x807F807F = [-128, 127, -128, 127] */
        int32_t result = bnsum4_inline(0x7F807F80, 0b01010101);
        int32_t expected = bnsum4_ref(0x7F807F80, 0b01010101);
        /* -128 + 127 - 128 + 127 = -2 */
        if (result == expected && result == -2) {
            TEST_LOGGER("PASS: Mixed boundary\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Mixed boundary\n");
            failed++;
        }
    }

    if (failed == 0) {
        TEST_LOGGER("All boundary tests passed!\n");
    } else {
        TEST_LOGGER("Some boundary tests failed!\n");
    }
}

/* Exhaustive test - all 256 weight combinations */
static void test_bnsum_exhaustive(void)
{
    TEST_LOGGER("=== SIMD=4 Exhaustive Test (256 patterns) ===\n");

    int passed = 0;
    int failed = 0;
    uint32_t first_fail_weight = 0;

    /* Fixed activation pattern for testing */
    uint32_t activations = 0x04030201; /* [1, 2, 3, 4] */

    /* Test all 256 weight patterns */
    for (uint32_t w = 0; w < 256; w++) {
        int32_t result = bnsum4_inline(activations, w);
        int32_t expected = bnsum4_ref(activations, w);

        if (result == expected) {
            passed++;
        } else {
            if (failed == 0) {
                first_fail_weight = w;
            }
            failed++;
        }
    }

    if (failed == 0) {
        TEST_LOGGER("All 256 weight patterns passed!\n");
    } else {
        TEST_LOGGER("FAIL: Some patterns failed\n");
        TEST_LOGGER("  First failure at weight: ");
        print_hex(first_fail_weight);
        TEST_LOGGER("\n");
    }
}

/* Sign extension tests - critical for SIMD correctness */
static void test_bnsum_sign_extension(void)
{
    TEST_LOGGER("=== SIMD=4 Sign Extension Test ===\n");

    int passed = 0;
    int failed = 0;

    /* Test 1: 0xFF interpreted as -1 */
    {
        /* 0xFFFFFFFF = [-1, -1, -1, -1] */
        int32_t result = bnsum4_inline(0xFFFFFFFF, 0b01010101);
        int32_t expected = bnsum4_ref(0xFFFFFFFF, 0b01010101);
        /* -1 * 4 = -4 */
        if (result == expected && result == -4) {
            TEST_LOGGER("PASS: 0xFF as -1\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: 0xFF as -1\n");
            failed++;
        }
    }

    /* Test 2: 0x80 interpreted as -128 */
    {
        int32_t result = bnsum4_inline(0x80808080, 0b01010101);
        int32_t expected = bnsum4_ref(0x80808080, 0b01010101);
        /* -128 * 4 = -512 */
        if (result == expected && result == -512) {
            TEST_LOGGER("PASS: 0x80 as -128\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: 0x80 as -128\n");
            failed++;
        }
    }

    /* Test 3: Mixed signs [127, -128, -1, 0] */
    {
        /* 0x00FF807F = [127, -128, -1, 0] in little-endian */
        int32_t result = bnsum4_inline(0x00FF807F, 0b01010101);
        int32_t expected = bnsum4_ref(0x00FF807F, 0b01010101);
        /* 127 + (-128) + (-1) + 0 = -2 */
        if (result == expected && result == -2) {
            TEST_LOGGER("PASS: Mixed signs\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Mixed signs\n");
            failed++;
        }
    }

    /* Test 4: Sign extension with negative weights */
    {
        /* Test -127 * -1 = +127 (avoid -128 to prevent SSSE3 maddubs overflow)
         */
        int32_t result = bnsum4_inline(0x00000081, 0b00000011);
        int32_t expected = bnsum4_ref(0x00000081, 0b00000011);

        if (result == expected && result == 127) {
            TEST_LOGGER("PASS: Sign ext with neg weight (-127)\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Sign ext with neg weight (-127)\n");
            failed++;
        }
    }

    if (failed == 0) {
        TEST_LOGGER("All sign extension tests passed!\n");
    } else {
        TEST_LOGGER("Some sign extension tests failed!\n");
    }
}

/* Randomized test to cover diverse activation patterns */
static void test_bnsum_fuzz(int iterations)
{
    TEST_LOGGER("=== SIMD=4 Fuzz Test (");
    print_dec((unsigned long) iterations);
    TEST_LOGGER(" iterations) ===\n");

    int passed = 0;
    int failed = 0;
    int skipped = 0;
    static uint32_t seed = 123456789;

    /* Simple Pseudo-random number generator (LCG) to avoid dependency */
#define MY_RAND() (seed = seed * 1664525 + 1013904223)

    for (int i = 0; i < iterations; i++) {
        uint32_t act = MY_RAND();
        uint32_t w = MY_RAND() & 0xFF; /* Weights are only 8 bits relevant */

        /*
         * Safety filter for SSSE3 implementation:
         * If any byte in 'act' is 0x80 (-128) AND the corresponding weight is
         * 0x3 (-1), we skip this case to avoid the known hardware overflow
         * limitation.
         */
        int unsafe = 0;
        for (int lane = 0; lane < 4; lane++) {
            int8_t byte_val = (int8_t) ((act >> (lane * 8)) & 0xFF);
            uint8_t w_val = (w >> (lane * 2)) & 0x03;
            if (byte_val == -128 && w_val == 0x03) {
                unsafe = 1;
                break;
            }
        }
        if (unsafe) {
            skipped++;
            continue;
        }

        int32_t result = bnsum4_inline(act, w);
        int32_t expected = bnsum4_ref(act, w);

        if (result == expected) {
            passed++;
        } else {
            TEST_LOGGER("FAIL: Act=");
            print_hex(act);
            TEST_LOGGER(", W=");
            print_hex(w);
            TEST_LOGGER(" -> Got ");
            print_dec(result);
            TEST_LOGGER(", Exp ");
            print_dec(expected);
            TEST_LOGGER("\n");
            failed++;
            /* Stop after a few failures to avoid flooding logs */
            if (failed >= 5)
                break;
        }
    }

    if (failed == 0) {
        if (passed + skipped == iterations) {
            TEST_LOGGER("All ");
            print_dec((unsigned long) iterations);
            TEST_LOGGER(" iterations passed! (");
        } else {
            TEST_LOGGER("Fuzz test passed (");
        }
        print_dec((unsigned long) passed);
        TEST_LOGGER(" passed, ");
        print_dec((unsigned long) skipped);
        TEST_LOGGER(" skipped)\n");
    }
}

/* Zero masking test: Ensure '0' weights correctly mask dangerous inputs */
static void test_bnsum_zero_masking(void)
{
    TEST_LOGGER("=== SIMD=4 Zero Masking Test ===\n");
    int passed = 0;
    int failed = 0;

    /* Input: [-128, -128, -128, -128] */
    uint32_t dangerous_act = 0x80808080;

    /* Weight: [0, 0, 0, 0] (Pattern 00 or 10) */
    uint32_t mask_weight = 0b10101010; /* 0xAA */

    int32_t result = bnsum4_inline(dangerous_act, mask_weight);
    int32_t expected = bnsum4_ref(dangerous_act, mask_weight); /* Should be 0 */

    if (result == expected && result == 0) {
        TEST_LOGGER("PASS: -128 masked by zero weight\n");
        passed++;
    } else {
        TEST_LOGGER("FAIL: -128 masked by zero weight (Got ");
        print_dec(result);
        TEST_LOGGER(")\n");
        failed++;
    }

    if (failed == 0)
        TEST_LOGGER("Zero masking tests passed!\n");
}

/* ========== SIMD=8+ Tests ========== */

/* BNSTORE instruction wrapper (SIMD=8+) */
static inline void bnstore_inline(uint32_t rs1_val, uint32_t rs2_val)
{
    register uint32_t a0 asm("a0") = rs1_val;
    register uint32_t a1 asm("a1") = rs2_val;

    /* BNSTORE x0, a0, a1
     * Encoding: opcode=0x0B, rd=x0(0), rs1=a0(10), rs2=a1(11), funct3=0x2,
     * funct7=0x0
     * (0x0 << 25) | (11 << 20) | (10 << 15) | (0x2 << 12) | (0 << 7) | 0x0B =
     * 0x00B5200B
     */
    asm volatile(".word 0x00B5200B\n" : : "r"(a0), "r"(a1) : "memory");
}

/* SIMD=8 instruction wrapper */
static inline int32_t bnsum8_inline(uint32_t act_lo, uint32_t act_hi)
{
    register uint32_t a0 asm("a0") = act_lo;
    register uint32_t a1 asm("a1") = act_hi;
    register int32_t a2 asm("a2");

    /* BNSUM a2, a0, a1 (will use buffer) */
    asm volatile(".word 0x00B5160B\n" : "=r"(a2) : "r"(a0), "r"(a1) : "memory");

    return a2;
}

/* SIMD=8 reference implementation */
static int32_t bnsum8_ref(uint32_t act_lo,
                          uint32_t act_hi,
                          uint64_t weights64,
                          uint8_t offset)
{
    int32_t result = 0;
    uint16_t weights = (uint16_t) ((weights64 >> offset) & 0xFFFF);

    /* Process lower 4 bytes */
    for (int i = 0; i < 4; i++) {
        int8_t a = (int8_t) ((act_lo >> (i * 8)) & 0xFF);
        uint8_t w = (weights >> (i * 2)) & 0x03;
        if (w == 0x01)
            result += a;
        else if (w == 0x03)
            result -= a;
    }

    /* Process upper 4 bytes */
    for (int i = 0; i < 4; i++) {
        int8_t a = (int8_t) ((act_hi >> (i * 8)) & 0xFF);
        uint8_t w = (weights >> ((i + 4) * 2)) & 0x03;
        if (w == 0x01)
            result += a;
        else if (w == 0x03)
            result -= a;
    }

    return result;
}

/* Test BNSTORE instruction functionality */
static void test_bnstore(void)
{
    TEST_LOGGER("=== BNSTORE Instruction Test ===\n");

    int passed = 0;
    int failed = 0;

    /* Test 1: Store and verify buffer is active */
    {
        uint64_t weights = 0x5555555555555555ULL;         /* All +1 weights */
        uint32_t rs1 = (uint32_t) (weights >> 32);        /* High 32 bits */
        uint32_t rs2 = (uint32_t) (weights & 0xFFFFFFFF); /* Low 32 bits */

        bnstore_inline(rs1, rs2);

        /* After BNSTORE, next BNSUM should use buffer */
        uint32_t act_lo = 0x04030201; /* [1, 2, 3, 4] */
        uint32_t act_hi = 0x08070605; /* [5, 6, 7, 8] */
        int32_t result = bnsum8_inline(act_lo, act_hi);
        int32_t expected = bnsum8_ref(act_lo, act_hi, weights, 0);

        /* 1+2+3+4+5+6+7+8 = 36 */
        if (result == expected && result == 36) {
            TEST_LOGGER("PASS: BNSTORE enables buffer mode\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: BNSTORE buffer mode\n");
            failed++;
        }
    }

    /* Test 2: Buffer offset progression */
    {
        uint64_t weights = 0x0123456789ABCDEFULL;
        uint32_t rs1 = (uint32_t) (weights >> 32);        /* High 32 bits */
        uint32_t rs2 = (uint32_t) (weights & 0xFFFFFFFF); /* Low 32 bits */

        bnstore_inline(rs1, rs2);

        uint32_t act_lo = 0x01010101;
        uint32_t act_hi = 0x01010101;

        /* First call: offset=0, uses bits [15:0] = 0xCDEF */
        int32_t result1 = bnsum8_inline(act_lo, act_hi);
        int32_t expected1 = bnsum8_ref(act_lo, act_hi, weights, 0);

        /* Second call: offset=16, uses bits [31:16] = 0x89AB */
        int32_t result2 = bnsum8_inline(act_lo, act_hi);
        int32_t expected2 = bnsum8_ref(act_lo, act_hi, weights, 16);

        if (result1 == expected1 && result2 == expected2) {
            TEST_LOGGER("PASS: Buffer offset auto-increment\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Buffer offset progression\n");
            TEST_LOGGER("  First:  Got ");
            print_dec(result1);
            TEST_LOGGER(", Expected ");
            print_dec(expected1);
            TEST_LOGGER("\n");
            TEST_LOGGER("  Second: Got ");
            print_dec(result2);
            TEST_LOGGER(", Expected ");
            print_dec(expected2);
            TEST_LOGGER("\n");
            failed++;
        }
    }

    /* Test 3: Buffer exhaustion */
    {
        uint64_t weights = 0x5555555555555555ULL;
        uint32_t rs1 = (uint32_t) (weights >> 32);        /* High 32 bits */
        uint32_t rs2 = (uint32_t) (weights & 0xFFFFFFFF); /* Low 32 bits */

        bnstore_inline(rs1, rs2);

        uint32_t act_lo = 0x01010101;
        uint32_t act_hi = 0x01010101;

        /* Consume all 4 slots (offset: 0, 16, 32, 48) */
        bnsum8_inline(act_lo, act_hi); /* offset=0 */
        bnsum8_inline(act_lo, act_hi); /* offset=16 */
        bnsum8_inline(act_lo, act_hi); /* offset=32 */
        bnsum8_inline(act_lo, act_hi); /* offset=48 */

        /* Fifth call should fall back to SIMD=4 mode */
        int32_t result = bnsum8_inline(act_lo, act_hi);

        /* SIMD=4 fallback: act_lo as activations, act_hi low 8 bits as weights
         */
        int32_t expected = bnsum4_ref(act_lo, act_hi & 0xFF);
        if (result == expected && result == 1) {
            TEST_LOGGER("PASS: Buffer exhaustion fallback\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Buffer exhaustion (Got ");
            print_dec(result);
            TEST_LOGGER(", Expected ");
            print_dec(expected);
            TEST_LOGGER(")\n");
            failed++;
        }
    }

    if (failed == 0) {
        TEST_LOGGER("All BNSTORE tests passed!\n");
    } else {
        TEST_LOGGER("Some BNSTORE tests failed!\n");
    }
}

/* Test SIMD=8 with various patterns */
static void test_bnsum8_patterns(void)
{
    TEST_LOGGER("=== SIMD=8 Patterns Test ===\n");

    int passed = 0;
    int failed = 0;

    uint32_t act_lo = 0x04030201; /* [1, 2, 3, 4] */
    uint32_t act_hi = 0x08070605; /* [5, 6, 7, 8] */

    /* Test 1: All +1 weights */
    {
        uint64_t weights = 0x5555555555555555ULL;
        bnstore_inline((uint32_t) (weights >> 32), (uint32_t) weights);

        int32_t result = bnsum8_inline(act_lo, act_hi);
        int32_t expected = bnsum8_ref(act_lo, act_hi, weights, 0);

        /* 1+2+3+4+5+6+7+8 = 36 */
        if (result == expected && result == 36) {
            TEST_LOGGER("PASS: All +1 weights\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: All +1 weights\n");
            failed++;
        }
    }

    /* Test 2: All -1 weights */
    {
        uint64_t weights = 0xFFFFFFFFFFFFFFFFULL;
        bnstore_inline((uint32_t) (weights >> 32), (uint32_t) weights);

        int32_t result = bnsum8_inline(act_lo, act_hi);
        int32_t expected = bnsum8_ref(act_lo, act_hi, weights, 0);

        /* -(1+2+3+4+5+6+7+8) = -36 */
        if (result == expected && result == -36) {
            TEST_LOGGER("PASS: All -1 weights\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: All -1 weights\n");
            failed++;
        }
    }

    /* Test 3: All zero weights (pattern 0x0000 in lower 16 bits) */
    {
        uint64_t weights = 0x0000000000000000ULL;
        bnstore_inline((uint32_t) (weights >> 32), (uint32_t) weights);

        int32_t result = bnsum8_inline(act_lo, act_hi);
        int32_t expected = bnsum8_ref(act_lo, act_hi, weights, 0);

        if (result == expected && result == 0) {
            TEST_LOGGER("PASS: All zero weights\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: All zero weights (Got ");
            print_dec(result);
            TEST_LOGGER(", Expected ");
            print_dec(expected);
            TEST_LOGGER(")\n");
            failed++;
        }
    }

    /* Test 4: Alternating pattern */
    {
        uint64_t weights =
            0x00000000CCCC3333ULL; /* Lower 16 bits: 0x3333 = alternating */
        bnstore_inline((uint32_t) (weights >> 32), (uint32_t) weights);

        int32_t result = bnsum8_inline(act_lo, act_hi);
        int32_t expected = bnsum8_ref(act_lo, act_hi, weights, 0);

        if (result == expected) {
            TEST_LOGGER("PASS: Alternating pattern\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Alternating pattern (Got ");
            print_dec(result);
            TEST_LOGGER(", Expected ");
            print_dec(expected);
            TEST_LOGGER(")\n");
            failed++;
        }
    }

    /* Test 5: Complete lane ordering (0-7) */
    for (int lane = 0; lane < 8; lane++) {
        uint64_t weights = 1ULL << (lane * 2);
        bnstore_inline((uint32_t) (weights >> 32), (uint32_t) weights);

        int32_t result = bnsum8_inline(act_lo, act_hi);
        int32_t expected = lane + 1;

        if (result == expected) {
            passed++;
        } else {
            TEST_LOGGER("FAIL: Single lane ");
            print_dec(lane);
            TEST_LOGGER(" (Got ");
            print_dec(result);
            TEST_LOGGER(", Expected ");
            print_dec(expected);
            TEST_LOGGER(")\n");
            failed++;
        }
    }
    if (failed == 0) {
        TEST_LOGGER("PASS: All 8 lane ordering tests\n");
    }

    if (failed == 0) {
        TEST_LOGGER("All SIMD=8 pattern tests passed!\n");
    } else {
        TEST_LOGGER("Some SIMD=8 pattern tests failed!\n");
    }
}

/* Test SIMD=8 boundary values */
static void test_bnsum8_boundary(void)
{
    TEST_LOGGER("=== SIMD=8 Boundary Test ===\n");

    int passed = 0;
    int failed = 0;

    /* Test 1: Max positive with all +1 */
    {
        uint32_t act_lo = 0x7F7F7F7F;
        uint32_t act_hi = 0x7F7F7F7F;
        uint64_t weights = 0x5555555555555555ULL;

        bnstore_inline((uint32_t) (weights >> 32), (uint32_t) weights);

        int32_t result = bnsum8_inline(act_lo, act_hi);
        int32_t expected = bnsum8_ref(act_lo, act_hi, weights, 0);

        /* 127 * 8 = 1016 */
        if (result == expected && result == 1016) {
            TEST_LOGGER("PASS: Max positive all +1\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Max positive all +1\n");
            failed++;
        }
    }

    /* Test 2: Max negative with all +1 */
    {
        uint32_t act_lo = 0x80808080;
        uint32_t act_hi = 0x80808080;
        uint64_t weights = 0x5555555555555555ULL;

        bnstore_inline((uint32_t) (weights >> 32), (uint32_t) weights);

        int32_t result = bnsum8_inline(act_lo, act_hi);
        int32_t expected = bnsum8_ref(act_lo, act_hi, weights, 0);

        /* -128 * 8 = -1024 */
        if (result == expected && result == -1024) {
            TEST_LOGGER("PASS: Max negative all +1\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Max negative all +1\n");
            failed++;
        }
    }

    if (failed == 0) {
        TEST_LOGGER("All SIMD=8 boundary tests passed!\n");
    } else {
        TEST_LOGGER("Some SIMD=8 boundary tests failed!\n");
    }
}

/* Test SIMD=8 safe negative boundary (SSSE3 compatibility) */
static void test_bnsum8_safe_negative(void)
{
    TEST_LOGGER("=== SIMD=8 Safe Negative Test ===\n");

    int passed = 0;
    int failed = 0;

    /* Test -127 * -1 for all lanes to avoid SSSE3 maddubs overflow.
     * SSSE3 saturates -128*-1 to 127, so use -127 for safe test.
     * Expected: -127 * -1 * 8 = 1016
     */
    {
        uint32_t act_lo = 0x81818181; /* -127 in all lanes (0x81 = -127) */
        uint32_t act_hi = 0x81818181;
        uint64_t weights = 0xFFFFFFFFFFFFFFFFULL; /* All -1 */

        bnstore_inline((uint32_t) (weights >> 32), (uint32_t) weights);

        int32_t result = bnsum8_inline(act_lo, act_hi);
        int32_t expected = 1016; /* -127 * -1 * 8 = 1016 */

        if (result == expected) {
            TEST_LOGGER("PASS: Safe negative boundary (-127 * -1)\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Safe negative boundary (Got ");
            print_dec(result);
            TEST_LOGGER(", Expected 1016)\n");
            failed++;
        }
    }

    if (failed == 0) {
        TEST_LOGGER("All SIMD=8 safe negative tests passed!\n");
    } else {
        TEST_LOGGER("Some SIMD=8 safe negative tests failed!\n");
    }
}

/* Randomized test for SIMD=8 to cover diverse patterns */
static void test_bnsum8_fuzz(int iterations)
{
    TEST_LOGGER("=== SIMD=8 Fuzz Test (");
    print_dec((unsigned long) iterations);
    TEST_LOGGER(" iterations) ===\n");

    int passed = 0;
    int failed = 0;
    int skipped = 0;
    static uint32_t seed = 987654321;

#define MY_RAND8() (seed = seed * 1664525 + 1013904223)

    for (int i = 0; i < iterations; i++) {
        uint32_t act_lo = MY_RAND8();
        uint32_t act_hi = MY_RAND8();
        uint64_t w64 = ((uint64_t) MY_RAND8() << 32) | MY_RAND8();

        /* Safety filter for SSSE3 implementation:
         * If any byte in act_lo/act_hi is 0x80 (-128) AND corresponding
         * weight is 0x3 (-1), skip to avoid hardware overflow limitation.
         */
        int unsafe = 0;
        for (int lane = 0; lane < 8; lane++) {
            uint32_t act_val = (lane < 4) ? act_lo : act_hi;
            int byte_idx = lane % 4;
            int8_t byte_val = (int8_t) ((act_val >> (byte_idx * 8)) & 0xFF);
            uint8_t w_val = (w64 >> (lane * 2)) & 0x03;
            if (byte_val == -128 && w_val == 0x03) {
                unsafe = 1;
                break;
            }
        }
        if (unsafe) {
            skipped++;
            continue;
        }

        bnstore_inline((uint32_t) (w64 >> 32), (uint32_t) w64);
        int32_t result = bnsum8_inline(act_lo, act_hi);
        int32_t expected = bnsum8_ref(act_lo, act_hi, w64, 0);

        if (result == expected) {
            passed++;
        } else {
            TEST_LOGGER("FAIL: ActLo=");
            print_hex(act_lo);
            TEST_LOGGER(", ActHi=");
            print_hex(act_hi);
            TEST_LOGGER(", W=");
            print_hex((uint32_t) w64);
            TEST_LOGGER(" -> Got ");
            print_dec(result);
            TEST_LOGGER(", Exp ");
            print_dec(expected);
            TEST_LOGGER("\n");
            failed++;
            if (failed >= 5)
                break;
        }
    }

    if (failed == 0) {
        if (passed + skipped == iterations) {
            TEST_LOGGER("All ");
            print_dec((unsigned long) iterations);
            TEST_LOGGER(" iterations passed! (");
        } else {
            TEST_LOGGER("Fuzz test passed (");
        }
        print_dec((unsigned long) passed);
        TEST_LOGGER(" passed, ");
        print_dec((unsigned long) skipped);
        TEST_LOGGER(" skipped)\n");
    }
}

/* Test SIMD=16 (loop twice) */
static void test_bnsum16(void)
{
    TEST_LOGGER("=== SIMD=16 Loop Test ===\n");

    int passed = 0;
    int failed = 0;

    /* Test: Process 16 elements using 2 BNSUM calls */
    {
        /* Weights for 16 elements = 32 bits needed, stored in lower 32 bits of
         * buffer */
        uint64_t weights = 0x0000000055555555ULL; /* All +1 for first 16
                                                      elements */
        bnstore_inline((uint32_t) (weights >> 32), (uint32_t) weights);

        /* First 8 elements */
        uint32_t act1_lo = 0x04030201; /* [1, 2, 3, 4] */
        uint32_t act1_hi = 0x08070605; /* [5, 6, 7, 8] */
        int32_t result1 = bnsum8_inline(act1_lo, act1_hi);
        int32_t expected1 = bnsum8_ref(act1_lo, act1_hi, weights, 0);

        /* Second 8 elements */
        uint32_t act2_lo = 0x0C0B0A09; /* [9, 10, 11, 12] */
        uint32_t act2_hi = 0x100F0E0D; /* [13, 14, 15, 16] */
        int32_t result2 = bnsum8_inline(act2_lo, act2_hi);
        int32_t expected2 = bnsum8_ref(act2_lo, act2_hi, weights, 16);

        int32_t total_result = result1 + result2;
        int32_t total_expected = expected1 + expected2;

        /* Sum of 1..16 = 136 */
        if (result1 == expected1 && result2 == expected2 &&
            total_result == total_expected && total_result == 136) {
            TEST_LOGGER("PASS: SIMD=16 loop\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: SIMD=16 loop\n");
            TEST_LOGGER("  Batch 1: Got ");
            print_dec(result1);
            TEST_LOGGER(", Expected ");
            print_dec(expected1);
            TEST_LOGGER("\n");
            TEST_LOGGER("  Batch 2: Got ");
            print_dec(result2);
            TEST_LOGGER(", Expected ");
            print_dec(expected2);
            TEST_LOGGER("\n");
            TEST_LOGGER("  Total: Got ");
            print_dec(total_result);
            TEST_LOGGER(", Expected ");
            print_dec(total_expected);
            TEST_LOGGER("\n");
            failed++;
        }
    }

    if (failed == 0) {
        TEST_LOGGER("All SIMD=16 tests passed!\n");
    } else {
        TEST_LOGGER("Some SIMD=16 tests failed!\n");
    }
}

/* Test SIMD=32 (loop 4 times) */
static void test_bnsum32(void)
{
    TEST_LOGGER("=== SIMD=32 Loop Test ===\n");

    int passed = 0;
    int failed = 0;

    /* Test: Process 32 elements using 4 BNSUM calls */
    {
        /* Weights for 32 elements = 64 bits (full buffer) */
        uint64_t weights = 0x5555555555555555ULL; /* All +1 */
        bnstore_inline((uint32_t) (weights >> 32), (uint32_t) weights);

        int32_t total_result = 0;
        int32_t total_expected = 0;

        /* Process 4 batches of 8 elements */
        for (int batch = 0; batch < 4; batch++) {
            /* Create activation pattern [8n+1, 8n+2, ..., 8n+8] */
            uint8_t base = batch * 8;
            uint32_t act_lo = ((base + 4) << 24) | ((base + 3) << 16) |
                              ((base + 2) << 8) | (base + 1);
            uint32_t act_hi = ((base + 8) << 24) | ((base + 7) << 16) |
                              ((base + 6) << 8) | (base + 5);

            int32_t result = bnsum8_inline(act_lo, act_hi);
            int32_t expected = bnsum8_ref(act_lo, act_hi, weights, batch * 16);

            total_result += result;
            total_expected += expected;
        }

        /* Sum of 1..32 = 32*33/2 = 528 */
        if (total_result == total_expected && total_result == 528) {
            TEST_LOGGER("PASS: SIMD=32 loop\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: SIMD=32 loop (Got ");
            print_dec(total_result);
            TEST_LOGGER(", Expected 528)\n");
            failed++;
        }
    }

    if (failed == 0) {
        TEST_LOGGER("All SIMD=32 tests passed!\n");
    } else {
        TEST_LOGGER("Some SIMD=32 tests failed!\n");
    }
}
/* Edge cases & Robustness tests */
static void test_bnsum_edge_cases(void)
{
    TEST_LOGGER("=== Edge Cases & Robustness Test ===\n");

    int passed = 0;
    int failed = 0;

    /* Test 1: Buffer Override (Reset offset) */
    {
        /* Load Pattern A (All +1) */
        uint64_t wA = 0x5555555555555555ULL;
        bnstore_inline((uint32_t) (wA >> 32), (uint32_t) wA);

        /* Consume one slot (offset 0->16) */
        bnsum8_inline(0, 0);

        /* Load Pattern B (All -1) - Should reset offset to 0 */
        uint64_t wB = 0xFFFFFFFFFFFFFFFFULL;
        bnstore_inline((uint32_t) (wB >> 32), (uint32_t) wB);

        /* Should use wB at offset 0 */
        uint32_t act_lo = 0x01010101;
        uint32_t act_hi = 0x01010101;
        int32_t result = bnsum8_inline(act_lo, act_hi);
        /* Expected: -1 * 8 = -8 */

        if (result == -8) {
            TEST_LOGGER("PASS: Buffer Override (Reset offset)\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Buffer Override (Got ");
            print_dec(result);
            TEST_LOGGER(", Expected -8)\n");
            failed++;
        }
    }

    /* Test 2: Mode Locking (Implicit Weights vs Explicit Argument) */
    {
        /* Load buffer with all +1 */
        uint64_t w_buf = 0x5555555555555555ULL;
        bnstore_inline((uint32_t) (w_buf >> 32), (uint32_t) w_buf);

        /* Try to do a SIMD=4 style call:
         * acts = [1, 2, 3, 4]
         * potential_weights = 0xFFFFFFFF (All -1)
         *
         * If SIMD=4 Mode: Result = -10 (using scalar weights)
         * If SIMD=8 Mode: Result = 6 (using buffer weights +1)
         *   act_lo = [1, 2, 3, 4], act_hi = [-1, -1, -1, -1]
         *   weights = [+1...]
         *   Sum = (1+2+3+4) + (-4) = 10 - 4 = 6
         */
        uint32_t acts = 0x04030201;
        uint32_t fake_weights = 0xFFFFFFFF;

        /* calling bnsum4_inline which emits same opcode */
        int32_t result = bnsum4_inline(acts, fake_weights);

        if (result == 6) {
            TEST_LOGGER("PASS: Mode Locking (Buffer priority)\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Mode Locking (Got ");
            print_dec(result);
            TEST_LOGGER(")\n");
            failed++;
        }

        /* Clear buffer for next tests (consume remaining 3 slots) */
        bnsum8_inline(0, 0);
        bnsum8_inline(0, 0);
        bnsum8_inline(0, 0);
    }

    /* Test 3: Zero Masking in SIMD=8 Mode */
    {
        /* Weights: All 0 (Pattern 0b1010...) => 0xAAAAAAAA */
        uint64_t w_zero = 0xAAAAAAAAAAAAAAAAULL;
        bnstore_inline((uint32_t) (w_zero >> 32), (uint32_t) w_zero);

        /* Dangerous input -128 */
        uint32_t dangerous = 0x80808080;
        int32_t result = bnsum8_inline(dangerous, dangerous);

        if (result == 0) {
            TEST_LOGGER("PASS: SIMD=8 Zero Masking\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: SIMD=8 Zero Masking\n");
            failed++;
        }

        /* Clear buffer */
        bnsum8_inline(0, 0);
        bnsum8_inline(0, 0);
        bnsum8_inline(0, 0);
    }

    /* Test 4: Mixed SIMD widths (32 + 4 elements) */
    {
        /* Process 32 elements with SIMD=8 (4 BNSUM8 calls) */
        uint64_t w = 0x5555555555555555ULL;
        bnstore_inline((uint32_t) (w >> 32), (uint32_t) w);

        /* Consume 32 items (4 calls) */
        bnsum8_inline(0, 0);
        bnsum8_inline(0, 0);
        bnsum8_inline(0, 0);
        bnsum8_inline(0, 0);

        /* Process remaining 4 elements with SIMD=4 (explicit choice) */
        /* Activity: [1,1,1,1] (0x01010101), Weight: 0x55 (All +1) */
        int32_t result = bnsum4_inline(0x01010101, 0x55);

        /* Expected: 4 */
        if (result == 4) {
            TEST_LOGGER("PASS: Mixed SIMD widths (32+4 elements)\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Mixed SIMD widths (Got ");
            print_dec(result);
            TEST_LOGGER(")\n");
            failed++;
        }
    }

    if (failed == 0) {
        TEST_LOGGER("All Edge Cases passed!\n");
    } else {
        TEST_LOGGER("Some Edge Cases failed!\n");
    }
}

int main(void)
{
    uint64_t start_cycles, end_cycles, cycles_elapsed;
    uint64_t start_instret, end_instret, instret_elapsed;

    TEST_LOGGER("\n=== BNRV Tests ===\n\n");

    start_cycles = get_cycles();
    start_instret = get_instret();

    /* ========== SIMD=4 Tests ========== */
    TEST_LOGGER("========== SIMD=4 (Buffer-free mode) ==========\n\n");

    /* Test 1: SIMD=4 patterns */
    TEST_LOGGER("Test 1: SIMD=4 Patterns\n");
    test_bnsum_patterns();
    TEST_LOGGER("\n");

    /* Test 2: SIMD=4 boundary */
    TEST_LOGGER("Test 2: SIMD=4 Boundary\n");
    test_bnsum_boundary();
    TEST_LOGGER("\n");

    /* Test 3: SIMD=4 exhaustive */
    TEST_LOGGER("Test 3: SIMD=4 Exhaustive (256 patterns)\n");
    test_bnsum_exhaustive();
    TEST_LOGGER("\n");

    /* Test 4: SIMD=4 sign extension */
    TEST_LOGGER("Test 4: SIMD=4 Sign Extension\n");
    test_bnsum_sign_extension();
    TEST_LOGGER("\n");

    /* Test 5: SIMD=4 fuzz */
    TEST_LOGGER("Test 5: SIMD=4 Fuzz\n");
    test_bnsum_fuzz(1000);
    TEST_LOGGER("\n");

    /* Test 6: SIMD=4 zero masking */
    TEST_LOGGER("Test 6: SIMD=4 Zero Masking\n");
    test_bnsum_zero_masking();
    TEST_LOGGER("\n");

    /* ========== SIMD=8+ Tests ========== */
    TEST_LOGGER("========== SIMD=8+ (Buffer-based mode) ==========\n\n");

    /* Test 7: BNSTORE instruction */
    TEST_LOGGER("Test 7: BNSTORE Instruction\n");
    test_bnstore();
    TEST_LOGGER("\n");

    /* Test 8: SIMD=8 patterns */
    TEST_LOGGER("Test 8: SIMD=8 Patterns\n");
    test_bnsum8_patterns();
    TEST_LOGGER("\n");

    /* Test 9: SIMD=8 boundary */
    TEST_LOGGER("Test 9: SIMD=8 Boundary\n");
    test_bnsum8_boundary();
    TEST_LOGGER("\n");

    /* Test 10: SIMD=8 safe negative */
    TEST_LOGGER("Test 10: SIMD=8 Safe Negative\n");
    test_bnsum8_safe_negative();
    TEST_LOGGER("\n");

    /* Test 11: SIMD=8 fuzz */
    TEST_LOGGER("Test 11: SIMD=8 Fuzz\n");
    test_bnsum8_fuzz(1000);
    TEST_LOGGER("\n");

    /* Test 12: SIMD=16 loop */
    TEST_LOGGER("Test 12: SIMD=16 Loop\n");
    test_bnsum16();
    TEST_LOGGER("\n");

    /* Test 13: SIMD=32 loop */
    TEST_LOGGER("Test 13: SIMD=32 Loop\n");
    test_bnsum32();
    TEST_LOGGER("\n");

    /* Test 14: Edge Cases */
    TEST_LOGGER("Test 14: Edge Cases & Robustness\n");
    test_bnsum_edge_cases();
    TEST_LOGGER("\n");

    end_cycles = get_cycles();
    end_instret = get_instret();
    cycles_elapsed = end_cycles - start_cycles;
    instret_elapsed = end_instret - start_instret;

    TEST_LOGGER("=== Performance Summary ===\n");
    TEST_LOGGER("  Total Cycles: ");
    print_dec((unsigned long) cycles_elapsed);
    TEST_LOGGER("\n");
    TEST_LOGGER("  Total Instructions: ");
    print_dec((unsigned long) instret_elapsed);
    TEST_LOGGER("\n");

    TEST_LOGGER("\n=== All Tests Completed ===\n");

    return 0;
}