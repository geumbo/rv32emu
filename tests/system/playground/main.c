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
    *p = '\n';
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
    *p = '\n';
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

/* BNSUM instruction test */
static inline int32_t bnsum_inline(uint32_t activations, uint32_t weights)
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

/* Reference implementation for validation */
static int32_t bnsum_ref(uint32_t activations, uint32_t weights)
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

static void test_bnsum(void)
{
    TEST_LOGGER("=== BNSUM Instruction Test ===\n");

    int passed = 0;
    int failed = 0;

    /* Test 1: All +1 weights */
    {
        int32_t result = bnsum_inline(0x04030201, 0b01010101);
        int32_t expected = bnsum_ref(0x04030201, 0b01010101);
        if (result == expected && result == 10) {
            TEST_LOGGER("PASS: All +1 weights\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: All +1 weights\n");
            failed++;
        }
    }

    /* Test 2: All -1 weights */
    {
        int32_t result = bnsum_inline(0x04030201, 0b11111111);
        int32_t expected = bnsum_ref(0x04030201, 0b11111111);
        if (result == expected && result == -10) {
            TEST_LOGGER("PASS: All -1 weights\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: All -1 weights\n");
            failed++;
        }
    }

    /* Test 3: All zero weights */
    {
        int32_t result = bnsum_inline(0x7F643219, 0b00000000);
        int32_t expected = bnsum_ref(0x7F643219, 0b00000000);
        if (result == expected && result == 0) {
            TEST_LOGGER("PASS: All zero weights\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: All zero weights\n");
            failed++;
        }
    }

    /* Test 4: Mixed weights */
    {
        int32_t result = bnsum_inline(0x08060402, 0b01101101);
        int32_t expected = bnsum_ref(0x08060402, 0b01101101);
        if (result == expected && result == 6) {
            TEST_LOGGER("PASS: Mixed weights\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Mixed weights\n");
            failed++;
        }
    }

    /* Test 5: Negative activations */
    {
        int32_t result = bnsum_inline(0xFFFEFDFC, 0b01010101);
        int32_t expected = bnsum_ref(0xFFFEFDFC, 0b01010101);
        if (result == expected && result == -10) {
            TEST_LOGGER("PASS: Negative activations\n");
            passed++;
        } else {
            TEST_LOGGER("FAIL: Negative activations\n");
            failed++;
        }
    }

    if (failed == 0) {
        TEST_LOGGER("All BNSUM tests passed!\n");
    } else {
        TEST_LOGGER("Some BNSUM tests failed!\n");
    }
}

int main(void)
{
    uint64_t start_cycles, end_cycles, cycles_elapsed;
    uint64_t start_instret, end_instret, instret_elapsed;

    TEST_LOGGER("\n=== BNRV Tests ===\n\n");

    /* Test 0: BNSUM */
    TEST_LOGGER("Test 0: BNSUM\n");
    start_cycles = get_cycles();
    start_instret = get_instret();

    test_bnsum();

    end_cycles = get_cycles();
    end_instret = get_instret();
    cycles_elapsed = end_cycles - start_cycles;
    instret_elapsed = end_instret - start_instret;

    TEST_LOGGER("  Cycles: ");
    print_dec((unsigned long) cycles_elapsed);
    TEST_LOGGER("  Instructions: ");
    print_dec((unsigned long) instret_elapsed);
    TEST_LOGGER("\n");

    TEST_LOGGER("\n=== All Tests Completed ===\n");

    return 0;
}