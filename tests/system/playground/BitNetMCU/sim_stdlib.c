#include "sim_stdlib.h"

/* ============================================================================
 * System control functions
 * ============================================================================
 */

void exit(int error)
{
    register long a7 asm("a7") = 93;  // exit syscall number
    register long a0 asm("a0") = error;
    asm volatile("ecall" : : "r"(a7), "r"(a0));
    while (1)
        ;
}

/* ============================================================================
 * Character and string I/O functions
 * ============================================================================
 */

int putchar(int c)
{
    char buf = (char) c;
    register long a7 asm("a7") = 64;  // write syscall number
    register long a0 asm("a0") = 1;   // stdout fd
    register long a1 asm("a1") = (long) &buf;
    register long a2 asm("a2") = 1;  // length
    asm volatile("ecall" : "+r"(a0) : "r"(a7), "r"(a1), "r"(a2) : "memory");
    return c;
}

static void printf_c(int c)
{
    putchar(c);
}

static void printf_s(char *p)
{
    while (*p) {
        putchar(*(p++));
    }
}

static void printf_d(int val)
{
    char buffer[32];
    char *p = buffer;
    if (val < 0) {
        printf_c('-');
        val = -val;
    }
    while (val || p == buffer) {
        *(p++) = '0' + val % 10;
        val = val / 10;
    }
    while (p != buffer)
        printf_c(*(--p));
}

static void printf_x(uint32_t val)
{
    char buffer[32];
    char *p = buffer;
    printf_c('0');
    printf_c('x');
    while (val || p == buffer) {
        int mod = val % 16;
        if (mod < 10) {
            *(p++) = '0' + mod;
        } else {
            *(p++) = 'A' + mod - 10;
        }
        val = val / 16;
    }
    while (p != buffer)
        printf_c(*(--p));
}

static void printf_llu(uint64_t val)
{
    char buffer[24];
    int pos = 0;

    if (val == 0) {
        printf_c('0');
        return;
    }

    // Extract digits
    // Start from the highest digit (10^19 is sufficient for 64-bit)
    uint64_t powers[] = {10000000000000000000ULL,
                         1000000000000000000ULL,
                         100000000000000000ULL,
                         10000000000000000ULL,
                         1000000000000000ULL,
                         100000000000000ULL,
                         10000000000000ULL,
                         1000000000000ULL,
                         100000000000ULL,
                         10000000000ULL,
                         1000000000ULL,
                         100000000ULL,
                         10000000ULL,
                         1000000ULL,
                         100000ULL,
                         10000ULL,
                         1000ULL,
                         100ULL,
                         10ULL,
                         1ULL};

    int started = 0;
    for (int i = 0; i < 20; i++) {
        int digit = 0;
        while (val >= powers[i]) {
            val -= powers[i];
            digit++;
        }
        if (digit > 0 || started) {
            buffer[pos++] = '0' + digit;
            started = 1;
        }
    }

    for (int i = 0; i < pos; i++) {
        printf_c(buffer[i]);
    }
}

int printf(const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    for (int i = 0; format[i]; i++) {
        if (format[i] == '%') {
            while (format[++i]) {
                if (format[i] == 'c') {
                    printf_c(va_arg(ap, int));
                    break;
                }
                if (format[i] == 's') {
                    printf_s(va_arg(ap, char *));
                    break;
                }
                if (format[i] == 'l' && format[i + 1] == 'l' &&
                    format[i + 2] == 'u') {
                    printf_llu(va_arg(ap, uint64_t));
                    i += 2;
                    break;
                }
                if (format[i] == 'd') {
                    printf_d(va_arg(ap, int));
                    break;
                }
                if (format[i] == 'x') {
                    printf_x(va_arg(ap, uint32_t));
                    break;
                }
            }
        } else {
            printf_c(format[i]);
        }
    }
    va_end(ap);
    return 0;
}

int puts(char *s)
{
    while (*s) {
        putchar(*s);
        s++;
    }
    putchar('\n');
    return 0;
}

int sprintf(char *str, const char *format, ...)
{
    char *str_ptr = str;
    va_list ap;
    va_start(ap, format);
    for (int i = 0; format[i]; i++)
        if (format[i] == '%') {
            while (format[++i]) {
                if (format[i] == 's') {
                    char *s = va_arg(ap, char *);
                    while (*s) {
                        *str_ptr = *s;
                        str_ptr++;
                        s++;
                    }
                    break;
                } else {
                    // other formats are not supported for now
                }
            }
        } else {
            *str_ptr = format[i];
            str_ptr++;
        }
    *str_ptr = '\0';
    va_end(ap);
    return 0;
}

/* ============================================================================
 * File I/O functions
 * ============================================================================
 */

int open(const char *pathname, int flags)
{
    register long a7 asm("a7") = 1024;  // open syscall
    register long a0 asm("a0") = (long) pathname;
    register long a1 asm("a1") = flags;
    register long a2 asm("a2") = 0644;  // mode
    asm volatile("ecall" : "+r"(a0) : "r"(a7), "r"(a1), "r"(a2) : "memory");
    return a0;
}

ssize_t read(int fd, void *buf, size_t count)
{
    register long a7 asm("a7") = 63;  // read syscall
    register long a0 asm("a0") = fd;
    register long a1 asm("a1") = (long) buf;
    register long a2 asm("a2") = count;
    asm volatile("ecall" : "+r"(a0) : "r"(a7), "r"(a1), "r"(a2) : "memory");
    return a0;
}

off_t lseek(int fd, off_t offset, int whence)
{
    register long a7 asm("a7") = 62;  // lseek syscall
    register long a0 asm("a0") = fd;
    register long a1 asm("a1") = offset;
    register long a2 asm("a2") = whence;
    asm volatile("ecall" : "+r"(a0) : "r"(a7), "r"(a1), "r"(a2) : "memory");
    return a0;
}

int close(int fd)
{
    register long a7 asm("a7") = 57;  // close syscall
    register long a0 asm("a0") = fd;
    asm volatile("ecall" : "+r"(a0) : "r"(a7) : "memory");
    return a0;
}

/* ============================================================================
 * Memory management functions
 * ============================================================================
 */

void *_sbrk(int incr)
{
    extern char end;               // Symbol defined by linker script (_end)
    static char *heap_end = NULL;  // Current heap break (initialized once)
    char *prev_heap_end;

    if (heap_end == NULL)
        heap_end = &end;  // Initialize heap at end of BSS

    prev_heap_end = heap_end;
    heap_end += incr;  // Bump allocator: grow heap upward
    return prev_heap_end;
}

/* ============================================================================
 * Utility functions
 * ============================================================================
 */

static int hex2int(char c)
{
    if (c >= '0' && c <= '9') {
        return c - '0';
    } else if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    } else if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return 0;
}

int tokscanf(const char *piece, unsigned char *byte_val)
{
    // const char* pat = "<0x00>";
    if (piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' ||
        piece[5] != '>') {
        return 0;
    }
    char h = piece[3];
    char l = piece[4];
    // convert the hex char to int
    *byte_val = (hex2int(h) << 4) | hex2int(l);
    return 1;
}

void *bsearch(const void *key,
              const void *ptr,
              size_t count,
              size_t size,
              int (*comp)(const void *, const void *))
{
    const char *base = ptr;
    size_t lim;
    int cmp;
    const void *p;

    for (lim = count; lim != 0; lim >>= 1) {
        p = base + (lim >> 1) * size;
        cmp = comp(key, p);
        if (cmp == 0) {
            return (void *) p;
        }
        if (cmp > 0) {
            base = (char *) p + size;
            lim--;
        }
    }
    return (void *) NULL;
}

/* ============================================================================
 * Newlib syscall stubs
 * ============================================================================
 */

// Stub for closing files (not used)
int _close(int file)
{
    return -1;
}

// Stub for seeking files (not used)
int _lseek(int file, int ptr, int dir)
{
    return 0;
}

// Stub for reading files (not used)
int _read(int file, char *ptr, int len)
{
    return 0;
}

// Implementation for writing to stdout (used by printf)
// Redirects output to the simulator console via putchar
int _write(int file, char *ptr, int len)
{
    for (int i = 0; i < len; i++) {
        // putchar is defined in this file
        putchar(ptr[i]);
    }
    return len;
}
