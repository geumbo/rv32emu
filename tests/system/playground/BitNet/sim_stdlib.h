/* Standard headers */
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

/* Type definitions */
typedef int ssize_t;
typedef long off_t;

/* File I/O flags and constants */
#define O_RDONLY 0
#define O_WRONLY 1
#define O_RDWR 2

#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2

/* System control functions */
void exit(int error) __attribute__((noreturn));

/* Performance counter functions */
unsigned long long get_cycles(void);
unsigned long long get_instret(void);

/* Character and string I/O functions */
int putchar(int c);
int puts(char *s);
int printf(const char *format, ...);
int sprintf(char *str, const char *format, ...);

/* File I/O functions */
int open(const char *pathname, int flags);
ssize_t read(int fd, void *buf, size_t count);
off_t lseek(int fd, off_t offset, int whence);
int close(int fd);

/* Utility functions */
int tokscanf(const char *piece, unsigned char *byte_val);
