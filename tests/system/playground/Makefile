include ../../../mk/toolchain.mk

ARCH = -march=rv32izicsr
LINKER_SCRIPT = linker.ld

EMU ?= ../../../build/rv32emu

AFLAGS = -g $(ARCH)
CFLAGS = -g -march=rv32i_zicsr
LDFLAGS = -T $(LINKER_SCRIPT)
EXEC = test.elf

CC = $(CROSS_COMPILE)gcc
AS = $(CROSS_COMPILE)as
LD = $(CROSS_COMPILE)ld
OBJDUMP = $(CROSS_COMPILE)objdump

OBJS = start.o main.o perfcounter.o

.PHONY: all run dump clean

all: $(EXEC)

$(EXEC): $(OBJS) $(LINKER_SCRIPT)
	$(LD) $(LDFLAGS) -o $@ $(OBJS)

%.o: %.S
	$(AS) $(AFLAGS) $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) $< -o $@ -c

run: $(EXEC)
	@test -f $(EMU) || (echo "Error: $(EMU) not found" && exit 1)
	@grep -q "CONFIG_ELF_LOADER=y" ../../../.config || (echo "Error: CONFIG_ELF_LOADER=y not set" && exit 1)
	@grep -q "CONFIG_SYSTEM=y" ../../../.config || (echo "Error: CONFIG_SYSTEM=y not set" && exit 1)
	@grep -q "CONFIG_BNRV=y" ../../../.config || (echo "Error: CONFIG_BNRV=y not set" && exit 1)
	$(EMU) $<

dump: $(EXEC)
	$(OBJDUMP) -Ds $< | less

clean:
	rm -f $(EXEC) $(OBJS)