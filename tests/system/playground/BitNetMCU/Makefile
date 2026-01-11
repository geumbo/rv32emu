# BitNetMCU for rv32emu
include ../../../../mk/toolchain.mk

# Architecture
ARCH = rv32imf
MABI = ilp32f

# Linker script
LINKER_SCRIPT = linker.ld

# BNRV configuration
USE_SIMD?=0

# Emulator path
EMU ?= ../../../../build/rv32emu

# Toolchain
CC = $(CROSS_COMPILE)gcc
AS = $(CROSS_COMPILE)as
OBJDUMP = $(CROSS_COMPILE)objdump

# Compiler flags
AFLAGS = -g -march=$(ARCH)_zicsr -mabi=$(MABI)
CFLAGS = -Ofast -march=$(ARCH) -mabi=$(MABI) -mcmodel=medany
CFLAGS += -fno-common -flto=auto -fomit-frame-pointer -funroll-loops
CFLAGS += -Wno-implicit-int -Wno-implicit-function-declaration
CFLAGS += -DUSE_SIMD=$(USE_SIMD)

# Linker flags
LDFLAGS = -T $(LINKER_SCRIPT) -nostdlib -nostartfiles -flto=auto
LDFLAGS += -march=$(ARCH) -mabi=$(MABI) -mcmodel=medany
LDFLAGS += -lm -lc -lgcc

# Output directory
OUT ?= build

# Source files
OBJS = start.o bnrv.o perfcounter.o run.o sim_stdlib.o
OBJS := $(addprefix $(OUT)/, $(OBJS))

# Output executable
EXEC = $(OUT)/bitnetmcu.elf

.PHONY: all run dump clean

all: $(EXEC)

$(OUT):
	mkdir -p $@

$(EXEC): $(OBJS) $(LINKER_SCRIPT) | $(OUT)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)
	@echo "Build complete: $(EXEC)"
	@ls -lh $(EXEC)

$(OUT)/%.o: %.S | $(OUT)
	$(CC) $(AFLAGS) -c $< -o $@

$(OUT)/%.o: %.c | $(OUT)
	$(CC) $(CFLAGS) -c $< -o $@

run: $(EXEC)
	@test -f $(EMU) || (echo "Error: $(EMU) not found" && exit 1)
	@grep -q "CONFIG_ELF_LOADER=y" ../../../../.config || (echo "Error: CONFIG_ELF_LOADER=y not set" && exit 1)
	@grep -q "CONFIG_SYSTEM=y" ../../../../.config || (echo "Error: CONFIG_SYSTEM=y not set" && exit 1)
	@grep -q "CONFIG_BNRV=y" ../../../../.config || (echo "Error: CONFIG_BNRV=y not set" && exit 1)
	$(EMU) $<

dump: $(EXEC)
	$(OBJDUMP) -Ds $< | less

clean:
	rm -rf $(OUT)