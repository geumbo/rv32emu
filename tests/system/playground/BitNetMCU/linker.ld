OUTPUT_ARCH( "riscv" )

ENTRY(_start)

SECTIONS
{
  . = 0x10000;
  .text : {
    *(.text._start)
    *(.text)
  }

  .data : { *(.data) }

  .bss : {
   __bss_start = .;
    *(.sbss*)
    *(.bss .bss.*)
    *(COMMON)
    __bss_end = .;
  }

  PROVIDE( _end = . );
  PROVIDE( end = . );

  PROVIDE(__stack_top = 0x01000000);
}