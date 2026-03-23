# Makefile for cocotb simulation of tt_um_bnn
# Usage: make        (runs all tests with iverilog)
#        make clean  (removes build artefacts)

SIM           ?= icarus
TOPLEVEL_LANG ?= verilog

# All Verilog sources — testbench first, then DUT hierarchy
VERILOG_SOURCES += $(PWD)/tb.v
VERILOG_SOURCES += $(PWD)/../src/project.v
VERILOG_SOURCES += $(PWD)/../src/hopfield_odd.v

TOPLEVEL = tb
MODULE   = test

# Cocotb simulation makefile
include $(shell cocotb-config --makefiles)/Makefile.sim
