# Makefile for cocotb simulation of tt_um_bnn
# Usage: make        (runs all tests with iverilog)
#        make clean  (removes build artefacts)

SIM          ?= icarus
TOPLEVEL_LANG ?= verilog
VERILOG_SOURCES = $(PWD)/../src/project.v \
                  $(PWD)/../src/tiny_bnn_comb.v \
                  $(PWD)/../src/tiny_bnn_seq.v

TOPLEVEL     = tb
MODULE       = test

# Include the standard TinyTapeout cocotb makefile fragment
include $(shell cocotb-config --makefiles)/Makefile.sim
