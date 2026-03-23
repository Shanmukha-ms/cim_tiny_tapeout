/*
 * TinyTapeout Verilog testbench for tt_um_bnn
 * Instantiates the standard TT DUT wrapper as required by the cocotb framework.
 * The actual test logic lives in test.py (cocotb).
 *
 * Simulation:
 *   cd test && make
 */

`default_nettype none
`timescale 1ns / 1ps

module tb ();

    // -------------------------------------------------------------------------
    // DUT signals (standard TinyTapeout interface)
    // -------------------------------------------------------------------------
    reg  clk;
    reg  rst_n;
    reg  ena;

    reg  [7:0] ui_in;
    wire [7:0] uo_out;
    reg  [7:0] uio_in;
    wire [7:0] uio_out;
    wire [7:0] uio_oe;

    // -------------------------------------------------------------------------
    // DUT instantiation
    // -------------------------------------------------------------------------
    tt_um_bnn dut (
        .ui_in   (ui_in),
        .uo_out  (uo_out),
        .uio_in  (uio_in),
        .uio_out (uio_out),
        .uio_oe  (uio_oe),
        .ena     (ena),
        .clk     (clk),
        .rst_n   (rst_n)
    );

    // -------------------------------------------------------------------------
    // Clock: 50 MHz → 20 ns period
    // -------------------------------------------------------------------------
    initial clk = 0;
    always #10 clk = ~clk;

    // -------------------------------------------------------------------------
    // VCD dump for GTKWave
    // -------------------------------------------------------------------------
    initial begin
        $dumpfile("tb.vcd");
        $dumpvars(0, tb);
        #1;
    end

endmodule
