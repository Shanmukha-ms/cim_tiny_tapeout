/*
 * TinyTapeout testbench wrapper for tt_um_hopfield
 * Actual test logic lives in test.py (cocotb)
 */
`default_nettype none
`timescale 1ns / 1ps

module tb ();

    reg  clk;
    reg  rst_n;
    reg  ena;
    reg  [7:0] ui_in;
    wire [7:0] uo_out;
    reg  [7:0] uio_in;
    wire [7:0] uio_out;
    wire [7:0] uio_oe;

    // 50 MHz clock
    initial clk = 0;
    always #10 clk = ~clk;

    tt_um_hopfield dut (
        .ui_in   (ui_in),
        .uo_out  (uo_out),
        .uio_in  (uio_in),
        .uio_out (uio_out),
        .uio_oe  (uio_oe),
        .ena     (ena),
        .clk     (clk),
        .rst_n   (rst_n)
    );

    initial begin
        $dumpfile("tb.fst");
        $dumpvars(0, tb);
        #1;
    end

endmodule
