/*
 * Copyright (c) 2024 Shanmukha Mangadahalli Siddaramu
 * SPDX-License-Identifier: Apache-2.0
 *
 * Hopfield Associative Memory — 7-segment display
 *
 * Stores two 7-segment patterns: digit 1 and digit 9.
 * Given a noisy or partial segment pattern on the inputs, the network
 * iterates each clock cycle until it recalls the closest stored digit.
 *
 * Pin mapping:
 *   ui_in[6:0]  — input segment pattern (possibly noisy)
 *                   bit0=a  bit1=b  bit2=c  bit3=d
 *                   bit4=e  bit5=f  bit6=g
 *   ui_in[7]    — unused
 *   uio_in[0]   — start pulse: load ui_in and begin recall (1-cycle high)
 *   uo_out[6:0] — recalled segment pattern (drives 7-segment display)
 *   uo_out[7]   — decimal point: lights up when network has converged
 *
 * How to use:
 *   1. Set ui_in[6:0] to a segment pattern (use DIP switches)
 *   2. Pulse uio_in[0] high for one clock cycle
 *   3. Watch the display update over a few cycles
 *   4. Decimal point lights up when settled → display shows recalled digit
 *
 * Segment safe-to-corrupt (will still recall correctly):  a, d, f, g
 * Segment fragile (may produce spurious state if corrupted): b, c, e
 */

`default_nettype none

module tt_um_hopfield (
    input  wire [7:0] ui_in,    // Dedicated inputs  — segment pattern
    output wire [7:0] uo_out,   // Dedicated outputs — 7-segment + converged
    input  wire [7:0] uio_in,   // IOs: Input path   — uio_in[0] = start
    output wire [7:0] uio_out,  // IOs: Output path  — unused
    output wire [7:0] uio_oe,   // IOs: Enable       — all inputs
    input  wire       ena,      // always 1 when powered
    input  wire       clk,
    input  wire       rst_n
);

    wire [6:0] state_out;
    wire       converged;

    hopfield_odd u_hop (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (uio_in[0]),
        .x_in      (ui_in[6:0]),
        .state_out (state_out),
        .converged (converged)
    );

    assign uo_out[6:0] = state_out;  // segment pattern
    assign uo_out[7]   = converged;  // decimal point = settled indicator

    assign uio_out = 8'b0;
    assign uio_oe  = 8'b0;

    wire _unused = &{ena, ui_in[7], uio_in[7:1], 1'b0};

endmodule
