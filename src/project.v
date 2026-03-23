/*
 * Copyright (c) 2024 Shanmukha Mangadahalli Siddaramu
 * SPDX-License-Identifier: Apache-2.0
 *
 * TinyTapeout top-level wrapper for a Binary Neural Network (BNN) perceptron
 * trained on the Breast Cancer Wisconsin dataset.
 *
 * Task    : Malignant (y=0) vs Benign (y=1) — 89.5% test accuracy
 * Weights : W = 8'b00000110  THRESH = 3
 *
 * Pin mapping:
 *   ui_in[7:0]  — 8 binarized patient features (x[7:0])
 *   uo_out[0]   — combinational BNN result  (1=benign, 0=malignant)
 *   uo_out[1]   — sequential BNN result     (latched, valid after 8 clocks)
 *   uo_out[2]   — sequential valid strobe   (pulses high for 1 cycle)
 *   uo_out[7:3] — tied 0
 *   uio_in[0]   — start pulse for sequential engine (rising edge triggers)
 *   uio_out     — 0 (unused)
 *   uio_oe      — 0 (all bidir pins are inputs)
 *
 * Feature → ui_in bit mapping (binarize measurements before sending):
 *   ui_in[0] = (mean radius        > 13.27)
 *   ui_in[1] = (mean perimeter     > 85.98)
 *   ui_in[2] = (mean area          > 541.8)
 *   ui_in[3] = (mean concave pts   > 0.0326)
 *   ui_in[4] = (worst radius       > 14.91)
 *   ui_in[5] = (worst perimeter    > 97.59)
 *   ui_in[6] = (worst area         > 683.4)
 *   ui_in[7] = (worst concave pts  > 0.0972)
 */

`default_nettype none

module tt_um_bnn (
    input  wire [7:0] ui_in,    // Dedicated inputs  — binarized feature vector
    output wire [7:0] uo_out,   // Dedicated outputs — BNN results
    input  wire [7:0] uio_in,   // IOs: Input path   — uio_in[0] = seq start
    output wire [7:0] uio_out,  // IOs: Output path  — unused
    output wire [7:0] uio_oe,   // IOs: Enable path  — all inputs
    input  wire       ena,      // always 1 when powered
    input  wire       clk,      // clock
    input  wire       rst_n     // active-low reset
);

    // -------------------------------------------------------------------------
    // Combinational BNN — result available within 3 logic levels
    // -------------------------------------------------------------------------
    wire y_comb;
    tiny_bnn_comb u_comb (
        .x (ui_in),
        .y (y_comb)
    );

    // -------------------------------------------------------------------------
    // Sequential BNN — processes 1 bit per clock, result after 8 cycles
    // start pulse is taken from uio_in[0]
    // -------------------------------------------------------------------------
    wire y_seq, valid_seq;
    tiny_bnn_seq u_seq (
        .clk   (clk),
        .rst_n (rst_n),
        .start (uio_in[0]),
        .x     (ui_in),
        .y     (y_seq),
        .valid (valid_seq)
    );

    // -------------------------------------------------------------------------
    // Output assignments
    // -------------------------------------------------------------------------
    assign uo_out[0] = y_comb;     // combinational result (instant)
    assign uo_out[1] = y_seq;      // sequential result    (latched)
    assign uo_out[2] = valid_seq;  // valid strobe
    assign uo_out[7:3] = 5'b0;

    assign uio_out = 8'b0;
    assign uio_oe  = 8'b0;         // all bidir pins are inputs

    // Suppress unused input warnings
    wire _unused = &{ena, uio_in[7:1], 1'b0};

endmodule
