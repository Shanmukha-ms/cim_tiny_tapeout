// =============================================================================
// tiny_bnn_2out.v  —  2-Neuron BNN (reference / extension design)
// =============================================================================
// Not instantiated by the TinyTapeout top-level (project.v).
// Kept as an extension example showing how to add a second output neuron.
// Gate estimate: ~110 GE total (well under 1000-gate budget).
// =============================================================================

`default_nettype none

module tiny_bnn_2out (
    input  wire [7:0] x,
    output wire       y0,
    output wire       y1
);

    localparam [7:0] W0 = 8'b00000110;   // neuron 0 — breast cancer (benign)
    localparam [7:0] W1 = 8'b11001010;   // neuron 1 — alternative pattern
    localparam [3:0] T0 = 4'd3;
    localparam [3:0] T1 = 4'd5;

    // --- Neuron 0 ---
    wire [7:0] match0 = ~(x ^ W0);
    wire ha0_0s = match0[0] ^ match0[1]; wire ha0_0c = match0[0] & match0[1];
    wire ha0_1s = match0[2] ^ match0[3]; wire ha0_1c = match0[2] & match0[3];
    wire ha0_2s = match0[4] ^ match0[5]; wire ha0_2c = match0[4] & match0[5];
    wire ha0_3s = match0[6] ^ match0[7]; wire ha0_3c = match0[6] & match0[7];
    wire [2:0] p0_01 = {ha0_0c, ha0_0s} + {ha0_1c, ha0_1s};
    wire [2:0] p0_23 = {ha0_2c, ha0_2s} + {ha0_3c, ha0_3s};
    wire [3:0] sum0  = {1'b0, p0_01} + {1'b0, p0_23};
    assign y0 = (sum0 >= T0);

    // --- Neuron 1 ---
    wire [7:0] match1 = ~(x ^ W1);
    wire ha1_0s = match1[0] ^ match1[1]; wire ha1_0c = match1[0] & match1[1];
    wire ha1_1s = match1[2] ^ match1[3]; wire ha1_1c = match1[2] & match1[3];
    wire ha1_2s = match1[4] ^ match1[5]; wire ha1_2c = match1[4] & match1[5];
    wire ha1_3s = match1[6] ^ match1[7]; wire ha1_3c = match1[6] & match1[7];
    wire [2:0] p1_01 = {ha1_0c, ha1_0s} + {ha1_1c, ha1_1s};
    wire [2:0] p1_23 = {ha1_2c, ha1_2s} + {ha1_3c, ha1_3s};
    wire [3:0] sum1  = {1'b0, p1_01} + {1'b0, p1_23};
    assign y1 = (sum1 >= T1);

endmodule
