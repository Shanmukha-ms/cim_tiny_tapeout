// =============================================================================
// tiny_bnn_seq.v  —  Sequential Binary Neural Network (1 bit per cycle)
// =============================================================================
// Architecture  : single accumulator, latched input, 8-cycle latency
// Dataset       : Breast Cancer Wisconsin — same weights as tiny_bnn_comb
// Gate estimate : ~84 GE including flip-flops
//
// Trade-off vs combinational:
//   - +29 GE overhead (mux, registers, control)
//   - 8× more latency
//   - Scales to 64+ inputs with NO extra adder gates
// =============================================================================

`default_nettype none

module tiny_bnn_seq (
    input  wire       clk,
    input  wire       rst_n,  // active-low reset (synchronous)
    input  wire       start,  // single-cycle pulse: latch x and begin
    input  wire [7:0] x,
    output reg        y,      // classification result (1=benign, 0=malignant)
    output reg        valid   // pulses high for 1 cycle when y is ready
);

    // Trained weights — must match tiny_bnn_comb.v
    localparam [7:0] W      = 8'b00000110;
    localparam [3:0] THRESH = 4'd3;

    reg [7:0] x_reg;
    reg [3:0] accum;
    reg [2:0] bit_idx;
    reg       active;

    wire x_bit    = x_reg[bit_idx];
    wire w_bit    = W[bit_idx];              // constant-index MUX → small LUT
    wire xnor_bit = ~(x_bit ^ w_bit);

    always @(posedge clk) begin
        if (!rst_n) begin
            x_reg   <= 8'd0;
            accum   <= 4'd0;
            bit_idx <= 3'd0;
            active  <= 1'b0;
            y       <= 1'b0;
            valid   <= 1'b0;
        end else begin
            valid <= 1'b0;

            if (start) begin
                x_reg   <= x;
                accum   <= 4'd0;
                bit_idx <= 3'd0;
                active  <= 1'b1;
            end else if (active) begin
                accum <= accum + {3'b000, xnor_bit};

                if (bit_idx == 3'd7) begin
                    y      <= (accum + {3'b000, xnor_bit}) >= THRESH;
                    valid  <= 1'b1;
                    active <= 1'b0;
                end else begin
                    bit_idx <= bit_idx + 3'd1;
                end
            end
        end
    end

endmodule
