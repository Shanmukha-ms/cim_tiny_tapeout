// =============================================================================
// hopfield_odd.v  —  Hopfield Associative Memory storing digits 1 & 9
// =============================================================================
// Architecture : 7-node synchronous Hopfield network with 2-cycle detection
//
// Stored patterns (bipolar Hebbian learning: W = p1⊗p1 + p9⊗p9):
//   digit 1 : 7'b0000110  segments b,c lit
//   digit 9 : 7'b1101111  segments a,b,c,d,f,g lit
//
// Weight matrix (symmetric, zero diagonal):
//      a    b    c    d    e    f    g
//  a [ 0,   0,   0,  +2,   0,  +2,  +2 ]
//  b [ 0,   0,  +2,   0,  -2,   0,   0 ]
//  c [ 0,  +2,   0,   0,  -2,   0,   0 ]
//  d [+2,   0,   0,   0,   0,  +2,  +2 ]
//  e [ 0,  -2,  -2,   0,   0,   0,   0 ]
//  f [+2,   0,   0,  +2,   0,   0,  +2 ]
//  g [+2,   0,   0,  +2,   0,  +2,   0 ]
//
// Hardware update rules (derived from W, binary domain):
//   new_a = majority(d,f,g)  = (d&f)|(f&g)|(d&g)
//   new_b = c & ~e
//   new_c = b & ~e
//   new_d = majority(a,f,g)  = (a&f)|(f&g)|(a&g)
//   new_e = ~(b & c)         = NAND(b,c)
//   new_f = majority(a,d,g)  = (a&d)|(d&g)|(a&g)
//   new_g = majority(a,d,f)  = (a&d)|(d&f)|(a&f)
//
// Segment bit mapping:
//   bit 0 = a (top)          bit 4 = e (bottom-left)
//   bit 1 = b (top-right)    bit 5 = f (top-left)
//   bit 2 = c (bottom-right) bit 6 = g (middle)
//   bit 3 = d (bottom)
//
// Basin of attraction (safe flips → correct recall):
//   Segments a, d, f, g : robust  (majority vote protects them)
//   Segments b, c, e    : fragile (mutually dependent triangle)
//
// Gate estimate : ~140 GE
// =============================================================================

`default_nettype none

module hopfield_odd (
    input  wire       clk,
    input  wire       rst_n,      // synchronous active-low reset
    input  wire       start,      // 1-cycle pulse: load x_in as initial state
    input  wire [6:0] x_in,       // noisy/partial input pattern
    output wire [6:0] state_out,  // settled output → drives 7-segment display
    output wire       converged   // high when network has reached a stable state
);

    // -------------------------------------------------------------------------
    // State registers
    // -------------------------------------------------------------------------
    reg [6:0] state;   // current network state
    reg [6:0] prev;    // previous state (for 2-cycle detection)

    // Segment aliases
    wire sa = state[0];
    wire sb = state[1];
    wire sc = state[2];
    wire sd = state[3];
    wire se = state[4];
    wire sf = state[5];
    wire sg = state[6];

    // -------------------------------------------------------------------------
    // Combinational Hopfield update (one synchronous iteration)
    // -------------------------------------------------------------------------
    wire [6:0] next_state;

    assign next_state[0] = (sd & sf) | (sf & sg) | (sd & sg);  // majority(d,f,g)
    assign next_state[1] = sc & ~se;                             // c AND NOT e
    assign next_state[2] = sb & ~se;                             // b AND NOT e
    assign next_state[3] = (sa & sf) | (sf & sg) | (sa & sg);  // majority(a,f,g)
    assign next_state[4] = ~(sb & sc);                           // NAND(b,c)
    assign next_state[5] = (sa & sd) | (sd & sg) | (sa & sg);  // majority(a,d,g)
    assign next_state[6] = (sa & sd) | (sd & sf) | (sa & sf);  // majority(a,d,f)

    // -------------------------------------------------------------------------
    // Convergence detection
    //   fixed_point : next == current  (stable fixed-point attractor)
    //   two_cycle   : next == previous (network oscillating → freeze it)
    // -------------------------------------------------------------------------
    wire fixed_point = (next_state == state);
    wire two_cycle   = (next_state == prev);
    assign converged = fixed_point | two_cycle;

    // -------------------------------------------------------------------------
    // Sequential update — iterate until converged, then freeze
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= 7'b0000110;  // power-on default: digit 1
            prev  <= 7'b0;
        end else if (start) begin
            state <= x_in;        // load user-supplied noisy pattern
            prev  <= 7'b0;        // reset history so two_cycle check is clean
        end else if (!converged) begin
            prev  <= state;       // shift current → previous
            state <= next_state;  // advance one Hopfield step
        end
        // converged: hold state and prev unchanged
    end

    assign state_out = state;

endmodule
