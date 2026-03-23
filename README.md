# Hopfield Associative Memory — Odd Digit Recall on 7-Segment Display

[![TinyTapeout](https://img.shields.io/badge/TinyTapeout-TTIHP-blue)](https://tinytapeout.com)
[![test](https://github.com/Shanmukha-ms/cim_tiny_tapeout/actions/workflows/test.yaml/badge.svg)](https://github.com/Shanmukha-ms/cim_tiny_tapeout/actions/workflows/test.yaml)
[![gds](https://github.com/Shanmukha-ms/cim_tiny_tapeout/actions/workflows/gds.yaml/badge.svg)](https://github.com/Shanmukha-ms/cim_tiny_tapeout/actions/workflows/gds.yaml)

A **7-node Hopfield Neural Network** implemented in ~140 gate equivalents on TinyTapeout.
Set any broken segment pattern on the DIP switches, press start — the chip
recalls the nearest stored digit and displays it on the 7-segment display.

**Two stored memories** · **~140 GE** · **Converges in 1–3 clock cycles**

---

## What is a Hopfield Network?

A Hopfield network is a **recurrent neural network** that acts as an
associative (content-addressable) memory. Unlike a lookup table, it does not
search through stored entries — it *converges* to the closest stored pattern
through iterated computation.

```
                    ┌─────────────────────────────────┐
                    │      Hopfield Network            │
  Noisy input  ───► │  [nodes update each other each  │ ───► Clean recalled
  (broken digit)    │   clock cycle until stable]      │      digit on display
                    └─────────────────────────────────┘
```

**Key property:** Give it a pattern with some segments wrong or missing,
and it fills in the correct segments automatically.

---

## Stored Memories

This design stores exactly **two** 7-segment display patterns:

```
  Digit 1              Digit 9
  ┌─ ─ ─┐             ┌─────┐
        |             |     |
        |             └─────┘
        |                   |
  └─ ─ ─┘             └─────┘

  Segments: b, c       Segments: a, b, c, d, f, g
  Binary:   0000110    Binary:   1101111
  Hex:      0x06       Hex:      0x6F
```

### Why only two patterns?

A Hopfield network of N nodes can reliably store at most **0.138 × N** patterns.
With 7 nodes (one per segment): capacity ≈ 0.138 × 7 = **~1 pattern**.

Storing exactly 2 works by carefully choosing patterns that are as
**orthogonal** (dissimilar) as possible — digit 1 and digit 9 have the lowest
cross-correlation (dot product = −1) of all odd digit pairs.

| Pair   | Dot product | Both stable? |
|--------|-------------|--------------|
| 1 & 9  | **−1**      | ✓ best choice |
| 5 & 7  | −1          | ✓             |
| 1 & 3  | +1          | ✓             |
| 3 & 9  | +5          | ✗ too similar |

---

## Architecture

### System overview

```
  ui_in[6:0] ──────────────────────────────────────────────────────────┐
  (DIP switches)                                                        │
                                                                        ▼
  uio_in[0] ──► [start]     ┌──────────────────────────────────────────┤
  (start btn)               │         hopfield_odd module               │
                            │                                           │
                            │  ┌─────────────────────────────────────┐ │
                            │  │   7 state flip-flops  (one/segment) │ │
                            │  │   sa  sb  sc  sd  se  sf  sg        │ │
                            │  └──────────────┬──────────────────────┘ │
                            │                 │  (feedback)             │
                            │  ┌──────────────▼──────────────────────┐ │
                            │  │   Combinational update logic         │ │
                            │  │                                      │ │
                            │  │   new_a = majority(d, f, g)         │ │
                            │  │   new_b = c AND NOT e               │ │
                            │  │   new_c = b AND NOT e               │ │
                            │  │   new_d = majority(a, f, g)         │ │
                            │  │   new_e = NAND(b, c)                │ │
                            │  │   new_f = majority(a, d, g)         │ │
                            │  │   new_g = majority(a, d, f)         │ │
                            │  └──────────────┬──────────────────────┘ │
                            │                 │                         │
                            │  ┌──────────────▼──────────────────────┐ │
                            │  │   Convergence detector               │ │
                            │  │   fixed_point = (next == state)      │ │
                            │  │   two_cycle   = (next == prev)       │ │
                            │  │   converged   = fixed_point|two_cycle│ │
                            │  └──────────────┬──────────────────────┘ │
                            └─────────────────┼─────────────────────────┘
                                              │
  uo_out[6:0] ◄─────────────────────────────── recalled segment pattern
  (7-segment display)
  uo_out[7]   ◄─────────────────── decimal point (high when converged)
```

### How weights become gates

The **weight matrix** is computed from stored patterns using the Hebbian rule:

```
W[i][j] = p1[i]×p1[j] + p9[i]×p9[j]     (bipolar: +1=ON, -1=OFF)
W[i][i] = 0                               (no self-connections)
```

Resulting weight matrix:

```
         a    b    c    d    e    f    g
    a  [ 0,   0,   0,  +2,   0,  +2,  +2 ]
    b  [ 0,   0,  +2,   0,  -2,   0,   0 ]
    c  [ 0,  +2,   0,   0,  -2,   0,   0 ]
    d  [ +2,  0,   0,   0,   0,  +2,  +2 ]
    e  [ 0,  -2,  -2,   0,   0,   0,   0 ]
    f  [ +2,  0,   0,  +2,   0,   0,  +2 ]
    g  [ +2,  0,   0,  +2,   0,  +2,   0 ]
```

Each row is then converted to a **combinational logic expression**:

| Neuron | Positive inputs | Negative inputs | Threshold | Gate expression |
|--------|----------------|-----------------|-----------|-----------------|
| a | d, f, g | — | ≥ 2 of 3 | `majority(d,f,g)` |
| b | c | e | c > e | `c & ~e` |
| c | b | e | b > e | `b & ~e` |
| d | a, f, g | — | ≥ 2 of 3 | `majority(a,f,g)` |
| e | — | b, c | neither | `~(b & c)` |
| f | a, d, g | — | ≥ 2 of 3 | `majority(a,d,g)` |
| g | a, d, f | — | ≥ 2 of 3 | `majority(a,d,f)` |

The weights are not stored in a register file — they are **hardwired into the
gate connections**. This is Computing-in-Memory (CiM): the memory and the
computation are the same physical structure.

### Gate count breakdown

| Block | Logic | GE |
|---|---|---|
| 4× majority gates (a,d,f,g) | 3 AND + 1 OR each | 28 |
| 2× AND-NOT gates (b,c) | AND2B cell each | 3 |
| 1× NAND gate (e) | NAND2 | 1 |
| 7× state flip-flops | DFF with reset | 42 |
| 7× prev flip-flops (2-cycle detection) | DFF | 42 |
| 2× 7-bit comparators (converge/2-cycle) | XOR+NOR tree | 20 |
| Mux / control | misc | 4 |
| **Total** | | **~140 GE** |

---

## How it works — step by step

### 1. Load phase (1 clock cycle)

```
User sets DIP switches → ui_in[6:0] = noisy segment pattern
User pulses uio_in[0] high for one clock cycle
→ state registers latch ui_in[6:0]
→ prev registers reset to 0
```

### 2. Iteration phase (1–3 clock cycles)

Each clock cycle, **all 7 neurons update simultaneously** based on the
current state of all other neurons:

```
Cycle 1:  state ← f(initial_input)
Cycle 2:  state ← f(f(initial_input))
...
Until:    f(state) == state   (fixed point)
   or:    f(state) == prev    (2-cycle → freeze)
```

The recurrent feedback path means the network is self-correcting —
noise in the input gets suppressed over iterations.

### 3. Converged phase

```
converged flag goes high
→ state registers freeze (no more updates)
→ uo_out[7] = 1 (decimal point lights up)
→ uo_out[6:0] shows the recalled digit
```

### Timing diagram

```
clk      ─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─
           └─┘ └─┘ └─┘ └─┘ └─┘ └─┘

start    ──┐ ┌─────────────────────
           └─┘  (1 cycle pulse)

state    ──[noisy]──[iter1]──[iter2]──[settled]──[settled]──
                                         ▲
                                    converged=1,
                                    dp lights up
```

---

## Basin of attraction

Not all noisy inputs successfully recall a stored pattern. Segments fall into
two groups based on their role in the weight matrix:

### Safe segments (a, d, f, g) — majority-vote protected

These are connected only through **positive** weights. Their update rule is a
majority vote of three other safe segments. One bit of corruption → corrected
in 1 cycle:

```
Digit 1 with seg-a flipped:   0x07 → corrected to 0x06 (digit 1) in 1 cycle ✓
Digit 9 with seg-d flipped:   0x67 → corrected to 0x6F (digit 9) in 1 cycle ✓
```

### Fragile segments (b, c, e) — mutually dependent triangle

These three neurons form a cycle of mutual dependencies (b↔c through e).
Corrupting any one of them breaks the triangle and may produce a **spurious
attractor** — a stable state that is neither digit 1 nor digit 9.

```
Digit 1 with seg-b flipped:   0x04 → spurious state  ✗
Digit 9 with seg-e flipped:   0x7F → spurious state  ✗
```

This is an intrinsic property of Hopfield networks — spurious attractors are
unavoidable mathematical side effects of the weight matrix.

### Energy landscape (conceptual)

```
Energy
  │
  │     ____         ____         ____
  │    /    \       /    \       /    \
  │   /      \     /      \     /      \
  │──/────────\───/────────\───/────────\──
  │          ▼             ▼
  │       digit 1       digit 9         ← stored memories (energy minima)
  │                  ▲
  │             spurious                ← unintended energy minimum
  │
  └─────────────────────────────────────► pattern space
```

The network always descends to the nearest energy minimum. Whether that is
a stored memory or a spurious attractor depends on the starting point.

---

## Pin mapping

| Pin | Direction | Description |
|-----|-----------|-------------|
| `ui_in[0]` | in | Segment **a** (top horizontal) |
| `ui_in[1]` | in | Segment **b** (top-right vertical) |
| `ui_in[2]` | in | Segment **c** (bottom-right vertical) |
| `ui_in[3]` | in | Segment **d** (bottom horizontal) |
| `ui_in[4]` | in | Segment **e** (bottom-left vertical) |
| `ui_in[5]` | in | Segment **f** (top-left vertical) |
| `ui_in[6]` | in | Segment **g** (middle horizontal) |
| `ui_in[7]` | — | Unused |
| `uio_in[0]` | in | **Start** — pulse high 1 cycle to begin recall |
| `uo_out[6:0]` | out | Recalled segment pattern (drives 7-segment display) |
| `uo_out[7]` | out | **Decimal point** — high when network has converged |

### 7-segment display layout

```
      a
   ┌─────┐
 f │     │ b
   ├──g──┤
 e │     │ c
   └─────┘
      d
```

---

## Test vectors

| Input hex | Input segments | Noise | Expected output | Cycles |
|-----------|---------------|-------|-----------------|--------|
| `0x06` | b, c | none (digit 1 exact) | **digit 1** | 1 |
| `0x6F` | a,b,c,d,f,g | none (digit 9 exact) | **digit 9** | 1 |
| `0x07` | a,b,c | digit 1 + seg-a | **digit 1** | 1 |
| `0x0E` | b,c,d | digit 1 + seg-d | **digit 1** | 1 |
| `0x26` | b,c,f | digit 1 + seg-f | **digit 1** | 1 |
| `0x46` | b,c,g | digit 1 + seg-g | **digit 1** | 1 |
| `0x6E` | b,c,d,f,g | digit 9 − seg-a | **digit 9** | 1 |
| `0x67` | a,b,c,f,g | digit 9 − seg-d | **digit 9** | 1 |
| `0x4F` | a,b,c,d,g | digit 9 − seg-f | **digit 9** | 1 |
| `0x2F` | a,b,c,d,f | digit 9 − seg-g | **digit 9** | 1 |

---

## Project structure

```
├── src/
│   ├── project.v          # tt_um_hopfield — TinyTapeout top wrapper
│   ├── hopfield_odd.v     # 7-node Hopfield network core (~140 GE)
│   └── config.json        # LibreLane / OpenLane synthesis config
├── test/
│   ├── tb.v               # cocotb Verilog wrapper
│   ├── test.py            # 3 cocotb test suites
│   ├── Makefile           # make → simulate with iverilog + cocotb
│   └── requirements.txt
├── docs/
│   └── info.md            # TinyTapeout datasheet
├── info.yaml              # TinyTapeout project metadata + pinout
└── README.md              # this file
```

---

## Simulate locally

```bash
# Install dependencies
pip install cocotb==1.9.2
sudo apt install iverilog       # or brew install icarus-verilog

# Run all tests
cd test && make

# Expected output:
#   test_exact_patterns   PASS
#   test_noisy_recall     PASS  (10/10 vectors)
#   test_convergence_speed PASS (all ≤ 4 cycles)
```

---

## Further reading

- [Hopfield, J.J. (1982) — Neural networks and physical systems with emergent collective computational abilities](https://www.pnas.org/doi/10.1073/pnas.79.8.2554)
- [GeeksForGeeks — Hopfield Neural Network](https://www.geeksforgeeks.org/machine-learning/hopfield-neural-network/)
- [TinyTapeout documentation](https://tinytapeout.com)
- [IHP SG13G2 PDK](https://github.com/IHP-GmbH/IHP-Open-PDK)
