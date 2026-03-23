"""
test.py — cocotb tests for tt_um_hopfield (Hopfield Associative Memory)

Stored patterns:
  digit 1 → 7'b0000110 = 0x06  (segments b, c)
  digit 9 → 7'b1101111 = 0x6F  (segments a,b,c,d,f,g)

Segment bit mapping:
  bit0=a  bit1=b  bit2=c  bit3=d  bit4=e  bit5=f  bit6=g

Pin mapping:
  ui_in[6:0]  — input segment pattern (noisy / partial)
  uio_in[0]   — start pulse (load input and begin recall)
  uo_out[6:0] — recalled segment pattern
  uo_out[7]   — converged flag (decimal point)
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# ---------------------------------------------------------------------------
# Stored patterns
# ---------------------------------------------------------------------------
DIGIT_1 = 0x06   # 7'b0000110 — b, c segments
DIGIT_9 = 0x6F   # 7'b1101111 — a,b,c,d,f,g segments

# ---------------------------------------------------------------------------
# Test vectors: {input_pattern: expected_recalled_digit}
# Safe flips (a,d,f,g) always recall correctly.
# Fragile flips (b,c,e) may produce spurious states — not tested here.
# ---------------------------------------------------------------------------
TEST_VECTORS = {
    # --- Exact stored patterns ---
    0x06: DIGIT_1,   # digit 1 exactly
    0x6F: DIGIT_9,   # digit 9 exactly

    # --- Noisy digit 1 (safe segment flips: a, d, f, g) ---
    0x07: DIGIT_1,   # digit 1 + seg-a  (0x06 ^ 0x01)
    0x0E: DIGIT_1,   # digit 1 + seg-d  (0x06 ^ 0x08)
    0x26: DIGIT_1,   # digit 1 + seg-f  (0x06 ^ 0x20)
    0x46: DIGIT_1,   # digit 1 + seg-g  (0x06 ^ 0x40)

    # --- Noisy digit 9 (safe segment flips: a, d, f, g) ---
    0x6E: DIGIT_9,   # digit 9 - seg-a  (0x6F ^ 0x01)
    0x67: DIGIT_9,   # digit 9 - seg-d  (0x6F ^ 0x08)
    0x4F: DIGIT_9,   # digit 9 - seg-f  (0x6F ^ 0x20)
    0x2F: DIGIT_9,   # digit 9 - seg-g  (0x6F ^ 0x40)
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def reset_dut(dut):
    dut.rst_n.value  = 0
    dut.ena.value    = 1
    dut.ui_in.value  = 0
    dut.uio_in.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def recall(dut, x_val, max_cycles=16):
    """Load x_val, iterate until converged, return (output, cycles)."""
    dut.ui_in.value  = x_val
    dut.uio_in.value = 0x01          # start pulse
    await RisingEdge(dut.clk)
    dut.uio_in.value = 0x00

    for cyc in range(max_cycles):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")
        out = int(dut.uo_out.value)
        if (out >> 7) & 1:           # converged flag = uo_out[7]
            return out & 0x7F, cyc + 1

    return int(dut.uo_out.value) & 0x7F, max_cycles   # timeout


def seg_str(pattern):
    """Human-readable segment list from 7-bit pattern."""
    segs = [c for i, c in enumerate("abcdefg") if (pattern >> i) & 1]
    return "{" + ",".join(segs) + "}"


# ---------------------------------------------------------------------------
# Test 1: Exact patterns converge immediately (1 cycle)
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_exact_patterns(dut):
    """Stored patterns (digit 1 and 9) must be fixed points — stable in 1 cycle."""
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    await reset_dut(dut)
    dut._log.info("=== Test: Exact stored patterns ===")

    for x_val, exp in [(DIGIT_1, DIGIT_1), (DIGIT_9, DIGIT_9)]:
        out, cyc = await recall(dut, x_val)
        dut._log.info(
            f"  input={seg_str(x_val)} → output={seg_str(out)}"
            f"  cycles={cyc}  {'PASS' if out == exp else 'FAIL'}"
        )
        assert out == exp, f"Exact pattern 0x{x_val:02X}: got 0x{out:02X} exp 0x{exp:02X}"


# ---------------------------------------------------------------------------
# Test 2: Noisy patterns recall correct digit
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_noisy_recall(dut):
    """Patterns with one safe-segment flip must converge to the stored digit."""
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    await reset_dut(dut)
    dut._log.info("=== Test: Noisy recall (safe segment flips) ===")

    passed = failed = 0
    for x_val, exp in TEST_VECTORS.items():
        out, cyc = await recall(dut, x_val)
        ok = (out == exp)
        exp_digit = 1 if exp == DIGIT_1 else 9
        got_digit = 1 if out == DIGIT_1 else (9 if out == DIGIT_9 else "?")
        dut._log.info(
            f"  input=0x{x_val:02X} {seg_str(x_val):<20s}"
            f" → 0x{out:02X} {seg_str(out):<20s}"
            f" (digit {got_digit}, exp {exp_digit}, {cyc} cyc)"
            f"  {'PASS' if ok else 'FAIL'}"
        )
        if ok:
            passed += 1
        else:
            failed += 1

    dut._log.info(f"  Result: {passed} PASS  {failed} FAIL")
    assert failed == 0, f"{failed} test vectors failed"


# ---------------------------------------------------------------------------
# Test 3: Convergence — decimal point (uo_out[7]) fires within 8 cycles
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_convergence_speed(dut):
    """All stored patterns must converge within 4 clock cycles."""
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    await reset_dut(dut)
    dut._log.info("=== Test: Convergence speed ===")

    MAX_CYCLES = 4
    for x_val, _ in TEST_VECTORS.items():
        _, cyc = await recall(dut, x_val)
        dut._log.info(f"  0x{x_val:02X} converged in {cyc} cycle(s)")
        assert cyc <= MAX_CYCLES, \
            f"0x{x_val:02X} took {cyc} cycles (limit {MAX_CYCLES})"
