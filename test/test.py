"""
test.py — cocotb testbench for tt_um_bnn (TinyTapeout BNN Breast Cancer Classifier)

Runs two test suites against the DUT:
  1. test_combinational — drives ui_in, checks uo_out[0] (y_comb) instantly
  2. test_sequential    — pulses uio_in[0] (start), waits for uo_out[2] (valid),
                          checks uo_out[1] (y_seq)

W = 8'b00000110,  THRESH = 3
ui_in[7:0] = binarized feature vector
uo_out[0]  = y_comb  (1=benign, 0=malignant)
uo_out[1]  = y_seq   (latched result)
uo_out[2]  = valid   (sequential valid strobe)
uio_in[0]  = start   (sequential engine trigger)
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


# ---------------------------------------------------------------------------
# Reference model — mirrors the Verilog gate-for-gate
# ---------------------------------------------------------------------------
W      = 0b00000110
THRESH = 3

def bnn_ref(x: int) -> int:
    """Software BNN: returns 1 (benign) or 0 (malignant)."""
    match    = (~(x ^ W)) & 0xFF
    popcount = bin(match).count('1')
    return 1 if popcount >= THRESH else 0


# ---------------------------------------------------------------------------
# Test vectors (derived from scripts/train_bnn.py test-set output)
# {x_byte: expected_y}
# ---------------------------------------------------------------------------
TEST_VECTORS = {
    0xFF: 0,   # malignant: all features above median → pop=2 (only x[1],x[2] match w=1)
    0x00: 1,   # benign:    all features below median → pop=6 (x[0,3,4,5,6,7] match w=0)
    0x7F: 1,   # benign:    x[7]=0 below median       → pop=3
    0xC0: 1,   # benign:    x[7:6]=1 above, rest low  → pop=4
    0x88: 1,   # benign:    mixed                     → pop=4
    0x06: 1,   # benign:    x = W exactly             → pop=8
    0xF9: 0,   # malignant: only x[1] matches         → pop=1
    0x3E: 0,   # malignant: borderline                → pop=2
}


# ---------------------------------------------------------------------------
# Helper: reset DUT
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


# ---------------------------------------------------------------------------
# Test 1: Combinational path
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_combinational(dut):
    """Test uo_out[0] (y_comb) — result available within same clock cycle."""
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    await reset_dut(dut)

    dut._log.info("=== Combinational BNN Tests (uo_out[0]) ===")
    passed = 0
    failed = 0

    for x_val, exp in TEST_VECTORS.items():
        dut.ui_in.value = x_val
        await Timer(2, units="ns")      # let combinational settle

        y_comb = int(dut.uo_out.value) & 0x1
        label  = "benign" if y_comb else "malignant"
        ok     = y_comb == exp

        dut._log.info(
            f"  x=0x{x_val:02X} ({x_val:08b})  y_comb={y_comb} ({label})  "
            f"exp={exp}  {'PASS' if ok else 'FAIL'}"
        )

        assert ok, (
            f"FAIL comb: x=0x{x_val:02X}  got y={y_comb}  expected y={exp}"
        )
        passed += 1

    dut._log.info(f"  Combinational: {passed} PASS  {failed} FAIL")


# ---------------------------------------------------------------------------
# Test 2: Sequential path
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_sequential(dut):
    """Test uo_out[1] (y_seq) — result valid after 8 clock cycles."""
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    await reset_dut(dut)

    dut._log.info("=== Sequential BNN Tests (uo_out[1], triggered via uio_in[0]) ===")
    passed = 0

    for x_val, exp in TEST_VECTORS.items():
        dut.ui_in.value  = x_val
        dut.uio_in.value = 0x00
        await RisingEdge(dut.clk)

        # Assert start for one cycle
        dut.uio_in.value = 0x01        # start = uio_in[0]
        await RisingEdge(dut.clk)
        dut.uio_in.value = 0x00

        # Wait for valid strobe (max 12 cycles)
        valid_seen = False
        for cycle in range(12):
            await RisingEdge(dut.clk)
            await Timer(1, units="ns")
            if (int(dut.uo_out.value) >> 2) & 1:
                valid_seen = True
                break

        assert valid_seen, f"TIMEOUT waiting for valid, x=0x{x_val:02X}"

        y_seq  = (int(dut.uo_out.value) >> 1) & 1
        label  = "benign" if y_seq else "malignant"
        ok     = y_seq == exp

        dut._log.info(
            f"  x=0x{x_val:02X}  y_seq={y_seq} ({label})  "
            f"exp={exp}  latency={cycle+1}cyc  {'PASS' if ok else 'FAIL'}"
        )

        assert ok, (
            f"FAIL seq: x=0x{x_val:02X}  got y={y_seq}  expected y={exp}"
        )
        passed += 1

    dut._log.info(f"  Sequential: {passed} PASS")


# ---------------------------------------------------------------------------
# Test 3: Combinational vs sequential agreement
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_comb_seq_agreement(dut):
    """Both outputs must agree on every test vector."""
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    await reset_dut(dut)

    dut._log.info("=== Comb vs Seq Agreement Test ===")

    for x_val in TEST_VECTORS:
        dut.ui_in.value  = x_val
        dut.uio_in.value = 0x01
        await RisingEdge(dut.clk)
        dut.uio_in.value = 0x00

        # Capture combinational immediately
        await Timer(2, units="ns")
        y_comb = int(dut.uo_out.value) & 1

        # Wait for sequential valid
        for _ in range(12):
            await RisingEdge(dut.clk)
            await Timer(1, units="ns")
            if (int(dut.uo_out.value) >> 2) & 1:
                break

        y_seq = (int(dut.uo_out.value) >> 1) & 1

        assert y_comb == y_seq, (
            f"MISMATCH x=0x{x_val:02X}: y_comb={y_comb} y_seq={y_seq}"
        )
        dut._log.info(f"  x=0x{x_val:02X}  y_comb={y_comb} y_seq={y_seq}  AGREE")
