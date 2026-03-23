# Testing

## Requirements

```bash
pip install -r requirements.txt   # installs cocotb
brew install icarus-verilog        # simulation backend
```

## Run

```bash
cd test
make
```

## What is tested

| Test | Description |
|------|-------------|
| `test_combinational` | Drives `ui_in`, checks `uo_out[0]` (y_comb) settles within 2 ns |
| `test_sequential` | Pulses `uio_in[0]` (start), waits for `uo_out[2]` (valid), checks `uo_out[1]` (y_seq) |
| `test_comb_seq_agreement` | Both engines must produce identical results on every test vector |

## Pin reference

| Pin | Direction | Description |
|-----|-----------|-------------|
| `ui_in[7:0]` | in | Binarized feature vector |
| `uo_out[0]` | out | Combinational result (1=benign) |
| `uo_out[1]` | out | Sequential result (latched) |
| `uo_out[2]` | out | Sequential valid strobe |
| `uio_in[0]` | in | Start pulse for sequential engine |
