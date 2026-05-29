# Option 1 – Threading simulation (MMLL / DSVM.py unchanged)

Runs federated BDSVM inside the real MMLL `MasterNode`/`WorkerNode` classes
but replaces the `pycloudmessenger` network layer with an in-process
`threading.Queue`-based message bus.

## Key design

| Component | Role |
|---|---|
| `TimedOutException` | Compatibility shim; module path set to match pycloudmessenger |
| `MockReceivedPacket` | Packet container with `.content` and `.notification` attributes |
| `ThreadedMessageBus` | One master queue + one queue per worker; all in memory |
| `MockComms` | Drop-in replacement for `pycloudmessenger` comms; plugged into MMLL nodes |
| `NullLogger` | Satisfies MMLL's logging interface without file I/O |

The MMLL `DSVM.py` model file is **not modified**. All changes live in the
test harness.

## How to run

```bash
# From the MMLL repo root (needs MMLL installed or on PYTHONPATH)
python bdsvm/option1_threading/test_bdsvm_fl_threading.py
```

Requirements: `numpy`, `scikit-learn`, `torch` (for MMLL), MMLL package.

## Expected output

```
Federated model accuracy : ~95.37%
Centralised baseline     : ~95.60%
```

The federated model matches the centralised baseline to within ~0.2 pp.
