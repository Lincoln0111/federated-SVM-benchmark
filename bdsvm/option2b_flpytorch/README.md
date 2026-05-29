# Option 2b – FL_PyTorch framework integration

Runs BDSVM inside the real [FL_PyTorch](https://github.com/burlachenkok/flpytorch)
federated learning framework using its simulation loop (no sockets needed).

See [INSTALL.md](INSTALL.md) for environment setup.

## Files

| File | Purpose |
|---|---|
| `mnist_bdsvm_dataset.py` | `MNISTBDSVMDataset` — patch for `fl_pytorch/data_preprocess/fl_datasets/` |
| `algorithms_bdsvm.py` | `BDSVMAlgorithm` — patch for `fl_pytorch/utils/algorithms.py` |
| `model_funcs_bdsvm.py` | `BDSVMModule` — patch for `fl_pytorch/utils/model_funcs.py` |
| `run_bdsvm_framework.py` | Entry point — calls `fl_pytorch.main()` with BDSVM args |

## Design notes

- `BDSVMModule` is a standard `nn.Module` whose single `nn.Parameter` is the
  SVM weight vector β.  The framework's optimiser is set to `global-lr=1.0,
  local-lr=0.0` (server side full update, no local gradient descent).
- `BDSVMAlgorithm.localGradientEvaluation` accumulates (C_m, d_m) and returns
  a zero dummy gradient — the real update happens in `serverGradient`.
- `serverGradient` solves `(ΣC_m + K̃_cc) β = Σd_m` and returns
  `params_current − β_momentum` so the SGD step `params − lr·grad` yields β_momentum.
- Compute type is forced to `fp64` throughout to match NumPy precision.

## Expected output

```
Round 10  loss: ~0.150  accuracy: ~94.84%
```

## How to run (after install)

```bash
python bdsvm/option2b_flpytorch/run_bdsvm_framework.py
```
