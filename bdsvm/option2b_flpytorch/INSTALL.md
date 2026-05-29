# INSTALL – Option 2b (FL_PyTorch integration)

## 1. Clone FL_PyTorch

```bash
git clone https://github.com/burlachenkok/flpytorch.git
cd flpytorch
```

## 2. Install dependencies

```bash
pip install torch torchvision numpy scikit-learn
pip install -e .
```

> **Note**: The system Python (not a venv with torch missing) must be used.
> Verified with: Python 3.12, torch 2.10.0+cpu.

## 3. Apply BDSVM patches

Copy the four patch files into the FL_PyTorch source tree:

```bash
# From your repo root (adjust paths as needed)
FLPT=path/to/flpytorch/fl_pytorch

cp bdsvm/option2b_flpytorch/mnist_bdsvm_dataset.py  $FLPT/data_preprocess/fl_datasets/
cp bdsvm/option2b_flpytorch/algorithms_bdsvm.py      $FLPT/utils/
cp bdsvm/option2b_flpytorch/model_funcs_bdsvm.py     $FLPT/utils/
```

Then add three lines to the existing framework files:

### `fl_pytorch/data_preprocess/fl_datasets/__init__.py`
```python
from .mnist_bdsvm_dataset import MNISTBDSVMDataset
```

### `fl_pytorch/data_preprocess/data_loader.py`
In `load_data()`:
```python
elif dataset == 'mnist_bdsvm':
    return MNISTBDSVMDataset(args)
```
In `get_num_classes()`:
```python
elif dataset == 'mnist_bdsvm':
    num_classes = 2
```
In `get_num_clients()`:
```python
elif dataset == 'mnist_bdsvm':
    num_clients = 3
```

### `fl_pytorch/utils/algorithms.py`
Import and register:
```python
from fl_pytorch.utils.algorithms_bdsvm import BDSVMAlgorithm
# in getImplClassForAlgo():
elif algorithm == "bdsvm":
    classImpl = BDSVMAlgorithm
# in getAlgorithmsList():  add "bdsvm"
```

### `fl_pytorch/utils/model_funcs.py`
Import and register:
```python
from fl_pytorch.utils.model_funcs_bdsvm import BDSVMModule
# in initialise_model(): add the  if model_name == "bdsvm": branch
```

### `fl_pytorch/opts.py`
Add `"mnist_bdsvm"` to the `--dataset` choices list.

## 4. Run

```bash
python bdsvm/option2b_flpytorch/run_bdsvm_framework.py
```

Checkpoints are written to `checkpoints/bdsvm_mnist_bdsvm/`.
Delete this directory to re-run from scratch.
