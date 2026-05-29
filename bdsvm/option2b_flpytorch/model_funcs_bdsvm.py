"""BDSVMModule: PyTorch module for Budget SVM.

Drop-in patch for fl_pytorch/utils/model_funcs.py.
Register with:
    if model_name == "bdsvm":
        P_plus_1 = dataset_ref.num_params
        model = BDSVMModule(P_plus_1)
        model.k_tilde_cc = torch.tensor(dataset_ref.k_tilde_cc, dtype=torch.float64)
"""
import torch
from torch import nn


class BDSVMModule(nn.Module):
    """Minimal module parameterised by SVM weight vector β.

    forward(k_tilde) → (batch, 1)

    The (batch, 1) shape keeps output.shape[1] == 1, which avoids the
    top-5 accuracy branch inside FL_PyTorch's update_metrics() for
    single-output (binary) models.
    """

    def __init__(self, num_params: int):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(num_params, dtype=torch.float64))
        # Required by FL framework's train_model()
        self.do_not_use_bn_and_dropout = False

    def forward(self, k_tilde):
        # k_tilde: (batch, P+1),  beta: (P+1,)  →  output: (batch, 1)
        return (k_tilde @ self.beta).unsqueeze(-1)
