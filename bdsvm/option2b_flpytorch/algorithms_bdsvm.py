"""BDSVMAlgorithm: FL_PyTorch algorithm class for Budget Distributed SVM.

Drop-in patch for fl_pytorch/utils/algorithms.py.
Register with:
    elif algorithm == "bdsvm":
        classImpl = BDSVMAlgorithm
and add "bdsvm" to getAlgorithmsList().
"""
import numpy as np
import torch
from models import mutils


class BDSVMAlgorithm:
    """Federated BDSVM via primal IRWLS.

    Local step: accumulate (C_m, d_m) sufficient statistics.
    Server step: solve (sum(C_m) + K_tilde_cc) @ beta = sum(d_m),
                 then blend with momentum (lambda=0.5).
    """

    LAMBDA = 0.5   # momentum blend factor

    @staticmethod
    def initializeServerState(args, model, D, total_clients, grad_start):
        k_tilde_cc = model.k_tilde_cc.detach().cpu().numpy().astype(np.float64)
        C = float(getattr(args, "bdsvm_C", 1.0))
        return {"bdsvm_k_tilde_cc": k_tilde_cc, "bdsvm_C": C}

    @staticmethod
    def clientState(H, clientId, client_data_samples, device):
        return {}

    @staticmethod
    def localGradientEvaluation(
        client_state, model, dataloader, criterion, is_rnn, local_iteration_number
    ):
        beta = mutils.get_params(model).detach().cpu().numpy().astype(np.float64)
        C = client_state["H"]["bdsvm_C"]
        P_plus_1 = len(beta)
        C_m = np.zeros((P_plus_1, P_plus_1))
        d_m = np.zeros(P_plus_1)

        for k_tilde_batch, y_batch in dataloader:
            k_tilde = k_tilde_batch.cpu().numpy().astype(np.float64)
            y = y_batch.cpu().numpy().reshape(-1).astype(np.float64)

            e = y - k_tilde @ beta   # e_i y_i > 0 at beta=0 (correct sign)
            ey = e * y
            # IRWLS weights: a_i = 2C / max(ey_i, eps)  if ey_i >= 0 else 0
            a = np.where(ey < 0, 0.0, 2.0 * C / np.maximum(ey, 1e-10))

            Ka = k_tilde.T * a        # (P+1, batch)
            C_m += Ka @ k_tilde       # (P+1, P+1)
            d_m += Ka @ y             # (P+1,)

        client_state["bdsvm_cm"] = C_m
        client_state["bdsvm_dm"] = d_m

        # Return a dummy zero gradient — actual update is in serverGradient
        return 0.0, torch.zeros(P_plus_1, dtype=model.fl_dtype)

    @staticmethod
    def serverGradient(clients_responses, clients, model, params_current, H):
        k_tilde_cc = H["bdsvm_k_tilde_cc"]
        P_plus_1 = len(params_current)
        C_total = np.zeros((P_plus_1, P_plus_1))
        d_total = np.zeros(P_plus_1)

        for i in range(clients):
            clients_responses.waitForItem()
            response = clients_responses.get(i)
            C_total += response["client_state"]["bdsvm_cm"]
            d_total += response["client_state"]["bdsvm_dm"]

        lhs = C_total + k_tilde_cc
        beta_new, _, _, _ = np.linalg.lstsq(lhs, d_total, rcond=None)

        # Momentum blend: beta = (1-lam)*beta_old + lam*beta_new
        beta_old = params_current.detach().cpu().numpy().astype(np.float64)
        lam = BDSVMAlgorithm.LAMBDA
        beta_momentum = (1.0 - lam) * beta_old + lam * beta_new

        beta_t = torch.tensor(
            beta_momentum, dtype=params_current.dtype, device=params_current.device
        )
        # Framework applies:  params_new = params_current - global_lr * grad
        # With global_lr=1.0 we need:  grad = params_current - beta_momentum
        return params_current - beta_t

    @staticmethod
    def serverGlobalStateUpdate(
        clients_responses, clients, model, paramsPrev, grad_server, H
    ):
        return H
