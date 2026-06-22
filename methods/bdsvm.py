"""BDSVM: Budget Distributed SVM (ACM TIST 2022, DOI 10.1145/3539734).

Strict implementation of Algorithms 2 (aggregator) and 3 (worker).

Model: f(x) = Σ_j β_j k(x, p_j) + b
where {p_j}_{j=1}^P are P random pre-image vectors generated from a
shared seed S (never a training sample — preserves data privacy).

For linear kernel k(x,p) = x^T p, this is equivalent to a linear SVM
parameterised through the random projection basis {p_j}.

Each worker computes local IRWLS matrices and sends them to the aggregator,
which solves a global (P+1)×(P+1) linear system. Communication per round
is O(P²) not O(d), which is the communication advantage of the budget.

References: Algorithms 2 & 3 in the paper, Section 2.1–2.2.
"""
import numpy as np


def _kernel_matrix(X, P_mat):
    """K̃_m = [X @ P_mat^T | 1]  shape (N, P+1).

    For linear kernel k(x, p) = x^T p.
    Works for sparse X (scipy CSR) and dense numpy arrays.
    """
    Km = X @ P_mat.T          # (N, P)  sparse/dense → dense
    if hasattr(Km, "toarray"):
        Km = Km.toarray()
    else:
        Km = np.asarray(Km)
    return np.hstack([Km, np.ones((Km.shape[0], 1))])   # (N, P+1)


def run(X_train, y_train, X_test, y_test, client_idx,
        P=100, C=1.0, lam=0.5, n_rounds=50, eta=5e-3, seed=0):
    """
    P     : budget — number of random pre-image vectors
    C     : SVM regularisation (hinge loss weight)
    lam   : momentum λ∈[0,1] for model update smoothing (0 = no momentum)
    n_rounds: max FL communication rounds (paper uses convergence criterion)
    eta   : convergence threshold η; stop when ‖β_new−β‖/‖β‖ < η
    seed  : random seed S shared by aggregator and all workers
    """
    rng = np.random.default_rng(seed)
    d = X_train.shape[1]

    # ── Aggregator: Algorithm 2, step 2 ──────────────────────────────────
    # Generate P random pre-image vectors p_j ∈ R^d using shared seed S.
    # Scale so ‖p_j‖ ≈ 1 across different feature dimensionalities.
    P_mat = rng.standard_normal((P, d)) / np.sqrt(d)   # (P, d)

    # ── Aggregator: Algorithm 2, step 3 ──────────────────────────────────
    # Pre-image kernel matrix K̃_p = [[P_mat @ P_mat^T, 0], [0^T, 0]]
    K_tilde_p = np.zeros((P + 1, P + 1))
    K_tilde_p[:P, :P] = P_mat @ P_mat.T                # linear kernel

    # ── Workers: Algorithm 3, step 3 ─────────────────────────────────────
    # Each worker precomputes and stores its kernel matrix K̃_m = [K_m | 1].
    K_tildes = []
    for ci in client_idx:
        if len(ci) == 0:
            K_tildes.append(None)
        else:
            K_tildes.append(_kernel_matrix(X_train[ci], P_mat))

    # ── Algorithm 2, step 1 ───────────────────────────────────────────────
    beta = np.zeros(P + 1)    # β̃^(0) = 0

    for epoch in range(n_rounds):
        C_sum = np.zeros((P + 1, P + 1))
        d_sum = np.zeros(P + 1)

        for k, ci in enumerate(client_idx):
            if len(ci) == 0 or K_tildes[k] is None:
                continue
            Km = K_tildes[k]          # (N_k, P+1)
            y_k = y_train[ci]         # (N_k,)

            # Algorithm 3, step 5: errors  e_m = y_m − K̃_m β̃^(n)
            e = y_k - Km @ beta

            # Algorithm 3, step 6: IRWLS weights
            #   a_i = 0              if  e_i y_i < 0
            #   a_i = 2C / (e_i y_i) if  e_i y_i ≥ 0
            margin = e * y_k          # = 1 − y_i f(x_i)
            a = np.where(margin < 0, 0.0,
                         2.0 * C / np.maximum(margin, 1e-10))
            a = np.minimum(a, 1e6)    # numerical stability

            # Algorithm 3, step 7: C_m = K̃_m^T D_a K̃_m,  d_m = K̃_m^T D_a y_m
            aKm = Km * a[:, None]     # (N_k, P+1)  avoids forming D_a
            C_sum += aKm.T @ Km       # (P+1, P+1)
            d_sum += Km.T @ (a * y_k) # (P+1,)

        # Algorithm 2, step 7: β̃_new = (Σ C_m + K̃_p)^{-1} Σ d_m
        beta_new = np.linalg.solve(C_sum + K_tilde_p, d_sum)

        # Algorithm 2, step 8: smooth update β̃^(n+1) = λ β̃^(n) + (1−λ) β̃_new
        beta_next = lam * beta + (1.0 - lam) * beta_new

        # Algorithm 2, step 9: convergence criterion ‖β_new−β‖/‖β‖ < η
        norm_beta = np.linalg.norm(beta)
        if norm_beta > 0 and np.linalg.norm(beta_next - beta) / norm_beta < eta:
            beta = beta_next
            break
        beta = beta_next

    # ── Prediction ────────────────────────────────────────────────────────
    K_tilde_test = _kernel_matrix(X_test, P_mat)
    scores = K_tilde_test @ beta
    preds = np.sign(scores)
    preds[preds == 0] = 1.0
    acc = float((preds == y_test).mean())
    return {"test_acc": acc, "beta": beta, "epochs": epoch + 1}