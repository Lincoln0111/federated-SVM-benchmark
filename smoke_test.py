from config import CONFIG
import run_benchmark as rb

print("=== Team alignment ===")
assert CONFIG["NUM_WORKERS"] == 10
assert CONFIG["ROUNDS"]      == 100
print("NUM_WORKERS =", CONFIG["NUM_WORKERS"], " OK")
print("ROUNDS      =", CONFIG["ROUNDS"],      " OK")

print("\n=== New execution/communication keys ===")
assert CONFIG["COMMUNICATION_BACKEND"] == "local_simulation"
assert CONFIG["EXECUTION_MODE"]        == "python_local_simulation"
assert CONFIG["FUTURE_TARGET_ENV"]     == "CCR_HPC"
print("COMMUNICATION_BACKEND =", CONFIG["COMMUNICATION_BACKEND"], " OK")
print("EXECUTION_MODE        =", CONFIG["EXECUTION_MODE"], " OK")
print("FUTURE_TARGET_ENV     =", CONFIG["FUTURE_TARGET_ENV"], " OK")

print("\n=== Reference values absent ===")
assert "LAMBDA"      not in CONFIG
assert "T0_FRACTION" not in CONFIG
print("LAMBDA absent       OK")
print("T0_FRACTION absent  OK")

print("\n=== Algorithmic params unchanged ===")
checks = {
    "FEDAVG_LAMBDA":  0.01,
    "FDR_LAMBDA":     0.01,
    "BDSVM_C":        1.0,
    "BDSVM_P":        100,
    "MOMENTUM_ALPHA": 0.5,
    "IRWLS_ETA":      5e-3,
    "FDR_RHO":        1.0,
    "FDR_EPS_SCALE":  1.0,
    "FEDSSL_D_ENC":   64,
    "FEDSSL_SVM_C":   1.0,
    "CENTRALIZED_C":  1.0,
}
for k, expected in checks.items():
    assert CONFIG[k] == expected, f"{k}: got {CONFIG[k]}, expected {expected}"
    print(" ", k, "=", CONFIG[k], " OK")

print("\n=== METHOD_KWARGS ===")
for name, kw in rb.METHOD_KWARGS.items():
    print(" ", name, ":", kw)
assert rb.METHOD_KWARGS["bdsvm"]["C"]               == CONFIG["BDSVM_C"]
assert rb.METHOD_KWARGS["bdsvm"]["lam"]             == CONFIG["MOMENTUM_ALPHA"]
assert rb.METHOD_KWARGS["bdsvm"]["eta"]             == CONFIG["IRWLS_ETA"]
assert rb.METHOD_KWARGS["fdr_svm"]["rho"]           == CONFIG["FDR_RHO"]
assert rb.METHOD_KWARGS["fdr_svm"]["lambda_reg"]    == CONFIG["FDR_LAMBDA"]
assert rb.METHOD_KWARGS["centralized"]["C"]         == CONFIG["CENTRALIZED_C"]
assert rb.METHOD_KWARGS["fedavg_svm"]["lambda_reg"] == CONFIG["FEDAVG_LAMBDA"]
assert rb.METHOD_KWARGS["fedssl_amc"]["d_enc"]      == CONFIG["FEDSSL_D_ENC"]

print("\nAll assertions passed.")
