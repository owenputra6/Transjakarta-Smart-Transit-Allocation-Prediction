"""
run_small.py — Inference Bus Kecil TransJakarta
"""
import sys
from pathlib import Path
import importlib.util
import pandas as pd
import torch

# ── PATHS ─────────────────────────────────────────────────────────────────────
CSV_PATH       = Path("./dummy_test/predict_bus_kecil.csv")
ARTIFACTS_DIR  = Path("./training_and_inferences/model_allocation")
OUTPUT_PATH    = Path("./hasil_small.csv")
INFERENCE_PATH = Path("./inference.py")
# ──────────────────────────────────────────────────────────────────────────────

spec = importlib.util.spec_from_file_location("inference", INFERENCE_PATH)
inf  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inf)
inf.ARTIFACTS_DIR = ARTIFACTS_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

df = inf.run_pipeline(str(CSV_PATH), "small", None, device)
df["bus_type"] = "small"

# ── Output: jumlah bus per jam per corridor ───────────────────────────────────
summary = (
    df[["corridorID", "hour", "alokasi_hari_ini", "overload_prob", "overload_pred"]]
    .sort_values(["corridorID", "hour"])
    .reset_index(drop=True)
)

print("\n" + "="*70)
print("  HASIL ALOKASI BUS KECIL — per Corridor per Jam")
print("="*70)
print(summary.to_string(index=False))

summary.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅  Disimpan ke: {OUTPUT_PATH}  ({len(summary):,} rows)")
