"""
inference.py — TransJakarta Bus Allocation (LSTM)
==================================================
Prediksi kebutuhan bus hari ini per (corridorID, jam) berdasarkan
7 hari data historis tap-in/tap-out.

Usage:
    # Predict today (auto-detect date from CSV)
    python inference.py --big bus_besar_all.csv --small bus_kecil_all.csv

    # Predict specific date (must exist in CSV)
    python inference.py --big bus_besar_all.csv --small bus_kecil_all.csv --date 2024-06-30

    # Only predict big bus
    python inference.py --big bus_besar_all.csv

    # Save output to CSV
    python inference.py --big bus_besar_all.csv --small bus_kecil_all.csv --out hasil_alokasi.csv

Artifacts yang dibutuhkan (dari model_artifacts/):
    model_big.pt        model_small.pt
    scaler_big.pkl      scaler_small.pkl
    thresh_big.txt      thresh_small.txt   (opsional — fallback ke 0.35)
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ── Constants (harus sama dengan training) ────────────────────────────────────
LOOKBACK_DAYS   = 7
SENIOR_CUTOFF   = 1980
PRIORITY_THRESH = 6       # bus besar: overload jika peak_senior_active > ini
SMALL_CAPACITY  = 15      # bus kecil: overload jika peak_active > ini
DEFAULT_THRESHOLD = 0.35
FEATURE_COLS    = [
    "overload", "peak_active", "peak_senior_active",
    "total_tapin", "day_of_week", "is_weekend",
]

ARTIFACTS_DIR = Path("model_artifacts")

# ── Corridor sets ─────────────────────────────────────────────────────────────
_BIG_BUS = {
    "1","2","3","4","5","6","7","8","9","10","11","12","13","14",
    "1A","1B","1C","1E","1F","1H","1K","1M","1N","1P","1Q","1R","1T",
    "2A","2B","2E","2F","2H","2P","2Q",
    "3A","3B","3C","3E","3F","3H",
    "4B","4C","4D","4E","4F",
    "5B","5C","5D","5F","5M","5N",
    "6A","6B","6C","6D","6H","6M","6N","6P","6Q","6T","6U","6V",
    "7A","7B","7C","7D","7E","7F","7P","7Q",
    "8A","8C","8D","8E","8K","8M",
    "9A","9C","9D","9E","9F","9H","9N",
    "10A","10B","10D","10H","10K",
    "11B","11C","11D","11K","11M","11N","11P","11Q",
    "12A","12B","12C","12F","12H","12P",
    "13B","13C","13D",
    "B11","B13","B14","B21","D11","D21","D31","D32",
    "S11","S12","S21","S22","S31","T11","T21",
    "R1A","BW9","JIS3","L13E",
}

_MIKROTRANS = {
    "JAK.01","JAK.02","JAK.03","JAK.04","JAK.05","JAK.06","JAK.07","JAK.08",
    "JAK.10","JAK.11","JAK.12","JAK.13","JAK.14","JAK.15","JAK.16","JAK.17",
    "JAK.18","JAK.19","JAK.20","JAK.21","JAK.22","JAK.23","JAK.24","JAK.25",
    "JAK.26","JAK.27","JAK.28","JAK.29","JAK.30","JAK.31","JAK.32","JAK.33",
    "JAK.34","JAK.35","JAK.36","JAK.37","JAK.38","JAK.39","JAK.40","JAK.41",
    "JAK.42","JAK.43B","JAK.43C","JAK.44","JAK.45","JAK.46","JAK.47",
    "JAK.48A","JAK.48B","JAK.49","JAK.50","JAK.51","JAK.52","JAK.53",
    "JAK.54","JAK.56","JAK.58","JAK.59","JAK.60","JAK.61","JAK.64",
    "JAK.71","JAK.72","JAK.73","JAK.74","JAK.75","JAK.77","JAK.80",
    "JAK.84","JAK.85","JAK.86","JAK.88","JAK.99",
    "JAK.106","JAK.110A","JAK.112","JAK.113","JAK.115","JAK.117","JAK.118","JAK.120",
}
_MINITRANS = {
    "M1","M1H","M2","M3","M4","M5","M6","M7","M7B",
    "M8","M9","M10","M11","M12","M13",
}
ALL_SMALL = _MIKROTRANS | _MINITRANS

BIG_BUS_ALLOCATION = {
    "1": 5, "2": 5, "3": 5, "4": 4, "5": 4, "6": 4, "7": 4, "8": 4,
    "9": 3, "10": 4, "11": 4, "12": 4, "13": 4, "14": 3,
    "1A": 2, "1B": 2, "1C": 3, "1E": 2, "1F": 2, "1H": 2, "1K": 2,
    "1M": 2, "1N": 2, "1P": 3, "1Q": 2, "1R": 2, "1T": 3,
    "2A": 2, "2B": 2, "2E": 2, "2F": 2, "2H": 2, "2P": 2, "2Q": 2,
    "3A": 2, "3B": 2, "3C": 2, "3E": 2, "3F": 2, "3H": 2,
    "4B": 2, "4C": 2, "4D": 2, "4E": 2, "4F": 2,
    "5B": 2, "5C": 2, "5D": 2, "5F": 2, "5M": 3, "5N": 2,
    "6A": 2, "6B": 2, "6C": 3, "6D": 2, "6H": 3, "6M": 2,
    "6N": 2, "6P": 2, "6Q": 2, "6T": 2, "6U": 2, "6V": 2,
    "7A": 2, "7B": 2, "7C": 2, "7D": 3, "7E": 2, "7F": 2, "7P": 2, "7Q": 2,
    "8A": 2, "8C": 2, "8D": 2, "8E": 2, "8K": 2, "8M": 2,
    "9A": 2, "9C": 2, "9D": 2, "9E": 2, "9F": 2, "9H": 2, "9N": 2,
    "10A": 2, "10B": 2, "10D": 3, "10H": 2, "10K": 2,
    "11B": 2, "11C": 2, "11D": 2, "11K": 2, "11M": 2, "11N": 2, "11P": 2, "11Q": 2,
    "12A": 2, "12B": 2, "12C": 2, "12F": 2, "12H": 2, "12P": 2,
    "13B": 2, "13C": 2, "13D": 2,
    "B11": 3, "B13": 3, "B14": 3, "B21": 3,
    "D11": 3, "D21": 3, "D31": 3, "D32": 3,
    "S11": 3, "S12": 3, "S21": 3, "S22": 3, "S31": 3,
    "T11": 3, "T21": 3, "R1A": 3, "BW9": 2, "JIS3": 2, "L13E": 2,
}
SMALL_BUS_ALLOCATION = {
    **{c: 2 for c in _MIKROTRANS},
    "M1": 3, "M1H": 2, "M2": 3, "M3": 2, "M4": 2,
    "M5": 2, "M6": 2, "M7": 3, "M7B": 2, "M8": 2,
    "M9": 2, "M10": 2, "M11": 2, "M12": 2, "M13": 2,
}


# ── Model definition (harus identik dengan training) ─────────────────────────
class BusLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


# ── Artifact loader ───────────────────────────────────────────────────────────
def load_artifacts(bus_type: str, device: torch.device) -> tuple:
    """
    Returns (model, scaler, threshold).
    Raises FileNotFoundError with a helpful message if artifacts are missing.
    """
    prefix = ARTIFACTS_DIR / bus_type  # e.g. model_artifacts/big

    model_path  = ARTIFACTS_DIR / f"model_{bus_type}.pt"
    scaler_path = ARTIFACTS_DIR / f"scaler_{bus_type}.pkl"
    thresh_path = ARTIFACTS_DIR / f"thresh_{bus_type}.txt"

    missing = [p for p in (model_path, scaler_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Artifact tidak ditemukan: {missing}\n"
            f"Jalankan notebook dulu untuk generate model_artifacts/."
        )

    # Load scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Load model
    model = BusLSTM(n_features=len(FEATURE_COLS)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load threshold (optional — fallback to default)
    if thresh_path.exists():
        threshold = float(thresh_path.read_text().strip())
    else:
        threshold = DEFAULT_THRESHOLD
        print(f"  [warn] {thresh_path} tidak ada, pakai default threshold={threshold}")

    print(
        f"  ✓ Loaded {bus_type.upper()} model  |  "
        f"threshold={threshold:.2f}  |  device={device}"
    )
    return model, scaler, threshold


# ── Feature engineering ───────────────────────────────────────────────────────
def _compute_hourly_peaks(df: pd.DataFrame, bus_type: str) -> pd.DataFrame:
    """
    Dari raw tap CSV → hourly (corridorID, date, hour) aggregation.
    Identik dengan load_and_aggregate() di notebook, tanpa print/tqdm.
    """
    corridor_set = _BIG_BUS if bus_type == "big" else ALL_SMALL

    df = df.copy()
    drop_cols = [c for c in df.columns if "Lat" in c or "Lon" in c]
    df = df.drop(columns=drop_cols + ["transID"], errors="ignore")

    df["tapInTime"]        = pd.to_datetime(df["tapInTime"],        errors="coerce")
    df["tapOutTime"]       = pd.to_datetime(df["tapOutTime"],       errors="coerce")
    df["payCardBirthDate"] = pd.to_numeric(df["payCardBirthDate"],  errors="coerce")

    df = df.dropna(subset=["tapInTime", "tapOutTime", "corridorID"])
    df = df[df["tapOutTime"] >= df["tapInTime"]].copy()
    df = df[df["corridorID"].isin(corridor_set)].copy()

    df["date"]      = df["tapInTime"].dt.date
    df["hour"]      = df["tapInTime"].dt.hour
    df["is_senior"] = (df["payCardBirthDate"] <= SENIOR_CUTOFF).astype(int)

    records = []
    for (corridor, day, hour), grp in df.groupby(["corridorID", "date", "hour"]):
        ev_in              = grp[["tapInTime", "is_senior"]].rename(columns={"tapInTime": "event_time"}).copy()
        ev_in["d_all"]     =  1
        ev_in["d_senior"]  =  ev_in["is_senior"]

        same_out = grp[grp["tapOutTime"].dt.hour == hour][["tapOutTime", "is_senior"]].rename(columns={"tapOutTime": "event_time"}).copy()
        same_out["d_all"]    = -1
        same_out["d_senior"] = -same_out["is_senior"]

        events = (
            pd.concat([ev_in[["event_time", "d_all", "d_senior"]],
                       same_out[["event_time", "d_all", "d_senior"]]],
                      ignore_index=True)
            .sort_values("event_time")
        )
        events["active_all"]    = events["d_all"].cumsum().clip(lower=0)
        events["active_senior"] = events["d_senior"].cumsum().clip(lower=0)

        records.append({
            "corridorID"         : corridor,
            "date"               : pd.Timestamp(day),
            "hour"               : hour,
            "peak_active"        : int(events["active_all"].max()),
            "peak_senior_active" : int(events["active_senior"].max()),
            "total_tapin"        : len(grp),
        })

    hourly = (
        pd.DataFrame(records)
        .sort_values(["corridorID", "date", "hour"])
        .reset_index(drop=True)
    )

    # Add overload label (needed as a feature — same logic as training)
    if bus_type == "big":
        hourly["overload"] = (hourly["peak_senior_active"] > PRIORITY_THRESH).astype(int)
    else:
        hourly["overload"] = (hourly["peak_active"] > SMALL_CAPACITY).astype(int)

    return hourly


def _build_window(grp: pd.DataFrame, scaler) -> Optional[np.ndarray]:
    """
    Ambil 7 baris terakhir, tambah calendar features, scale, return (1,7,6) array.
    Returns None jika data kurang dari LOOKBACK_DAYS.
    """
    grp = grp.sort_values("date").copy()
    if len(grp) < LOOKBACK_DAYS:
        return None

    grp["day_of_week"] = grp["date"].dt.dayofweek
    grp["is_weekend"]  = (grp["day_of_week"] >= 5).astype(int)

    window = grp[FEATURE_COLS].values[-LOOKBACK_DAYS:].astype(np.float32)
    return scaler.transform(window)[np.newaxis]  # (1, 7, 6)


# ── Core inference ────────────────────────────────────────────────────────────
def predict_allocation(
    hourly: pd.DataFrame,
    model: BusLSTM,
    scaler,
    threshold: float,
    alloc_table: dict,
    default_alloc: int,
    device: torch.device,
    target_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Untuk setiap (corridorID, hour):
      1. Ambil 7 hari data sebelum target_date sebagai window
      2. Prediksi probabilitas overload
      3. alokasi_hari_ini = alokasi_eksisting + int(prob >= threshold)

    target_date: tanggal prediksi. Default = hari terbaru di data.
    """
    if target_date is None:
        target_date = hourly["date"].max()
    else:
        target_date = pd.Timestamp(target_date)

    # Filter: hanya pakai data s.d. target_date - 1 hari sebagai lookback
    lookback_df = hourly[hourly["date"] < target_date].copy()

    records = []
    for (corridor, hour), grp in lookback_df.groupby(["corridorID", "hour"]):
        window = _build_window(grp, scaler)
        if window is None:
            continue

        x = torch.from_numpy(window).to(device)
        with torch.no_grad():
            prob = model.predict_proba(x).cpu().item()

        overload_pred  = int(prob >= threshold)
        alloc_existing = alloc_table.get(corridor, default_alloc)

        records.append({
            "corridorID"       : corridor,
            "hour"             : int(hour),
            "tanggal_prediksi" : target_date.date(),
            "hari"             : target_date.day_name(),
            "alokasi_eksisting": alloc_existing,
            "overload_prob"    : round(prob, 4),
            "overload_pred"    : overload_pred,
            "alokasi_hari_ini" : alloc_existing + overload_pred,
        })

    return (
        pd.DataFrame(records)
        .sort_values(["corridorID", "hour"])
        .reset_index(drop=True)
    )


# ── Summary printer ───────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame, bus_type: str) -> None:
    label     = "Bus Besar" if bus_type == "big" else "Bus Kecil"
    total     = len(df)
    n_overload = (df["overload_pred"] == 1).sum()

    print(f"\n{'='*65}")
    print(f"  ALOKASI HARI INI — {label.upper()}")
    print(f"  Tanggal : {df['tanggal_prediksi'].iloc[0]}  ({df['hari'].iloc[0]})")
    print(f"{'='*65}")
    print(f"  Total slot (corridor × jam) : {total:,}")
    print(f"  Slot butuh tambahan bus     : {n_overload:,}  ({n_overload/total*100:.1f}%)")

    print(f"\n  Ringkasan per jam:")
    summary = (
        df.groupby("hour")
        .agg(
            n_corridor    =("corridorID",        "count"),
            n_overload    =("overload_pred",      "sum"),
            avg_alokasi   =("alokasi_hari_ini",   "mean"),
            max_alokasi   =("alokasi_hari_ini",   "max"),
        )
        .reset_index()
    )
    summary["avg_alokasi"] = summary["avg_alokasi"].round(2)
    print(summary.to_string(index=False))

    top5 = (
        df[df["overload_pred"] == 1]
        .groupby("corridorID")["overload_pred"]
        .count()
        .sort_values(ascending=False)
        .head(5)
    )
    if not top5.empty:
        print(f"\n  Top-5 corridor paling sering overload:")
        for corr, cnt in top5.items():
            print(f"    {corr:<12}  {cnt} jam overload")
    print()


# ── Pipeline: load CSV → hourly → predict ────────────────────────────────────
def run_pipeline(
    csv_path: str,
    bus_type: str,
    target_date: Optional[str],
    device: torch.device,
) -> pd.DataFrame:
    alloc_table   = BIG_BUS_ALLOCATION if bus_type == "big" else SMALL_BUS_ALLOCATION
    default_alloc = 2

    print(f"\n[{bus_type.upper()}] Loading {csv_path} …")
    raw = pd.read_csv(csv_path, low_memory=False)
    print(f"  Raw rows: {len(raw):,}")

    print(f"[{bus_type.upper()}] Computing hourly peaks …")
    hourly = _compute_hourly_peaks(raw, bus_type)
    print(f"  Hourly slots: {len(hourly):,}  |  "
          f"Dates: {hourly['date'].min().date()} → {hourly['date'].max().date()}  |  "
          f"Corridors: {hourly['corridorID'].nunique()}")

    print(f"[{bus_type.upper()}] Loading model artifacts …")
    model, scaler, threshold = load_artifacts(bus_type, device)

    td = pd.Timestamp(target_date) if target_date else None
    if td is None:
        # Predict "today" = latest date in data
        td = hourly["date"].max()
        print(f"[{bus_type.upper()}] target_date not set → using latest: {td.date()}")

    print(f"[{bus_type.upper()}] Predicting allocation for {td.date()} …")
    result = predict_allocation(
        hourly, model, scaler, threshold,
        alloc_table, default_alloc, device, td,
    )
    print(f"  Predicted slots: {len(result):,}")
    print_summary(result, bus_type)
    return result


# ── Single-row inference (API / streaming use-case) ───────────────────────────
def predict_single(
    corridor_id: str,
    hour: int,
    history_7days: pd.DataFrame,
    bus_type: str,
    device: torch.device,
) -> dict:
    """
    Minimal inference untuk 1 (corridor, hour) tanpa re-loading CSV.

    history_7days: DataFrame dengan kolom:
        [date, peak_active, peak_senior_active, total_tapin, overload]
        Harus berisi tepat >= 7 baris, diurutkan ascending by date.

    Returns dict:
        corridor_id, hour, overload_prob, overload_pred, alokasi_hari_ini
    """
    model, scaler, threshold = load_artifacts(bus_type, device)

    alloc_table   = BIG_BUS_ALLOCATION if bus_type == "big" else SMALL_BUS_ALLOCATION
    default_alloc = 2

    df = history_7days.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(LOOKBACK_DAYS)

    if len(df) < LOOKBACK_DAYS:
        raise ValueError(
            f"history_7days harus berisi >= {LOOKBACK_DAYS} baris, dapat {len(df)}."
        )

    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    window = df[FEATURE_COLS].values.astype(np.float32)
    window_scaled = scaler.transform(window)[np.newaxis]  # (1,7,6)

    x = torch.from_numpy(window_scaled).to(device)
    with torch.no_grad():
        prob = model.predict_proba(x).cpu().item()

    overload_pred  = int(prob >= threshold)
    alloc_existing = alloc_table.get(corridor_id, default_alloc)

    return {
        "corridor_id"      : corridor_id,
        "hour"             : hour,
        "overload_prob"    : round(prob, 4),
        "overload_pred"    : overload_pred,
        "alokasi_eksisting": alloc_existing,
        "alokasi_hari_ini" : alloc_existing + overload_pred,
        "threshold_used"   : threshold,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prediksi alokasi bus TransJakarta hari ini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--big",   metavar="CSV", help="./predict_bus_besar.csv")
    p.add_argument("--small", metavar="CSV", help="./predict_bus_kecil.csv")
    p.add_argument(
        "--date", metavar="YYYY-MM-DD",
        help="Tanggal prediksi (default: tanggal terbaru di CSV)",
    )
    p.add_argument(
        "--out", metavar="FILE",
        help="Simpan hasil ke CSV (contoh: hasil_alokasi.csv)",
    )
    p.add_argument(
        "--artifacts", metavar="DIR", default="model_artifacts",
        help="Direktori model artifacts (default: model_artifacts)",
    )
    p.add_argument("--cpu", action="store_true", help="Paksa pakai CPU")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    global ARTIFACTS_DIR
    ARTIFACTS_DIR = Path(args.artifacts)

    if not args.big and not args.small:
        print("Error: harus specify minimal --big atau --small")
        sys.exit(1)

    device = torch.device("cpu") if args.cpu else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = []

    if args.big:
        df_big = run_pipeline(args.big, "big", args.date, device)
        df_big["bus_type"] = "big"
        results.append(df_big)

    if args.small:
        df_small = run_pipeline(args.small, "small", args.date, device)
        df_small["bus_type"] = "small"
        results.append(df_small)

    if results and args.out:
        combined = pd.concat(results, ignore_index=True)
        combined.to_csv(args.out, index=False)
        print(f"\n✓ Hasil disimpan ke: {args.out}  ({len(combined):,} rows)")


if __name__ == "__main__":
    main()
