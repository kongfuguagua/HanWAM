"""Shared raw CSV access for all simulation and control modules."""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "raw"

# Read by position because source CSV headers appear in multiple encodings.
INDEX_TO_NAME = {
    0: "ts",
    1: "T_out",
    2: "T_out_coil",
    3: "T_out_discharge",
    4: "freq",
    5: "eev",
    6: "fan_out",
    7: "I_comp",
    8: "T_in",
    9: "T_in_coil",
    10: "fan_in",
    12: "RH_in",
    13: "T_set",
    14: "mode",
    15: "energy_cum",
    23: "freq_in_tgt",
}


def discover_csvs(data_dir: Path = DATA_DIR) -> list[Path]:
    return sorted(data_dir.rglob("*.csv"))


def infer_split(path: Path) -> str:
    text = str(path)
    if f"{Path('3轮')}" in text:
        return "test"
    if f"{Path('2轮')}" in text:
        return "val"
    return "train"


def safe_source_name(path: Path, data_dir: Path = DATA_DIR) -> str:
    try:
        rel = path.relative_to(data_dir)
    except ValueError:
        rel = path
    return re.sub(r"[^0-9A-Za-z_-]+", "_", str(rel.with_suffix("")))


def read_raw_csv(path: Path, cols: list[str] | None = None) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("gbk", "utf-8-sig", "utf-8"):
        try:
            raw = pd.read_csv(path, encoding=encoding)
            break
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_error = exc
    else:
        raise ValueError(f"Cannot decode {path}: {last_error}")

    if raw.shape[1] <= max(INDEX_TO_NAME):
        raise ValueError(f"{path} has only {raw.shape[1]} columns")
    frame = raw.iloc[:, list(INDEX_TO_NAME)].copy()
    frame.columns = list(INDEX_TO_NAME.values())
    frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce", format="mixed")
    for col in frame.columns:
        if col != "ts":
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["source"] = safe_source_name(path)
    if cols is not None:
        return frame[cols].copy()
    return frame


def load_all_raw(data_dir: Path = DATA_DIR, verbose: bool = True) -> pd.DataFrame:
    frames = []
    for path in discover_csvs(data_dir):
        try:
            frames.append(read_raw_csv(path))
        except ValueError as exc:
            if verbose:
                print(f"[skip] {exc}")
    if not frames:
        raise ValueError(f"No readable CSV files under {data_dir}")
    frame = pd.concat(frames, ignore_index=True).dropna(subset=["ts"])
    if verbose:
        print(f"[load_all_raw] files={len(frames)}, rows={len(frame)}")
    return frame


def remove_constant_cols(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    constant_cols = []
    for col in frame.columns:
        if col in {"source", "ts"}:
            continue
        values = frame[col].dropna()
        if len(values) and values.nunique() <= 1:
            constant_cols.append(col)
    return frame.drop(columns=constant_cols), constant_cols
