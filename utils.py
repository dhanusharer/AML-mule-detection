"""
AML Mule Detection - Utility Functions
"""
import pandas as pd
import numpy as np
from glob import glob
import logging, os, gc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─── I/O Helpers ─────────────────────────────────────────────────────────────

def read_parquet_glob(pattern: str, columns=None, sample_frac=None) -> pd.DataFrame:
    """Read all parquet files matching glob, with optional column selection and sampling."""
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files match: {pattern}")
    log.info(f"Reading {len(files)} files matching {pattern}")
    dfs = []
    for f in files:
        df = pd.read_parquet(f, columns=columns)
        if sample_frac:
            df = df.sample(frac=sample_frac, random_state=42)
        dfs.append(df)
        del df; gc.collect()
    out = pd.concat(dfs, ignore_index=True)
    log.info(f"  → {len(out):,} rows, {out.shape[1]} cols")
    return out


def safe_read(path: str, columns=None) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=columns)
    log.info(f"Read {path}: {df.shape}")
    return df


def reduce_mem(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


# ─── Feature Utils ────────────────────────────────────────────────────────────

def safe_ratio(a: pd.Series, b: pd.Series, fill=0.0) -> pd.Series:
    return a.div(b.replace(0, np.nan)).fillna(fill)


def clip_percentile(series: pd.Series, lo=1, hi=99) -> pd.Series:
    lo_v = series.quantile(lo / 100)
    hi_v = series.quantile(hi / 100)
    return series.clip(lo_v, hi_v)


def gini(series: pd.Series) -> float:
    """Gini coefficient for a distribution."""
    arr = np.abs(series.dropna().values)
    if len(arr) == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    cumulative = np.cumsum(arr)
    return (2 * np.sum((np.arange(1, n + 1) * arr))) / (n * cumulative[-1]) - (n + 1) / n


def entropy(series: pd.Series) -> float:
    """Normalized Shannon entropy."""
    vc = series.value_counts(normalize=True)
    if len(vc) <= 1:
        return 0.0
    return float(-(vc * np.log2(vc + 1e-12)).sum() / np.log2(len(vc)))