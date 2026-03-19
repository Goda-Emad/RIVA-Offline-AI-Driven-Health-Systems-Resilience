"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           RIVA Health Platform — Sentiment Data Preparation                  ║
║           ai-core/training/prepare_sentiment_data.py                         ║
║                                                                              ║
║  Purpose : Load Arabic sentiment datasets (HTL, RES, PROD, ATT, MOV),       ║
║            clean and balance them, split train/val/test, and save as         ║
║            processed .npz files ready for DistilBERT fine-tuning.           ║
║                                                                              ║
║  Strategy                                                                    ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Phase 1 — Pre-train base:  HTL (15572) + RES (10970) = 26542 reviews       ║
║             → Teaches the model Egyptian Arabic language patterns,           ║
║               negation, intensifiers, and colloquial sentiment.             ║
║                                                                              ║
║  Phase 2 — Fine-tune medical: medical_labels.csv (500-1000 sentences)       ║
║             → Specialises the model for clinical context.                   ║
║             → File to be created manually by the medical team.              ║
║                                                                              ║
║  Why NOT MOV (movies)?  1524 reviews — movie language differs too much      ║
║  from patient speech. Risk of learning irrelevant sentiment patterns.        ║
║                                                                              ║
║  Usage                                                                       ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  python prepare_sentiment_data.py                 # full pipeline            ║
║  python prepare_sentiment_data.py --phase 1       # base only               ║
║  python prepare_sentiment_data.py --phase 2       # medical fine-tune only  ║
║  python prepare_sentiment_data.py --stats         # print stats, no write   ║
║                                                                              ║
║  Author  : Goda Emad (AI Core)                                               ║
║  Version : 1.0.0                                                             ║
║  Updated : 2026-03-18                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_sentiment")


# ═══════════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════════

REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
DATA_RAW     = REPO_ROOT / "data" / "raw"  / "arabic_sentiment" / "datasets"
DATA_PROC    = REPO_ROOT / "data" / "processed"
MEDICAL_CSV  = DATA_RAW  / "medical_labels.csv"   # Phase 2 — created by medical team

# Phase 1 sources (ordered by size + relevance to healthcare language)
PHASE1_SOURCES: list[dict] = [
    {"file": DATA_RAW / "HTL.csv",  "name": "hotel",      "size": 15572,
     "note": "Best transfer — patient complaints mirror hotel complaints (service, waiting, staff)"},
    {"file": DATA_RAW / "RES.csv",  "name": "restaurant", "size": 10970,
     "note": "Good for Egyptian colloquial expressions"},
    {"file": DATA_RAW / "PROD.csv", "name": "product",    "size": 4272,
     "note": "Useful for quality/satisfaction language"},
    {"file": DATA_RAW / "ATT.csv",  "name": "attraction", "size": 2154,
     "note": "Smaller dataset — included for vocabulary diversity"},
    # MOV.csv intentionally excluded — movie language too distant from medical
]

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

RANDOM_SEED     = 42
TEST_RATIO      = 0.10    # 10% test
VAL_RATIO       = 0.10    # 10% validation
MAX_TEXT_LEN    = 256     # tokens — DistilBERT limit
MIN_TEXT_LEN    = 5       # characters — filter very short noise
BALANCE_CLASSES = True    # oversample minority class to 50/50


# ═══════════════════════════════════════════════════════════════════════════
#  Text cleaner
# ═══════════════════════════════════════════════════════════════════════════

# Arabic diacritics (tashkeel) — remove for normalisation
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")

# URLs, mentions, hashtags, non-Arabic/English chars
_NOISE = re.compile(r"http\S+|@\w+|#\w+|[^\u0600-\u06FF\u0750-\u077F\s\w]")

# Repeated characters (أنا كويييييس → أنا كويس)
_REPEATED = re.compile(r"(.)\1{2,}")


def clean_text(text: str) -> str:
    """
    Normalise Arabic review text for DistilBERT tokenisation.

    Steps
    ─────
    1. Remove diacritics
    2. Remove URLs / mentions / hashtags
    3. Collapse repeated characters (كويييييس → كويس)
    4. Normalise whitespace
    5. Strip
    """
    text = str(text)
    text = _DIACRITICS.sub("", text)
    text = _NOISE.sub(" ", text)
    text = _REPEATED.sub(r"\1\1", text)     # keep max 2 repeats
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ═══════════════════════════════════════════════════════════════════════════
#  Loader
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset(path: Path, name: str) -> pd.DataFrame | None:
    """
    Load a dataset CSV with columns: [text, label] or [review, sentiment].
    Returns a normalised DataFrame with columns: text | label | source.
    label: 1 = positive, 0 = negative.
    """
    if not path.exists():
        logger.warning("File not found — skipping: %s", path.name)
        return None

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1256")

    df.columns = [c.lower().strip() for c in df.columns]

    # Detect text column
    text_col = next(
        (c for c in df.columns if c in ("text", "review", "comment", "نص", "مراجعة")),
        df.columns[0],
    )
    # Detect label column
    label_col = next(
        (c for c in df.columns if c in ("label", "sentiment", "polarity", "class", "تصنيف")),
        df.columns[1],
    )

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]
    df = df.dropna()
    df["text"]   = df["text"].astype(str)
    df["label"]  = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"]  = df["label"].astype(int)

    # Normalise label: keep only 0 (negative) and 1 (positive)
    df = df[df["label"].isin([0, 1, -1])]
    df["label"] = df["label"].replace(-1, 0)   # -1 → 0 (negative)

    df["source"] = name

    logger.info(
        "Loaded %-12s | rows=%5d  pos=%5d  neg=%5d",
        name, len(df),
        (df["label"] == 1).sum(),
        (df["label"] == 0).sum(),
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def build_phase1(stats_only: bool = False) -> pd.DataFrame | None:
    """Load, clean, and balance Phase 1 datasets (HTL + RES + PROD + ATT)."""

    frames: list[pd.DataFrame] = []
    for src in PHASE1_SOURCES:
        df = load_dataset(src["file"], src["name"])
        if df is not None:
            frames.append(df)

    if not frames:
        logger.error("No Phase 1 datasets found in %s", DATA_RAW)
        return None

    combined = pd.concat(frames, ignore_index=True)

    # ── Clean text ─────────────────────────────────────────────────────────
    combined["text"] = combined["text"].apply(clean_text)

    # ── Filter short texts ─────────────────────────────────────────────────
    before = len(combined)
    combined = combined[combined["text"].str.len() >= MIN_TEXT_LEN]
    logger.info("Dropped %d rows (text too short)", before - len(combined))

    # ── Truncate long texts ────────────────────────────────────────────────
    combined["text"] = combined["text"].str[:MAX_TEXT_LEN * 4]   # rough char limit

    # ── Drop duplicates ────────────────────────────────────────────────────
    before = len(combined)
    combined = combined.drop_duplicates(subset=["text"])
    logger.info("Dropped %d duplicate rows", before - len(combined))

    if stats_only:
        _print_stats("Phase 1 (combined)", combined)
        return combined

    # ── Balance classes ────────────────────────────────────────────────────
    if BALANCE_CLASSES:
        pos = combined[combined["label"] == 1]
        neg = combined[combined["label"] == 0]
        min_size = min(len(pos), len(neg))
        logger.info(
            "Balancing: pos=%d neg=%d → each capped at %d",
            len(pos), len(neg), min_size,
        )
        pos = resample(pos, n_samples=min_size, random_state=RANDOM_SEED)
        neg = resample(neg, n_samples=min_size, random_state=RANDOM_SEED)
        combined = pd.concat([pos, neg]).sample(frac=1, random_state=RANDOM_SEED)

    _print_stats("Phase 1 (balanced)", combined)
    return combined


def build_phase2(stats_only: bool = False) -> pd.DataFrame | None:
    """Load medical fine-tuning data (medical_labels.csv)."""

    if not MEDICAL_CSV.exists():
        logger.warning(
            "medical_labels.csv not found at %s — Phase 2 skipped.\n"
            "  To create it: annotate 500+ Egyptian Arabic medical sentences\n"
            "  with columns: text,label  (1=positive/calm, 0=negative/distress)",
            MEDICAL_CSV,
        )
        return None

    df = load_dataset(MEDICAL_CSV, "medical")
    if df is None:
        return None

    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() >= MIN_TEXT_LEN]

    if stats_only:
        _print_stats("Phase 2 (medical)", df)
        return df

    _print_stats("Phase 2 (medical)", df)
    return df


def split_and_save(df: pd.DataFrame, prefix: str) -> None:
    """
    Split DataFrame into train/val/test and save to data/processed/.

    Saves three .npz files:
        {prefix}_train.npz
        {prefix}_val.npz
        {prefix}_test.npz
    Each file contains: texts (array of str) + labels (array of int).
    """
    DATA_PROC.mkdir(parents=True, exist_ok=True)

    train_val, test = train_test_split(
        df, test_size=TEST_RATIO, random_state=RANDOM_SEED, stratify=df["label"]
    )
    val_ratio_adj = VAL_RATIO / (1 - TEST_RATIO)
    train, val = train_test_split(
        train_val, test_size=val_ratio_adj, random_state=RANDOM_SEED,
        stratify=train_val["label"],
    )

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        out_path = DATA_PROC / f"{prefix}_{split_name}.npz"
        np.savez(
            out_path,
            texts  = split_df["text"].values,
            labels = split_df["label"].values,
        )
        pos = (split_df["label"] == 1).sum()
        neg = (split_df["label"] == 0).sum()
        logger.info(
            "Saved %-30s | rows=%5d  pos=%5d  neg=%5d",
            out_path.name, len(split_df), pos, neg,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _print_stats(title: str, df: pd.DataFrame) -> None:
    pos = (df["label"] == 1).sum()
    neg = (df["label"] == 0).sum()
    total = len(df)
    logger.info("━" * 55)
    logger.info("  %s", title)
    logger.info("  Total   : %d", total)
    logger.info("  Positive: %d (%.1f%%)", pos, 100 * pos / total if total else 0)
    logger.info("  Negative: %d (%.1f%%)", neg, 100 * neg / total if total else 0)
    if "source" in df.columns:
        for src, grp in df.groupby("source"):
            logger.info("  %-12s: %d rows", src, len(grp))
    logger.info("━" * 55)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare Arabic sentiment datasets for RIVA DistilBERT training",
    )
    p.add_argument(
        "--phase", type=int, choices=[1, 2], default=None,
        help="Run only phase 1 (base) or phase 2 (medical fine-tune). Default: both.",
    )
    p.add_argument(
        "--stats", action="store_true",
        help="Print dataset statistics only — no files written.",
    )
    p.add_argument(
        "--no-balance", action="store_true",
        help="Skip class balancing (keep original distribution).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    global BALANCE_CLASSES
    if args.no_balance:
        BALANCE_CLASSES = False

    run_phase1 = args.phase in (None, 1)
    run_phase2 = args.phase in (None, 2)

    if run_phase1:
        logger.info("═" * 55)
        logger.info("  PHASE 1 — Base pre-training data")
        logger.info("═" * 55)
        df1 = build_phase1(stats_only=args.stats)
        if df1 is not None and not args.stats:
            split_and_save(df1, prefix="sentiment_base")

    if run_phase2:
        logger.info("═" * 55)
        logger.info("  PHASE 2 — Medical fine-tuning data")
        logger.info("═" * 55)
        df2 = build_phase2(stats_only=args.stats)
        if df2 is not None and not args.stats:
            split_and_save(df2, prefix="sentiment_medical")

    if not args.stats:
        logger.info("Done ✅  — files saved to %s", DATA_PROC)


if __name__ == "__main__":
    main()
