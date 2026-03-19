"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           RIVA Health Platform — Lexicon Builder                             ║
║           ai-core/training/build_lexicon.py                                  ║
║                                                                              ║
║  Purpose : Load ALL_lex.csv (1913 terms) from the arabic_sentiment dataset, ║
║            filter medically-relevant terms, assign clinical weights, and    ║
║            merge them into egyptian_medical_lexicon.json.                   ║
║                                                                              ║
║  Run once at setup — output is the final lexicon used by SentimentAnalyzer. ║
║                                                                              ║
║  Usage                                                                       ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  python build_lexicon.py                          # uses default paths       ║
║  python build_lexicon.py --dry-run                # preview, no file write   ║
║  python build_lexicon.py --source LABR_lex.csv    # use book domain only    ║
║                                                                              ║
║  Author  : Goda Emad (AI Core)                                               ║
║  Version : 1.0.0                                                             ║
║  Updated : 2026-03-18                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_lexicon")


# ═══════════════════════════════════════════════════════════════════════════
#  Paths (relative to repo root)
# ═══════════════════════════════════════════════════════════════════════════

REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
LEXICON_DIR  = REPO_ROOT / "data" / "raw" / "arabic_sentiment" / "lexicons"
OUTPUT_JSON  = LEXICON_DIR / "egyptian_medical_lexicon.json"

# Priority: ALL_lex first, then LABR (book) for emotional depth, then RES for
# service-quality terms that transfer well to healthcare satisfaction.
SOURCE_FILES = [
    LEXICON_DIR / "ALL_lex.csv",
    LEXICON_DIR / "LABR_lex.csv",
    LEXICON_DIR / "RES_lex.csv",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Medical relevance filter
# ═══════════════════════════════════════════════════════════════════════════
# Terms whose polarity in a medical context differs from general sentiment
# are remapped here.  E.g. "حار" (hot) is positive in food reviews but
# maps to fever/heat in a medical context → excluded.
# Terms not in any list are kept at their original polarity.

# Completely exclude from the medical lexicon (domain mismatch)
_EXCLUDE_TERMS: frozenset[str] = frozenset({
    "حار", "لذيذ", "طازج", "رائع", "ممتاز",   # food/service positive — not medical
    "سعر", "غالي", "رخيص",                      # price — not medical
    "سريع", "بطيء",                              # service speed — ambiguous
    "نظيف", "وسخ",                               # cleanliness — could be wound context but too ambiguous
    "جميل", "حلو",                               # aesthetic positive — not medical
})

# Remap weight for terms that appear in ALL_lex with wrong polarity for medical use
_WEIGHT_REMAP: dict[str, float] = {
    "نشيط"   : -0.10,   # "active" is positive in reviews but can mean hyperactive (neuro)
    "خفيف"   : 0.30,    # ALL_lex may score neutral; in medical context = mild symptom = positive
    "ثقيل"   : -0.25,   # "heavy" — maps to heaviness/pressure sensation
}

# Minimum absolute weight to include a term (filter near-neutral noise)
_MIN_ABS_WEIGHT = 0.10

# Map ALL_lex polarity labels → numeric weights
# (ALL_lex uses: 1 = positive, -1 = negative, 0 = neutral)
_POLARITY_WEIGHT: dict[int, float] = {
    1  :  0.30,   # positive → moderate positive in medical context
    -1 : -0.30,   # negative → moderate negative
    0  :  0.00,   # neutral → excluded by _MIN_ABS_WEIGHT
}


# ═══════════════════════════════════════════════════════════════════════════
#  Loader
# ═══════════════════════════════════════════════════════════════════════════

def load_csv(path: Path) -> pd.DataFrame:
    """
    Load a lexicon CSV file.

    Expected columns (ALL_lex format):
        word      : Arabic term
        polarity  : 1 (positive) | -1 (negative) | 0 (neutral)

    Falls back to positional loading if columns are unnamed.
    """
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1256")   # Windows Arabic encoding fallback

    # Normalise column names
    df.columns = [c.lower().strip() for c in df.columns]

    if "word" not in df.columns:
        # Positional: assume col0=word, col1=polarity
        df.columns = ["word", "polarity"] + list(df.columns[2:])

    df = df[["word", "polarity"]].dropna()
    df["word"]     = df["word"].astype(str).str.strip()
    df["polarity"] = pd.to_numeric(df["polarity"], errors="coerce").fillna(0).astype(int)

    logger.info("Loaded %s — %d terms", path.name, len(df))
    return df


def filter_and_weight(df: pd.DataFrame, source_name: str) -> list[dict]:
    """
    Apply medical relevance filter and convert polarity to float weight.

    Returns a list of term dicts ready to merge into the JSON lexicon.
    """
    results = []
    skipped_exclude = 0
    skipped_weight  = 0

    for _, row in df.iterrows():
        term     = str(row["word"]).strip()
        polarity = int(row["polarity"])

        # Skip excluded terms
        if term in _EXCLUDE_TERMS:
            skipped_exclude += 1
            continue

        # Compute weight
        if term in _WEIGHT_REMAP:
            weight = _WEIGHT_REMAP[term]
        else:
            weight = _POLARITY_WEIGHT.get(polarity, 0.0)

        # Skip near-neutral
        if abs(weight) < _MIN_ABS_WEIGHT:
            skipped_weight += 1
            continue

        # Determine medical category from weight
        if weight <= -0.60:
            category = "negative_severe"
        elif weight <= -0.20:
            category = "negative_moderate"
        elif weight < 0:
            category = "negative_mild"
        elif weight >= 0.55:
            category = "positive_strong"
        elif weight >= 0.15:
            category = "positive_moderate"
        else:
            category = "positive_mild"

        results.append({
            "term"            : term,
            "weight"          : round(weight, 2),
            "trigger"         : False,
            "urgency_override": None,
            "source"          : source_name,
            "context_tags"    : ["from_csv"],
            "_category"       : category,   # internal — removed before JSON write
        })

    logger.info(
        "  filtered: kept=%d  excluded=%d  near-neutral=%d",
        len(results), skipped_exclude, skipped_weight,
    )
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Merger
# ═══════════════════════════════════════════════════════════════════════════

def merge_into_lexicon(
    new_terms   : list[dict],
    lexicon_path: Path,
    dry_run     : bool = False,
) -> dict:
    """
    Merge new_terms into the existing JSON lexicon.

    Strategy
    ────────
    • If a term already exists in the JSON: keep the manually-curated entry
      (higher quality) and mark its source as "manual+csv".
    • If a term is new: append it to the appropriate section.
    • Never overwrite manually-set weights or trigger flags.

    Returns the updated lexicon dict.
    """
    with open(lexicon_path, encoding="utf-8") as f:
        lexicon = json.load(f)

    # Build a flat index of all existing terms for fast lookup
    existing: dict[str, str] = {}   # term → section_name
    section_names = [
        "emergency_triggers", "negative_severe", "negative_moderate",
        "negative_mild", "positive_strong", "positive_moderate",
        "pregnancy_module", "school_health_module",
    ]
    for section in section_names:
        for entry in lexicon.get(section, {}).get("terms", []):
            existing[entry["term"]] = section

    added   = 0
    updated = 0
    skipped = 0

    for item in new_terms:
        term     = item["term"]
        category = item.pop("_category")   # remove internal key

        if term in existing:
            # Mark as also found in CSV but keep manual weight
            section = existing[term]
            for entry in lexicon[section]["terms"]:
                if entry["term"] == term:
                    if "csv" not in entry.get("source", ""):
                        entry["source"] = entry["source"] + "+csv"
                    updated += 1
                    break
            skipped += 1
            continue

        # New term — append to correct section
        target_section = category
        if target_section not in lexicon:
            lexicon[target_section] = {"_note": "auto-generated from CSV", "terms": []}

        lexicon[target_section]["terms"].append(item)
        existing[term] = target_section
        added += 1

    logger.info(
        "Merge complete | added=%d  updated_source=%d  already_existed=%d",
        added, updated, skipped,
    )

    # Update meta
    total = sum(
        len(lexicon.get(s, {}).get("terms", []))
        for s in section_names
    )
    if "_meta" in lexicon:
        lexicon["_meta"]["total_terms"] = total
        lexicon["_meta"]["updated"]     = "2026-03-18"

    if not dry_run:
        with open(lexicon_path, "w", encoding="utf-8") as f:
            json.dump(lexicon, f, ensure_ascii=False, indent=2)
        logger.info("Saved → %s  (total_terms=%d)", lexicon_path, total)
    else:
        logger.info("[DRY RUN] Would save %d total terms — no file written", total)

    return lexicon


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build / update RIVA medical lexicon from Arabic sentiment CSVs",
    )
    p.add_argument(
        "--source", type=str, default=None,
        help="Process only this CSV file (e.g. ALL_lex.csv). Default: all 3 sources.",
    )
    p.add_argument(
        "--output", type=Path, default=OUTPUT_JSON,
        help="Path to the output JSON lexicon file.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without writing to disk.",
    )
    p.add_argument(
        "--min-weight", type=float, default=_MIN_ABS_WEIGHT,
        help="Minimum absolute weight to include a term (default 0.10).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    global _MIN_ABS_WEIGHT
    _MIN_ABS_WEIGHT = args.min_weight

    # Determine which source files to process
    if args.source:
        sources = [LEXICON_DIR / args.source]
    else:
        sources = SOURCE_FILES

    # Validate output JSON exists
    if not args.output.exists():
        logger.error("Output JSON not found: %s", args.output)
        logger.error("Run from repo root and make sure egyptian_medical_lexicon.json exists.")
        sys.exit(1)

    all_new_terms: list[dict] = []

    for csv_path in sources:
        if not csv_path.exists():
            logger.warning("CSV not found — skipping: %s", csv_path)
            continue

        df       = load_csv(csv_path)
        new_terms = filter_and_weight(df, source_name=csv_path.stem)
        all_new_terms.extend(new_terms)

    if not all_new_terms:
        logger.warning("No terms to merge — check CSV paths and filter settings.")
        return

    logger.info("Total new candidate terms: %d", len(all_new_terms))

    merge_into_lexicon(
        new_terms    = all_new_terms,
        lexicon_path = args.output,
        dry_run      = args.dry_run,
    )

    logger.info("Done ✅")


if __name__ == "__main__":
    main()
