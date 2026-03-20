"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           RIVA Health Platform — Sentiment Analyzer Test Suite               ║
║           ai-core/training/test_sentiment.py                                 ║
║                                                                              ║
║  Purpose : Evaluate SentimentAnalyzer accuracy on:                          ║
║            1. Unit tests  — hand-crafted Egyptian Arabic medical cases       ║
║            2. Lexicon tests — edge cases (negation, intensifier, punctuation)║
║            3. Dataset eval — sentiment_base_test.npz / medical_test.npz     ║
║            4. ONNX eval   — compare lexicon-only vs hybrid scores           ║
║                                                                              ║
║  Metrics reported                                                            ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  • Accuracy, Weighted F1, Macro F1                                          ║
║  • Confusion matrix                                                          ║
║  • Per-engine breakdown (lexicon vs hybrid)                                 ║
║  • Latency per call (ms)                                                     ║
║  • Edge case pass rate                                                       ║
║                                                                              ║
║  Usage                                                                       ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  python test_sentiment.py                    # run all tests                 ║
║  python test_sentiment.py --unit             # unit tests only               ║
║  python test_sentiment.py --dataset          # dataset evaluation only       ║
║  python test_sentiment.py --edge-cases       # edge case tests only         ║
║  python test_sentiment.py --report report.json  # save JSON report          ║
║                                                                              ║
║  Author  : Goda Emad  (AI Core)                                              ║
║  Version : 1.1.0                                                             ║
║  Updated : 2026-03-18                                                        ║
║                                                                              ║
║  Changelog v1.1.0                                                            ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  • FIX  pregnancy_طلق: restored "طلق" + natural Egyptian phrasing.          ║
║  • FIX  duration_بقالي: restored "بقالي يومين" anchor.                      ║
║  • FIX  ZeroDivisionError in run_dataset_eval: guarded latency avg.         ║
║  • FIX  Report mkdir(parents=True) — works for nested paths.                ║
║  • CHG  Summary: _summarise() helper — cleaner, less brittle.               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_sentiment")

# ── Optional sklearn ──────────────────────────────────────────────────────
try:
    from sklearn.metrics import (
        accuracy_score, f1_score,
        classification_report, confusion_matrix,
    )
    _SKLEARN = True
except ImportError:
    _SKLEARN = False
    logger.warning("scikit-learn not installed — dataset metrics unavailable")

# ── SentimentAnalyzer import ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from voice.dialect_model.sentiment_analyzer import SentimentAnalyzer
except ImportError:
    logger.error(
        "Cannot import SentimentAnalyzer.\n"
        "  Make sure you run from the repo root:\n"
        "  python ai-core/training/test_sentiment.py"
    )
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROC = REPO_ROOT / "data" / "processed"


# ═══════════════════════════════════════════════════════════════════════════
#  Result dataclasses
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CaseResult:
    """Result of a single test case."""
    name        : str
    text        : str
    passed      : bool
    score       : float
    expected    : str        # "negative" | "positive" | "neutral" | "emergency"
    actual_label: str
    engine_used : str
    latency_ms  : float
    note        : str = ""


@dataclass
class SuiteReport:
    """Aggregated report for a test suite."""
    suite_name   : str
    total        : int
    passed       : int
    failed       : int
    pass_rate    : float
    avg_latency_ms: float
    cases        : list[CaseResult] = field(default_factory=list)

    @property
    def failed_cases(self) -> list[CaseResult]:
        return [c for c in self.cases if not c.passed]


# ═══════════════════════════════════════════════════════════════════════════
#  Test cases — hand-crafted Egyptian Arabic medical scenarios
# ═══════════════════════════════════════════════════════════════════════════

# Format: (name, text, expected_polarity, expected_label_fragment, note)
# expected_polarity: "negative" | "positive" | "neutral" | "emergency"
# label_fragment is checked with `in` — use the first distinctive chars
# "انزعاج" matches both "انزعاج خفيف" and "انزعاج"
# "قلق"    matches "قلق" label
# "مطمئن"  matches "مطمئن" and "مطمئن نسبياً"
# ""       means any label is acceptable (score polarity check only)

UNIT_CASES: list[tuple[str, str, str, str, str]] = [

    # ── Emergency ────────────────────────────────────────────────────────
    ("emergency_نجدة",
     "نجدة مش قادر أتنفس",
     "emergency", "ضائقة", "Direct emergency trigger"),

    ("emergency_بيموت",
     "المريض بيموت",
     "emergency", "ضائقة", "Life threat keyword"),

    ("emergency_جلطة",
     "نجدة جلطة",              # نجدة triggers emergency
     "emergency", "ضائقة", "Stroke with emergency trigger"),

    # ── Severe negative ───────────────────────────────────────────────────
    ("severe_وجع_ناري",
     "عندي وجع ناري في صدري",
     "negative", "انزعاج", "Severe pain with intensifier"),

    ("severe_خايف_أوي",
     "أنا خايف أوي وقلقان",
     "negative", "انزعاج", "Fear + anxiety combined"),

    ("severe_ضيق_تنفس",
     "عندي ضيق تنفس من امبارح",
     "negative", "انزعاج", "Breathing difficulty"),

    # ── Moderate negative ─────────────────────────────────────────────────
    ("moderate_تعبان",
     "أنا تعبان وعندي حرارة",
     "negative", "انزعاج", "Sick with fever"),

    ("moderate_دايخ",
     "دايخ وعندي صداع",
     "negative", "انزعاج", "Dizziness with headache"),

    ("moderate_قيء",
     "عندي غثيان وقيء",
     "negative", "انزعاج", "Nausea and vomiting"),

    # ── Negation cases ────────────────────────────────────────────────────
    ("negation_مش_تعبان",
     "أنا مش تعبان",
     "neutral", "", "Negated sickness — score near 0"),

    ("negation_معنديش_ألم",
     "معنديش ألم دلوقتي",
     "neutral", "محايد", "Circumfix negation — no pain"),

    ("negation_مش_دايخ",
     "الحمد لله مش دايخ",
     "positive", "مطمئن", "Negated dizziness + gratitude"),

    # ── Negation + Intensifier (FIX 1) ───────────────────────────────────
    ("neg_intens_مش_تعبان_أوي",
     "أنا مش تعبان أوي",
     "negative", "انزعاج", "Negation + intensifier amplifies residual"),

    ("neg_intens_مش_تعبان_خالص",
     "أنا مش تعبان خالص",
     "negative", "انزعاج", "Negation + خالص — still slightly residual"),

    # ── Punctuation cases (FIX 2) ─────────────────────────────────────────
    ("punct_arabic_فاصلة",
     "أنا تعبان، ومش قادر أقوم",
     "negative", "انزعاج", "Arabic comma attached to word"),

    ("punct_exclamation",
     "عندي وجع شديد! من امبارح",
     "negative", "انزعاج", "Exclamation mark mid-sentence"),

    ("punct_question",
     "عندي تورم في رجلي؟",
     "negative", "انزعاج", "Arabic question mark"),

    # ── Multi-occurrence (FIX 3) ──────────────────────────────────────────
    ("multi_occurrence_وجع",
     "وجع في صدري وكمان وجع في ضهري",
     "negative", "انزعاج", "Same term twice — should score both"),

    ("multi_occurrence_ألم",
     "ألم في راسي وألم في بطني",
     "negative", "انزعاج", "Two pain complaints"),

    # ── Negated emergency (FIX 5) ─────────────────────────────────────────
    ("negated_emergency_مش_بيموت",
     "الحمدلله المريض مش بيموت",
     "neutral", "", "Negated death — should NOT trigger emergency"),

    ("negated_emergency_مش_بتموت",
     "هي مش بتموت بس تعبانة",
     "negative", "", "Negated death + sick — انزعاج خفيف or lower"),

    # ── Positive / improvement ────────────────────────────────────────────
    ("positive_اتحسن",
     "الحمد لله اتحسن كتير",
     "positive", "مطمئن", "Recovery with intensifier"),

    ("positive_كويس",
     "أنا كويس تمام والحمدلله",
     "positive", "مطمئن", "Multiple positive signals"),

    ("positive_أحسن",
     "بقيت أحسن",
     "positive", "مطمئن", "Improvement"),

    # ── Mild / neutral ────────────────────────────────────────────────────
    ("mild_شوية",
     "عندي وجع بسيط شوية",
     "neutral", "محايد", "Mild pain minimised"),

    ("mild_intermittent",
     "الألم بيجي ويروح",
     "positive", "مطمئن", "Intermittent — positive mitigation"),

    # ── Pregnancy module ──────────────────────────────────────────────────
    ("pregnancy_طلق",
     "عندها طلق تعبانة قلقانة",    # بدون واو لاصقة — كل كلمة token مستقل
     "negative", "انزعاج", "Labor pain + distress"),

    ("pregnancy_اتحسن",
     "الحمدلله كويسة",
     "positive", "مطمئن", "Reassurance"),

    # ── Duration context ──────────────────────────────────────────────────
    ("duration_بقالي",
     "بقالي يومين تعبان وحاسس بإرهاق",   # FIX v1.1: بقالي restored — تعبان/إرهاق score
     "negative", "انزعاج", "Egyptian duration anchor + sickness"),

    ("duration_من_امبارح",
     "من امبارح وأنا تعبان",
     "negative", "انزعاج", "Since yesterday"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  Edge-case tests — specific to v2.1 fixes
# ═══════════════════════════════════════════════════════════════════════════

EDGE_CASES: list[tuple[str, str, tuple[float, float], str]] = [
    # (name, text, (score_min, score_max), note)

    # FIX 1 — negation magnitude scales with intensifier
    ("fix1_negation_baseline",
     "مش تعبان",              (-0.15, +0.15), "baseline negation"),
    ("fix1_negation_plus_أوي",
     "مش تعبان أوي",          (-0.25, +0.10), "أوي amplifies negated term"),
    ("fix1_negation_plus_خالص",
     "مش تعبان خالص",         (-0.20, +0.20), "خالص negation — near neutral"),

    # FIX 2 — punctuation tolerance
    ("fix2_arabic_comma",
     "تعبان، جداً",            (-0.60, -0.10), "Arabic comma before intensifier"),
    ("fix2_mixed_punct",
     "وجع! شديد؟ من امبارح",  (-0.60, -0.10), "Multiple punctuation marks"),

    # FIX 3 — repeated terms
    ("fix3_single_وجع",
     "وجع",                   (-0.40, -0.05), "single occurrence"),
    ("fix3_double_وجع",
     "وجع وجع",               (-0.80, -0.15), "double should score lower"),

    # FIX 5 — negated emergency calibration
    ("fix5_real_emergency",
     "نجدة",                   (-1.01, -0.99), "must be -1.0"),
    ("fix5_negated_emergency",
     "مش بيموت",              (-0.15, +0.15), "negated — near 0"),
    ("fix5_negated_نجدة",
     "مش محتاج نجدة",         (-0.30, +0.10), "negated call-for-help"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════════

class SentimentTester:

    def __init__(self, analyzer: SentimentAnalyzer) -> None:
        self._sa = analyzer

    # ── Unit tests ───────────────────────────────────────────────────────

    def run_unit_tests(self) -> SuiteReport:
        """Run all hand-crafted medical test cases."""
        results: list[CaseResult] = []

        for name, text, expected_pol, lbl_frag, note in UNIT_CASES:
            t0  = time.perf_counter()
            res = self._sa.score_full(text)
            ms  = (time.perf_counter() - t0) * 1000

            # Determine actual polarity from score
            actual_pol = self._score_to_polarity(res.score, res.engine_used)

            # Pass condition: polarity matches AND label contains fragment
            # Special case: if label is "محايد" always treat as neutral
            if res.label == "محايد":
                actual_pol = "neutral"
            passed = (
                actual_pol == expected_pol
                and (lbl_frag == "" or lbl_frag in res.label)
            )

            results.append(CaseResult(
                name         = name,
                text         = text,
                passed       = passed,
                score        = round(res.score, 3),
                expected     = expected_pol,
                actual_label = res.label,
                engine_used  = res.engine_used,
                latency_ms   = round(ms, 2),
                note         = note,
            ))

        return self._make_report("unit_tests", results)

    # ── Edge case tests ───────────────────────────────────────────────────

    def run_edge_cases(self) -> SuiteReport:
        """Run v2.1 fix-specific edge cases — score range assertions."""
        results: list[CaseResult] = []

        for name, text, (lo, hi), note in EDGE_CASES:
            t0  = time.perf_counter()
            res = self._sa.score_full(text)
            ms  = (time.perf_counter() - t0) * 1000

            passed = lo <= res.score <= hi

            results.append(CaseResult(
                name         = name,
                text         = text,
                passed       = passed,
                score        = round(res.score, 3),
                expected     = f"[{lo}, {hi}]",
                actual_label = res.label,
                engine_used  = res.engine_used,
                latency_ms   = round(ms, 2),
                note         = note,
            ))

        return self._make_report("edge_cases_v2.1", results)

    # ── Dataset evaluation ────────────────────────────────────────────────

    def run_dataset_eval(self) -> dict:
        """
        Evaluate on sentiment_base_test.npz and sentiment_medical_test.npz.

        Converts SentimentAnalyzer's [-1, +1] score → binary label:
          score >= 0  → positive (1)
          score <  0  → negative (0)
        """
        if not _SKLEARN:
            logger.error("scikit-learn required for dataset evaluation")
            return {}

        results = {}
        for prefix in ("sentiment_base", "sentiment_medical"):
            npz_path = DATA_PROC / f"{prefix}_test.npz"
            if not npz_path.exists():
                logger.warning("Test file not found: %s", npz_path)
                continue

            data   = np.load(npz_path, allow_pickle=True)
            texts  = data["texts"].tolist()
            labels = data["labels"].tolist()

            preds   = []
            latency = []
            for text in texts:
                t0  = time.perf_counter()
                res = self._sa.score_full(str(text))
                latency.append((time.perf_counter() - t0) * 1000)
                preds.append(1 if res.score >= 0 else 0)

            acc = accuracy_score(labels, preds)
            f1w = f1_score(labels, preds, average="weighted")
            f1m = f1_score(labels, preds, average="macro")
            cm  = confusion_matrix(labels, preds).tolist()

            logger.info("━" * 55)
            logger.info("  Dataset : %s", prefix)
            logger.info("  Samples : %d", len(texts))
            logger.info("  Accuracy: %.4f", acc)
            logger.info("  F1 (w)  : %.4f", f1w)
            logger.info("  F1 (m)  : %.4f", f1m)
            # FIX v1.1: guard against ZeroDivisionError if latency list is empty
            avg_lat = sum(latency) / len(latency) if latency else 0.0
            logger.info("  Avg lat : %.2f ms", avg_lat)
            logger.info("\n%s", classification_report(
                labels, preds, target_names=["negative", "positive"]
            ))

            results[prefix] = {
                "samples"         : len(texts),
                "accuracy"        : round(acc, 4),
                "f1_weighted"     : round(f1w, 4),
                "f1_macro"        : round(f1m, 4),
                "confusion_matrix": cm,
                "avg_latency_ms"  : round(avg_lat, 2),
                "engine"          : self._sa.active_engine if hasattr(self._sa, "active_engine") else "unknown",
            }

        return results

    # ── Latency benchmark ─────────────────────────────────────────────────

    def run_latency_benchmark(self, n: int = 200) -> dict:
        """
        Measure average latency over n calls.
        Target: < 10ms per call (offline, lexicon mode).
        """
        sample_texts = [t for _, t, *_ in UNIT_CASES]
        # Repeat to reach n calls
        texts  = (sample_texts * (n // len(sample_texts) + 1))[:n]
        times  = []

        for text in texts:
            t0 = time.perf_counter()
            self._sa.score(text)
            times.append((time.perf_counter() - t0) * 1000)

        avg = sum(times) / len(times)
        p95 = sorted(times)[int(0.95 * len(times))]
        p99 = sorted(times)[int(0.99 * len(times))]

        result = {
            "n_calls"   : n,
            "avg_ms"    : round(avg, 3),
            "p95_ms"    : round(p95, 3),
            "p99_ms"    : round(p99, 3),
            "target_ms" : 10.0,
            "passed"    : avg < 10.0,
        }

        logger.info("━" * 55)
        logger.info("  Latency Benchmark (%d calls)", n)
        logger.info("  Avg : %.3f ms  %s", avg, "✅" if avg < 10 else "⚠️ >10ms")
        logger.info("  P95 : %.3f ms", p95)
        logger.info("  P99 : %.3f ms", p99)

        return result

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _score_to_polarity(score: float, engine: str) -> str:
        if engine == "emergency":
            return "emergency"
        if score >= 0.12:
            return "positive"
        if score <= -0.09:
            return "negative"
        return "neutral"

    @staticmethod
    def _make_report(name: str, results: list[CaseResult]) -> SuiteReport:
        passed = sum(1 for r in results if r.passed)
        total  = len(results)
        avg_ms = sum(r.latency_ms for r in results) / total if total else 0
        return SuiteReport(
            suite_name    = name,
            total         = total,
            passed        = passed,
            failed        = total - passed,
            pass_rate     = round(passed / total, 4) if total else 0,
            avg_latency_ms= round(avg_ms, 2),
            cases         = results,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Printer
# ═══════════════════════════════════════════════════════════════════════════

def print_report(report: SuiteReport) -> None:
    """Print a formatted test report to stdout."""
    icon = "✅" if report.pass_rate == 1.0 else ("⚠️" if report.pass_rate >= 0.80 else "❌")
    logger.info("═" * 58)
    logger.info("  %s  %s", icon, report.suite_name)
    logger.info("  Passed : %d / %d  (%.1f%%)",
                report.passed, report.total, report.pass_rate * 100)
    logger.info("  Avg latency: %.2f ms", report.avg_latency_ms)

    if report.failed_cases:
        logger.info("  ── Failed cases ──")
        for c in report.failed_cases:
            logger.info(
                "  ❌ %-35s score=%-7.3f expected=%-12s got=%s",
                c.name, c.score, c.expected, c.actual_label,
            )
            logger.info("     text: %s", c.text)
    logger.info("═" * 58)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test RIVA SentimentAnalyzer — unit + edge + dataset"
    )
    p.add_argument("--unit",        action="store_true", help="Run unit tests only")
    p.add_argument("--edge-cases",  action="store_true", help="Run edge case tests only")
    p.add_argument("--dataset",     action="store_true", help="Run dataset evaluation only")
    p.add_argument("--latency",     action="store_true", help="Run latency benchmark only")
    p.add_argument("--onnx",        action="store_true", help="Load ONNX model for hybrid eval")
    p.add_argument("--report",      type=str, default=None,
                   help="Save full JSON report to this path")
    p.add_argument("--n-latency",   type=int, default=200,
                   help="Number of calls for latency benchmark (default 200)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Decide which tests to run
    run_all = not any([args.unit, args.edge_cases, args.dataset, args.latency])

    # Init analyzer
    if args.onnx:
        logger.info("Loading SentimentAnalyzer with ONNX model...")
        sa = SentimentAnalyzer()
    else:
        logger.info("Loading SentimentAnalyzer in lexicon-only mode...")
        sa = SentimentAnalyzer(onnx_model_path=False)

    tester     = SentimentTester(sa)
    full_report: dict = {
        "engine"    : "hybrid" if sa.has_onnx_model else "lexicon",
        "suites"    : {},
    }

    # ── Unit tests ────────────────────────────────────────────────────────
    if run_all or args.unit:
        logger.info("\n" + "─" * 58)
        logger.info("  Running unit tests (%d cases)...", len(UNIT_CASES))
        unit_report = tester.run_unit_tests()
        print_report(unit_report)
        full_report["suites"]["unit_tests"] = {
            "pass_rate"    : unit_report.pass_rate,
            "passed"       : unit_report.passed,
            "total"        : unit_report.total,
            "avg_latency_ms": unit_report.avg_latency_ms,
            "failed_cases" : [c.name for c in unit_report.failed_cases],
        }

    # ── Edge cases ────────────────────────────────────────────────────────
    if run_all or args.edge_cases:
        logger.info("\n" + "─" * 58)
        logger.info("  Running edge case tests (%d cases)...", len(EDGE_CASES))
        edge_report = tester.run_edge_cases()
        print_report(edge_report)
        full_report["suites"]["edge_cases"] = {
            "pass_rate"    : edge_report.pass_rate,
            "passed"       : edge_report.passed,
            "total"        : edge_report.total,
            "failed_cases" : [c.name for c in edge_report.failed_cases],
        }

    # ── Dataset evaluation ────────────────────────────────────────────────
    if run_all or args.dataset:
        logger.info("\n" + "─" * 58)
        logger.info("  Running dataset evaluation...")
        dataset_results = tester.run_dataset_eval()
        full_report["suites"]["dataset_eval"] = dataset_results

    # ── Latency benchmark ─────────────────────────────────────────────────
    if run_all or args.latency:
        logger.info("\n" + "─" * 58)
        logger.info("  Running latency benchmark (%d calls)...", args.n_latency)
        lat = tester.run_latency_benchmark(args.n_latency)
        full_report["suites"]["latency"] = lat

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "═" * 58)
    logger.info("  SUMMARY")
    logger.info("═" * 58)

    # FIX v1.1: use a helper to determine suite type — avoids brittle key checks
    def _summarise(suite: str, data: dict) -> None:
        if "pass_rate" in data:
            icon = "✅" if data["pass_rate"] == 1.0 else "❌"
            logger.info(
                "  %s %-25s %.1f%% (%d/%d)",
                icon, suite,
                data["pass_rate"] * 100,
                data.get("passed", 0),
                data.get("total", 0),
            )
        elif "avg_ms" in data:
            icon = "✅" if data.get("passed") else "⚠️"
            logger.info(
                "  %s %-25s avg=%.2fms  p95=%.2fms",
                icon, suite, data["avg_ms"], data["p95_ms"],
            )
        elif isinstance(data, dict):
            # dataset_eval — each key is a dataset name mapping to metrics dict
            for ds_name, metrics in data.items():
                if isinstance(metrics, dict) and "accuracy" in metrics:
                    logger.info(
                        "  📊 %-25s acc=%.3f  f1=%.3f",
                        ds_name,
                        metrics.get("accuracy", 0),
                        metrics.get("f1_weighted", 0),
                    )

    for suite, data in full_report["suites"].items():
        _summarise(suite, data)

    # ── Save report ───────────────────────────────────────────────────────
    if args.report:
        report_path = Path(args.report)
        # FIX v1.1: create parent directories if they don't exist
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, ensure_ascii=False, indent=2)
        logger.info("\nReport saved → %s", report_path)

    logger.info("Done ✅")


if __name__ == "__main__":
    main()
