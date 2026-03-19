"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           RIVA Health Platform — Sentiment Model Trainer                     ║
║           ai-core/training/train_sentiment.py                                ║
║                                                                              ║
║  Purpose : Fine-tune CAMeLBERT-Mix on Arabic sentiment data in two phases:  ║
║                                                                              ║
║  Phase 1 — Base pre-training                                                 ║
║    Input : sentiment_base_train.npz  (HTL + RES + PROD = ~26K reviews)      ║
║    Output: models/chatbot/sentiment_base/                                    ║
║    Goal  : Teach the model Egyptian Arabic language, negation, intensifiers  ║
║                                                                              ║
║  Phase 2 — Medical fine-tune                                                 ║
║    Input : sentiment_medical_train.npz  (500-1000 clinical sentences)        ║
║    Output: models/chatbot/sentiment_medical/ → converted to ONNX INT8        ║
║    Goal  : Specialise for patient distress vs reassurance classification     ║
║                                                                              ║
║  Final output                                                                ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ai-core/models/chatbot/model_int8.onnx   ← used by SentimentAnalyzer       ║
║  ai-core/models/chatbot/tokenizer_config.json                               ║
║                                                                              ║
║  Usage                                                                       ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  python train_sentiment.py                     # full pipeline (phase 1+2)  ║
║  python train_sentiment.py --phase 1           # base only                  ║
║  python train_sentiment.py --phase 2           # medical fine-tune only     ║
║  python train_sentiment.py --evaluate          # evaluate on test set       ║
║  python train_sentiment.py --export-onnx       # export INT8 ONNX only      ║
║                                                                              ║
║  Author  : Goda Emad  (AI Core)                                              ║
║  Version : 1.0.0                                                             ║
║  Updated : 2026-03-18                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_sentiment")

# ── Optional heavy imports — checked at runtime ───────────────────────────
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — training unavailable")

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    from datasets import Dataset as HFDataset
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    logger.warning("HuggingFace Transformers not installed")

try:
    from sklearn.metrics import (
        accuracy_score, f1_score,
        classification_report, confusion_matrix,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
#  Paths & constants
# ═══════════════════════════════════════════════════════════════════════════

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
DATA_PROC   = REPO_ROOT / "data" / "processed"
MODELS_DIR  = REPO_ROOT / "ai-core" / "models" / "chatbot"

# CAMeLBERT-Mix — best available model for Egyptian Arabic sentiment
# Trained on MSA + dialectal Arabic including Egyptian
BASE_MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"

# Fallback if CAMeLBERT unavailable (lighter, still Arabic-aware)
FALLBACK_MODEL  = "aubmindlab/bert-base-arabertv02"

LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {0: "negative", 1: "positive"}

# ═══════════════════════════════════════════════════════════════════════════
#  Training hyperparameters
# ═══════════════════════════════════════════════════════════════════════════

PHASE1_ARGS = dict(
    num_train_epochs          = 3,
    per_device_train_batch_size = 32,
    per_device_eval_batch_size  = 64,
    learning_rate             = 2e-5,
    warmup_ratio              = 0.1,
    weight_decay              = 0.01,
    lr_scheduler_type         = "cosine",
    evaluation_strategy       = "epoch",
    save_strategy             = "epoch",
    load_best_model_at_end    = True,
    metric_for_best_model     = "f1",
    greater_is_better         = True,
    fp16                      = True,   # uses half precision if GPU available
    dataloader_num_workers    = 2,
    logging_steps             = 100,
    report_to                 = "none", # no WandB/MLflow needed
)

PHASE2_ARGS = dict(
    num_train_epochs          = 5,      # more epochs — smaller dataset
    per_device_train_batch_size = 16,   # smaller batch — less data
    per_device_eval_batch_size  = 32,
    learning_rate             = 5e-6,   # lower LR — fine-tune, not scratch
    warmup_ratio              = 0.15,
    weight_decay              = 0.01,
    lr_scheduler_type         = "cosine",
    evaluation_strategy       = "epoch",
    save_strategy             = "epoch",
    load_best_model_at_end    = True,
    metric_for_best_model     = "f1",
    greater_is_better         = True,
    fp16                      = True,
    dataloader_num_workers    = 2,
    logging_steps             = 10,
    report_to                 = "none",
)

MAX_LENGTH = 128   # tokens — enough for medical sentences, faster than 256


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset wrapper
# ═══════════════════════════════════════════════════════════════════════════

class SentimentDataset:
    """
    Loads a .npz file produced by prepare_sentiment_data.py and wraps it
    as a HuggingFace Dataset for the Trainer API.
    """

    def __init__(self, npz_path: Path, tokenizer, max_length: int = MAX_LENGTH):
        data       = np.load(npz_path, allow_pickle=True)
        self.texts  = data["texts"].tolist()
        self.labels = data["labels"].tolist()

        logger.info(
            "Loaded %s | rows=%d  pos=%d  neg=%d",
            npz_path.name, len(self.texts),
            sum(1 for l in self.labels if l == 1),
            sum(1 for l in self.labels if l == 0),
        )

        # Tokenise all at once (faster than per-sample)
        encodings = tokenizer(
            self.texts,
            truncation   = True,
            padding      = True,
            max_length   = max_length,
            return_tensors = "pt",
        )

        self.hf_dataset = HFDataset.from_dict({
            "input_ids"     : encodings["input_ids"].tolist(),
            "attention_mask": encodings["attention_mask"].tolist(),
            "labels"        : self.labels,
        })

    def __len__(self) -> int:
        return len(self.texts)


# ═══════════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(eval_pred) -> dict:
    """HuggingFace Trainer metrics function — returns accuracy + weighted F1."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc   = accuracy_score(labels, preds)
    f1    = f1_score(labels, preds, average="weighted")
    return {"accuracy": round(acc, 4), "f1": round(f1, 4)}


# ═══════════════════════════════════════════════════════════════════════════
#  ONNX export
# ═══════════════════════════════════════════════════════════════════════════

def export_to_onnx_int8(model_dir: Path, output_dir: Path) -> Path:
    """
    Export a fine-tuned model to ONNX INT8 format for offline inference.

    Steps
    ─────
    1. Load fine-tuned PyTorch model
    2. Export to ONNX FP32
    3. Quantise to INT8 (dynamic quantisation — no calibration data needed)
    4. Save to output_dir/model_int8.onnx

    INT8 reduces model size ~4× and inference latency ~2× with <1% accuracy loss.
    """
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        from optimum.onnxruntime import ORTQuantizer
    except ImportError:
        logger.error(
            "optimum[onnxruntime] not installed.\n"
            "  pip install optimum[onnxruntime]"
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting to ONNX FP32...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        str(model_dir), export=True
    )
    ort_model.save_pretrained(str(output_dir))

    logger.info("Quantising to INT8...")
    quantizer = ORTQuantizer.from_pretrained(output_dir)

    # FIX v1.1 — Hardware-agnostic quantisation config
    # avx512_vnni is fastest on modern Intel CPUs but crashes on ARM/AMD/Mac.
    # We auto-detect the platform and pick the right config so model_int8.onnx
    # runs on any hardware — Vercel serverless, AWS Graviton, Mac M-series, etc.
    import platform
    machine = platform.machine().lower()
    system  = platform.system().lower()

    if "arm" in machine or "aarch64" in machine or system == "darwin":
        # ARM (AWS Graviton, Mac M-series, Raspberry Pi)
        logger.info("ARM/Apple Silicon detected → using arm64 quantisation config")
        qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
    elif "x86" in machine or "amd64" in machine or "x86_64" in machine:
        # x86 — check for AVX512 support before using avx512_vnni
        try:
            import cpuinfo
            flags = cpuinfo.get_cpu_info().get("flags", [])
            has_avx512 = "avx512vnni" in flags or "avx512f" in flags
        except ImportError:
            has_avx512 = False   # py-cpuinfo not installed — safe fallback

        if has_avx512:
            logger.info("Intel AVX512 detected → using avx512_vnni quantisation config")
            qconfig = AutoQuantizationConfig.avx512_vnni(
                is_static=False, per_channel=False
            )
        else:
            logger.info("x86 without AVX512 → using avx2 quantisation config")
            qconfig = AutoQuantizationConfig.avx2(
                is_static=False, per_channel=False
            )
    else:
        # Unknown / generic — safest option, works everywhere
        logger.info("Unknown platform (%s %s) → using generic ARM64 config", system, machine)
        qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)

    quantizer.quantize(
        save_dir            = str(output_dir),
        quantization_config = qconfig,
    )

    onnx_path = output_dir / "model_int8.onnx"
    logger.info("ONNX INT8 saved → %s", onnx_path)
    return onnx_path


# ═══════════════════════════════════════════════════════════════════════════
#  Trainer
# ═══════════════════════════════════════════════════════════════════════════

class SentimentTrainer:
    """
    Two-phase sentiment trainer for RIVA.

    Phase 1: pre-train on large Arabic review datasets (HTL + RES + PROD)
    Phase 2: fine-tune on small medical Egyptian Arabic dataset
    """

    def __init__(
        self,
        base_model : str  = BASE_MODEL_NAME,
        models_dir : Path = MODELS_DIR,
        data_dir   : Path = DATA_PROC,
        max_length : int  = MAX_LENGTH,
    ) -> None:
        if not _TORCH_AVAILABLE or not _HF_AVAILABLE:
            raise RuntimeError(
                "PyTorch and HuggingFace Transformers are required.\n"
                "  pip install torch transformers datasets optimum[onnxruntime]"
            )

        self._base_model  = base_model
        self._models_dir  = models_dir
        self._data_dir    = data_dir
        self._max_length  = max_length

        # Detect device
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Device: %s", self._device)
        if self._device == "cpu":
            logger.warning(
                "No GPU detected — training will be slow.\n"
                "  Consider Google Colab (free T4 GPU) for Phase 1."
            )

        # Load tokenizer once — shared across both phases
        logger.info("Loading tokenizer: %s", base_model)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(base_model)
        except Exception:
            logger.warning(
                "Could not load %s — falling back to %s", base_model, FALLBACK_MODEL
            )
            self._tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
            self._base_model = FALLBACK_MODEL

    # ── Public API ───────────────────────────────────────────────────────

    def train_phase1(self) -> Path:
        """
        Phase 1: Pre-train on large Arabic review datasets.

        Input  : data/processed/sentiment_base_train.npz
                 data/processed/sentiment_base_val.npz
        Output : ai-core/models/chatbot/sentiment_base/
        """
        train_npz = self._data_dir / "sentiment_base_train.npz"
        val_npz   = self._data_dir / "sentiment_base_val.npz"
        output    = self._models_dir / "sentiment_base"

        self._check_data_exists(train_npz, "Phase 1 train")
        self._check_data_exists(val_npz,   "Phase 1 val")

        logger.info("═" * 55)
        logger.info("  PHASE 1 — Base pre-training")
        logger.info("  Model  : %s", self._base_model)
        logger.info("  Output : %s", output)
        logger.info("═" * 55)

        train_data = SentimentDataset(train_npz, self._tokenizer, self._max_length)
        val_data   = SentimentDataset(val_npz,   self._tokenizer, self._max_length)

        model = AutoModelForSequenceClassification.from_pretrained(
            self._base_model,
            num_labels = 2,
            id2label   = ID2LABEL,
            label2id   = LABEL2ID,
        )

        args = TrainingArguments(
            output_dir = str(output),
            **PHASE1_ARGS,
            fp16 = (self._device == "cuda"),   # FP16 only on GPU
        )

        trainer = Trainer(
            model            = model,
            args             = args,
            train_dataset    = train_data.hf_dataset,
            eval_dataset     = val_data.hf_dataset,
            compute_metrics  = compute_metrics,
            callbacks        = [EarlyStoppingCallback(early_stopping_patience=2)],
        )

        t_start = time.time()
        trainer.train()
        elapsed = (time.time() - t_start) / 60
        logger.info("Phase 1 training complete in %.1f minutes", elapsed)

        trainer.save_model(str(output))
        self._tokenizer.save_pretrained(str(output))
        self._save_training_report(trainer, output, "phase1")

        logger.info("Phase 1 model saved → %s", output)
        return output

    def train_phase2(self, base_model_dir: Path | None = None) -> Path:
        """
        Phase 2: Fine-tune on medical Egyptian Arabic data.

        Input  : data/processed/sentiment_medical_train.npz
                 data/processed/sentiment_medical_val.npz
        Output : ai-core/models/chatbot/sentiment_medical/
                 ai-core/models/chatbot/model_int8.onnx   ← final offline model
        """
        train_npz = self._data_dir / "sentiment_medical_train.npz"
        val_npz   = self._data_dir / "sentiment_medical_val.npz"
        output    = self._models_dir / "sentiment_medical"

        self._check_data_exists(train_npz, "Phase 2 train")
        self._check_data_exists(val_npz,   "Phase 2 val")

        # Start from Phase 1 model if available, otherwise base model
        start_model = str(
            base_model_dir
            if base_model_dir and base_model_dir.exists()
            else self._base_model
        )
        logger.info("═" * 55)
        logger.info("  PHASE 2 — Medical fine-tune")
        logger.info("  Starting from : %s", start_model)
        logger.info("  Output        : %s", output)
        logger.info("═" * 55)

        train_data = SentimentDataset(train_npz, self._tokenizer, self._max_length)
        val_data   = SentimentDataset(val_npz,   self._tokenizer, self._max_length)

        model = AutoModelForSequenceClassification.from_pretrained(
            start_model,
            num_labels = 2,
            id2label   = ID2LABEL,
            label2id   = LABEL2ID,
            ignore_mismatched_sizes = True,
        )

        args = TrainingArguments(
            output_dir = str(output),
            **PHASE2_ARGS,
            fp16 = (self._device == "cuda"),
        )

        trainer = Trainer(
            model           = model,
            args            = args,
            train_dataset   = train_data.hf_dataset,
            eval_dataset    = val_data.hf_dataset,
            compute_metrics = compute_metrics,
            callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)],
        )

        t_start = time.time()
        trainer.train()
        elapsed = (time.time() - t_start) / 60
        logger.info("Phase 2 training complete in %.1f minutes", elapsed)

        trainer.save_model(str(output))
        self._tokenizer.save_pretrained(str(output))
        self._save_training_report(trainer, output, "phase2")

        # Export to ONNX INT8 → final model for SentimentAnalyzer
        onnx_out = self._models_dir
        logger.info("Exporting final model to ONNX INT8...")
        onnx_path = export_to_onnx_int8(output, onnx_out)
        if onnx_path:
            logger.info("Final ONNX model → %s", onnx_path)

        return output

    def evaluate(self, model_dir: Path | None = None) -> dict:
        """
        Evaluate model on test sets and print a full classification report.

        Tests both:
          - sentiment_base_test.npz  (general Arabic)
          - sentiment_medical_test.npz  (clinical — if available)
        """
        if not _SKLEARN_AVAILABLE:
            logger.error("scikit-learn required for evaluation")
            return {}

        eval_dir = model_dir or (self._models_dir / "sentiment_medical")
        if not eval_dir.exists():
            eval_dir = self._models_dir / "sentiment_base"
        if not eval_dir.exists():
            logger.error("No trained model found at %s", eval_dir)
            return {}

        logger.info("Loading model from %s", eval_dir)
        model     = AutoModelForSequenceClassification.from_pretrained(str(eval_dir))
        model.to(self._device)   # FIX v1.1: move model to GPU/CPU before inference
        tokenizer = AutoTokenizer.from_pretrained(str(eval_dir))
        model.eval()

        results = {}
        for prefix in ("sentiment_base", "sentiment_medical"):
            test_npz = self._data_dir / f"{prefix}_test.npz"
            if not test_npz.exists():
                continue

            logger.info("Evaluating on %s", test_npz.name)
            data    = np.load(test_npz, allow_pickle=True)
            texts   = data["texts"].tolist()
            labels  = data["labels"].tolist()

            preds = self._predict_batch(model, tokenizer, texts)

            acc = accuracy_score(labels, preds)
            f1  = f1_score(labels, preds, average="weighted")

            logger.info("━" * 50)
            logger.info("  %s", prefix)
            logger.info("  Accuracy : %.4f", acc)
            logger.info("  F1       : %.4f", f1)
            logger.info("  Report:")
            logger.info("\n%s", classification_report(
                labels, preds, target_names=["negative", "positive"]
            ))
            logger.info("  Confusion Matrix:")
            logger.info("\n%s", confusion_matrix(labels, preds))

            results[prefix] = {"accuracy": acc, "f1": f1}

        return results

    # ── Private helpers ──────────────────────────────────────────────────

    def _predict_batch(self, model, tokenizer, texts: list[str], batch_size: int = 64) -> list[int]:
        """
        Run inference in batches — returns list of predicted label indices.

        FIX v1.1 — Device consistency
        ───────────────────────────────
        enc tensors are moved to model.device before inference.
        Without this, GPU-loaded models raise:
          RuntimeError: Expected all tensors to be on the same device
        """
        all_preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc   = tokenizer(
                batch, truncation=True, padding=True,
                max_length=self._max_length, return_tensors="pt",
            )
            # FIX: move all input tensors to the same device as the model
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
            preds = torch.argmax(logits, dim=-1).tolist()
            all_preds.extend(preds)
        return all_preds

    @staticmethod
    def _check_data_exists(path: Path, label: str) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} data not found: {path}\n"
                f"  Run: python prepare_sentiment_data.py --phase 1"
            )

    @staticmethod
    def _save_training_report(trainer, output_dir: Path, phase: str) -> None:
        """Save training loss history and final metrics to a JSON report."""
        history = trainer.state.log_history
        report  = {
            "phase"        : phase,
            "best_metric"  : trainer.state.best_metric,
            "epochs_run"   : trainer.state.epoch,
            "log_history"  : history,
        }
        report_path = output_dir / f"training_report_{phase}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("Training report → %s", report_path)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train RIVA sentiment model (CAMeLBERT → ONNX INT8)"
    )
    p.add_argument(
        "--phase", type=int, choices=[1, 2], default=None,
        help="Run phase 1 (base) or phase 2 (medical). Default: both.",
    )
    p.add_argument(
        "--evaluate", action="store_true",
        help="Evaluate trained model on test sets.",
    )
    p.add_argument(
        "--export-onnx", action="store_true",
        help="Export existing fine-tuned model to ONNX INT8 only.",
    )
    p.add_argument(
        "--base-model", type=str, default=BASE_MODEL_NAME,
        help=f"HuggingFace model name (default: {BASE_MODEL_NAME})",
    )
    p.add_argument(
        "--max-length", type=int, default=MAX_LENGTH,
        help=f"Max token length (default: {MAX_LENGTH})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    trainer = SentimentTrainer(
        base_model = args.base_model,
        max_length = args.max_length,
    )

    # ── Export only ───────────────────────────────────────────────────────
    if args.export_onnx:
        medical_dir = MODELS_DIR / "sentiment_medical"
        base_dir    = MODELS_DIR / "sentiment_base"
        src = medical_dir if medical_dir.exists() else base_dir
        export_to_onnx_int8(src, MODELS_DIR)
        return

    # ── Evaluate only ─────────────────────────────────────────────────────
    if args.evaluate:
        trainer.evaluate()
        return

    # ── Training pipeline ─────────────────────────────────────────────────
    phase1_output = None

    if args.phase in (None, 1):
        phase1_output = trainer.train_phase1()

    if args.phase in (None, 2):
        trainer.train_phase2(base_model_dir=phase1_output)

    logger.info("═" * 55)
    logger.info("  Training complete ✅")
    logger.info("  Final model : %s/model_int8.onnx", MODELS_DIR)
    logger.info("  Next step   : Update SentimentAnalyzer to use ONNX model")
    logger.info("═" * 55)


if __name__ == "__main__":
    main()
