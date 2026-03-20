"""
RIVA - Triage Classifier v2.0
==============================
واجهة ONNX Model مع Batch Inference
Single: ~3.9ms | Batch 50: ~7ms (0.14ms/patient)
Author: GODA EMAD
"""
import json
import pickle
import logging
import numpy as np
import pandas as pd
import onnxruntime as rt
from typing import Optional

logger = logging.getLogger("RIVA.TriageClassifier")

class TriageClassifier:
    """
    يحمّل ONNX Model ويعمل inference
    Single + Batch prediction
    """
    def __init__(
        self,
        model_path:    str = "ai-core/models/triage/model_int8.onnx",
        features_path: str = "ai-core/models/triage/features.json",
        imputer_path:  str = "ai-core/models/triage/imputer.pkl",
        scaler_path:   str = "ai-core/models/triage/scaler.pkl"
    ):
        # ONNX Session — Optimized
        opts = rt.SessionOptions()
        opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads     = 1
        opts.execution_mode           = rt.ExecutionMode.ORT_SEQUENTIAL

        self.sess     = rt.InferenceSession(
            model_path,
            sess_options = opts,
            providers    = ["CPUExecutionProvider"]
        )
        self.inp_name = self.sess.get_inputs()[0].name

        with open(imputer_path, "rb") as f: self.imputer = pickle.load(f)
        with open(scaler_path,  "rb") as f: self.scaler  = pickle.load(f)

        with open(features_path, encoding="utf-8") as f:
            data          = json.load(f)
            self.features = data["features"]
            self.classes  = data.get("classes", ["No Diabetes", "Diabetes"])
            self.metrics  = data.get("metrics", {})

        logger.info(f"TriageClassifier v2.0 loaded | features={len(self.features)}")

    # ── SINGLE PREDICT ───────────────────────────
    def predict(self, features: dict) -> dict:
        """
        يأخذ dict من الـ features ويرجع التنبؤ
        ~3.9ms per patient
        """
        import time
        start = time.time() * 1000

        feat_df  = pd.DataFrame([features], columns=self.features)
        feat_imp = self.imputer.transform(feat_df)
        feat_s   = self.scaler.transform(feat_imp).astype(np.float32)

        outputs  = self.sess.run(None, {self.inp_name: feat_s})
        pred     = int(outputs[0][0])
        proba    = outputs[1][0]
        conf     = float(proba[pred])

        ms = round(time.time() * 1000 - start, 2)
        logger.info(f"Single predict: {self.classes[pred]} | conf={conf:.3f} | {ms}ms")

        return {
            "prediction":   pred,
            "label":        self.classes[pred],
            "confidence":   round(conf, 3),
            "probabilities": {
                "no_diabetes": round(float(proba[0]), 3),
                "diabetes":    round(float(proba[1]), 3)
            },
            "inference_ms": ms
        }

    # ── BATCH PREDICT ────────────────────────────
    def predict_batch(self, features_list: list) -> dict:
        """
        يأخذ list من الـ features ويرجع list من النتائج
        ~7ms لكل 50 مريض = 0.14ms/patient
        28x أسرع من loop
        """
        import time
        start = time.time() * 1000

        df      = pd.DataFrame(features_list, columns=self.features)
        imp     = self.imputer.transform(df)
        sc      = self.scaler.transform(imp).astype(np.float32)
        outputs = self.sess.run(None, {self.inp_name: sc})
        preds   = outputs[0]
        probas  = outputs[1]

        results = [
            {
                "prediction":   int(preds[i]),
                "label":        self.classes[preds[i]],
                "confidence":   round(float(probas[i][preds[i]]), 3),
                "probabilities": {
                    "no_diabetes": round(float(probas[i][0]), 3),
                    "diabetes":    round(float(probas[i][1]), 3)
                }
            }
            for i in range(len(features_list))
        ]

        total_ms = round(time.time() * 1000 - start, 2)
        avg_ms   = round(total_ms / len(features_list), 3)

        logger.info(
            f"Batch predict: {len(features_list)} patients | "
            f"total={total_ms}ms | avg={avg_ms}ms"
        )

        return {
            "results":        results,
            "total":          len(results),
            "diabetes_count": sum(1 for r in results if r["prediction"]==1),
            "total_ms":       total_ms,
            "avg_ms":         avg_ms
        }

    # ── MODEL INFO ───────────────────────────────
    @property
    def model_info(self) -> dict:
        return {
            "features":   self.features,
            "classes":    self.classes,
            "metrics":    self.metrics,
            "input_name": self.inp_name,
            "performance": {
                "single_ms": "~3.9ms",
                "batch_50_ms": "~7ms",
                "avg_per_patient_ms": "~0.14ms"
            }
        }
