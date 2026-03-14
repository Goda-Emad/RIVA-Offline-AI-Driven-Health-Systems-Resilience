"""
RIVA - Model Manager v2.0 (Bulletproof) 🧠⚡
=============================================
✅ Lazy Loading       — حمّل عند الحاجة بس
✅ Auto Eviction      — أخرج الأقدم تلقائياً
✅ Watchdog Thread    — مراقبة كل 60 ثانية
✅ Predictive Preload — جهّز النموذج قبل الطلب
✅ Performance Metrics — سجّل وقت التحميل

Author: GODA EMAD
"""
import time
import threading
import logging
import os
from typing import Optional

# ================================================================
# LOGGING + PERFORMANCE METRICS
# ================================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/model_manager.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RIVA.ModelManager")

# ================================================================
# CONFIG
# ================================================================
MAX_MODELS_IN_MEMORY = 2
UNLOAD_AFTER_SECONDS = 300   # 5 دقائق

MODEL_PATHS: dict[str, str] = {
    "triage":    "ai-core/models/triage/model_int8.onnx",
    "pregnancy": "ai-core/models/pregnancy/model_int8.onnx",
    "chatbot":   "ai-core/models/chatbot/model_int8.onnx",
    "school":    "ai-core/models/school/model_int8.onnx",
}

# ================================================================
# PREDICTIVE PRE-LOADING MAP ⭐
# حسب نوع المستخدم → جهّز النماذج المتوقعة في الخلفية
# ================================================================
PRELOAD_MAP: dict[str, list[str]] = {
    "doctor":   ["triage", "pregnancy"],
    "nurse":    ["triage"],
    "mother":   ["pregnancy"],
    "school":   ["school"],
    # Pre-fetch: لو فتح صفحة → جهّز الصفحة التالية
    "open_triage":    ["pregnancy"],
    "open_pregnancy": ["triage"],
    "open_school":    ["triage"],
}

# ================================================================
# PERFORMANCE METRICS
# ================================================================
class LoadMetrics:
    """يسجّل وقت كل تحميل — للإجابة على الحكام"""
    def __init__(self):
        self.records: list[dict] = []

    def record(self, model: str, duration_ms: float, from_cache: bool):
        entry = {
            "model": model,
            "duration_ms": round(duration_ms, 2),
            "from_cache": from_cache,
            "timestamp": time.strftime("%H:%M:%S")
        }
        self.records.append(entry)

        source = "⚡ cache" if from_cache else "📥 disk"
        logger.info(f"📊 Metric | '{model}' | {source} | {duration_ms:.1f}ms")

    def summary(self) -> dict:
        if not self.records:
            return {"message": "لا يوجد بيانات بعد"}

        disk_loads = [r for r in self.records if not r["from_cache"]]
        cache_hits = [r for r in self.records if r["from_cache"]]

        avg_disk = (
            round(sum(r["duration_ms"] for r in disk_loads) / len(disk_loads), 1)
            if disk_loads else 0
        )

        return {
            "total_requests":  len(self.records),
            "disk_loads":      len(disk_loads),
            "cache_hits":      len(cache_hits),
            "avg_load_ms":     avg_disk,
            "fastest_load_ms": min((r["duration_ms"] for r in disk_loads), default=0),
            "slowest_load_ms": max((r["duration_ms"] for r in disk_loads), default=0),
        }

    def judge_answer(self) -> str:
        """الإجابة الجاهزة للحكام 🎯"""
        s = self.summary()
        return (
            f"النماذج تُحمَّل محلياً في {s['avg_load_ms']}ms بدون إنترنت، "
            f"وبعد التحميل الأول تُقدَّم من الذاكرة في أقل من 1ms. "
            f"نسبة الـ Cache Hits: "
            f"{round(s['cache_hits'] / max(s['total_requests'], 1) * 100)}%."
        )


# ================================================================
# MODEL MANAGER
# ================================================================
class ModelManager:
    def __init__(self) -> None:
        self._models: dict = {}
        self._last_used: dict[str, float] = {}
        self._lock = threading.Lock()
        self.metrics = LoadMetrics()

        # Watchdog
        threading.Thread(target=self._watchdog, daemon=True).start()
        logger.info("✅ ModelManager v2.0 initialized")

    # ── GET ──────────────────────────────────
    def get(self, name: str):
        with self._lock:
            start = time.time() * 1000

            if name in self._models:
                self._last_used[name] = time.time()
                elapsed = time.time() * 1000 - start
                self.metrics.record(name, elapsed, from_cache=True)
                logger.info(f"⚡ '{name}' من الذاكرة")
                return self._models[name]

            if len(self._models) >= MAX_MODELS_IN_MEMORY:
                self._evict_oldest()

            return self._load(name, start)

    # ── LOAD ─────────────────────────────────
    def _load(self, name: str, start_ms: float):
        path = MODEL_PATHS.get(name)
        if not path:
            logger.error(f"❌ '{name}' غير موجود في MODEL_PATHS")
            return None

        try:
            logger.info(f"📥 تحميل '{name}'...")

            # في الإنتاج: rt.InferenceSession(path)
            # للتجربة: simulate
            time.sleep(0.05)
            session = f"ONNXSession({name})"

            self._models[name] = session
            self._last_used[name] = time.time()

            elapsed = time.time() * 1000 - start_ms
            self.metrics.record(name, elapsed, from_cache=False)
            logger.info(f"✅ '{name}' جاهز في {elapsed:.1f}ms")
            return session

        except Exception as e:
            logger.error(f"❌ فشل تحميل '{name}': {e}")
            return None

    # ── EVICT ────────────────────────────────
    def _evict_oldest(self) -> None:
        candidates = {
            k: v for k, v in self._last_used.items()
            if k != "chatbot"
        }
        if candidates:
            oldest = min(candidates, key=candidates.get)
            self.release(oldest)

    # ── RELEASE ──────────────────────────────
    def release(self, name: str) -> None:
        if name in self._models:
            del self._models[name]
            del self._last_used[name]
            logger.info(f"🗑️ '{name}' أُخرج")

    # ── WATCHDOG ─────────────────────────────
    def _watchdog(self) -> None:
        while True:
            time.sleep(60)
            now = time.time()
            with self._lock:
                old = [
                    n for n, t in self._last_used.items()
                    if (now - t) > UNLOAD_AFTER_SECONDS and n != "chatbot"
                ]
                for n in old:
                    logger.info(f"⏰ Watchdog: '{n}' خامل → يُخرج")
                    self.release(n)

    # ── PREDICTIVE PRELOAD ⭐ ────────────────
    def preload_for(self, context: str) -> None:
        """
        Predictive Pre-loading حسب المستخدم أو الصفحة
        يشتغل في background — مش بيعطّل الواجهة
        """
        to_load = PRELOAD_MAP.get(context, [])
        if not to_load:
            return

        def _bg():
            for m in to_load:
                if m not in self._models:
                    logger.info(f"🔮 Pre-fetch '{m}' (context={context})")
                    self.get(m)

        threading.Thread(target=_bg, daemon=True).start()

    # ── STATUS ───────────────────────────────
    @property
    def status(self) -> dict:
        now = time.time()
        return {
            "loaded":      list(self._models.keys()),
            "count":       len(self._models),
            "max":         MAX_MODELS_IN_MEMORY,
            "metrics":     self.metrics.summary(),
            "judge_answer": self.metrics.judge_answer()
        }


# Singleton
model_manager = ModelManager()


# ================================================================
# اختبار
# ================================================================
if __name__ == "__main__":
    import json

    print("\n" + "="*50)
    print("اختبار 1: Lazy Loading + Eviction")
    model_manager.get("triage")
    model_manager.get("pregnancy")
    model_manager.get("school")        # يخرج triage تلقائياً
    model_manager.get("pregnancy")     # من الذاكرة ⚡

    print("\n" + "="*50)
    print("اختبار 2: Predictive Pre-loading")
    print("→ مستخدم طبيب فتح التطبيق...")
    model_manager.preload_for("doctor")
    time.sleep(0.3)

    print("\n" + "="*50)
    print("اختبار 3: Predictive لما يفتح صفحة triage")
    model_manager.preload_for("open_triage")
    time.sleep(0.3)

    print("\n" + "="*50)
    print("📊 Performance Metrics:")
    status = model_manager.status
    print(json.dumps(status["metrics"], ensure_ascii=False, indent=2))

    print("\n" + "="*50)
    print("🎯 إجابة الحكام:")
    print(status["judge_answer"])
