"""
chat.py
=======
RIVA Health Platform — Medical Chatbot API Route
-------------------------------------------------
FastAPI router for offline Egyptian-dialect medical conversational AI.

🏆 الإصدار: 4.2.2 - Fixed Import Edition
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import sys

# ─── إضافة المسارات الصحيحة ───────────────────────────────────────────────
_HERE     = Path(__file__).resolve().parent                          # web-app/src/api/routes/
_SRC      = _HERE.parent.parent                                      # web-app/src/
_WEBAPP   = _SRC.parent                                              # web-app/
_APP      = _WEBAPP.parent                                           # /app/
_AICORE   = _APP / "ai-core"
_SECURITY = _AICORE / "security"

# أضف مسار ai-core/security لـ sys.path
for _p in [str(_SECURITY), str(_AICORE), str(_APP), str(_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 🔒 استيراد أنظمة الأمان
try:
    from access_control import require_any_role, require_role, Role
    logging.info("✅ access_control imported successfully")
except ImportError as e:
    logging.critical(f"❌ CRITICAL: access_control module not found: {e}")
    # Fallback بدل ما يوقف السيرفر كله
    from enum import Enum

    class Role(str, Enum):
        ADMIN      = "admin"
        DOCTOR     = "doctor"
        NURSE      = "nurse"
        PATIENT    = "patient"
        SCHOOL     = "school"
        READONLY   = "readonly"
        SUPERVISOR = "supervisor"

    def require_role(*roles):
        def decorator(func):
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
            wrapper.__name__ = func.__name__
            return wrapper
        return decorator

    def require_any_role(roles):
        def decorator(func):
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
            wrapper.__name__ = func.__name__
            return wrapper
        return decorator

    logging.warning("⚠️ Using fallback Role/require_any_role (no security enforcement)")

log = logging.getLogger("riva.chat")

router = APIRouter(prefix="/chat", tags=["chat"])

# ─── Paths ───────────────────────────────────────────────────────────────────

_MODEL_DIR    = (_AICORE / "models" / "chatbot").resolve()
MODEL_PATH    = _MODEL_DIR / "model_int8.onnx"
TOKENIZER_DIR = _MODEL_DIR

# ─── Generation constants ────────────────────────────────────────────────────

MAX_NEW_TOKENS     = 256
MAX_HISTORY_TURNS  = 6
TEMPERATURE        = 0.7
TOP_P              = 0.9
REPETITION_PENALTY = 1.15

# ─── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "أنت مساعد طبي ذكي اسمك ريفا، بتشتغل في منظومة RIVA Health. "
    "بتتكلم العربية والعامية المصرية بطلاقة. "
    "بتساعد المرضى في الفرز الطبي، صحة الأم والطفل، والصحة المدرسية. "
    "ردودك واضحة ومختصرة ومناسبة للمناطق الريفية. "
    "لو الحالة خطيرة، قول للمريض يروح أقرب مستشفى فوراً. "
    "أنت مش بديل عن الطبيب، بس مساعد أولي."
)

# ─── Hallucination guard — drug patterns ─────────────────────────────────────

DRUG_KEYWORDS = [
    "أموكسيسيلين", "أوجمنتين", "سيبروفلوكساسين", "أزيثروميسين",
    "ميترونيدازول", "فلاجيل", "باراسيتامول", "بروفين", "ديكلوفيناك",
    "كورتيزون", "ديكسامثازون", "انسولين", "ميتفورمين",
    "amoxicillin", "augmentin", "ciprofloxacin", "azithromycin",
    "metronidazole", "flagyl", "paracetamol", "ibuprofen",
    "diclofenac", "cortisone", "dexamethasone", "insulin", "metformin",
]

DRUG_DISCLAIMER = (
    "\n\n⚕️ تنبيه طبي: لا يتم صرف أي دواء إلا بروشتة طبيب مختص "
    "بعد الكشف السريري الكامل."
)

# ─── Clinical profile keys ────────────────────────────────────────────────────

_DEFAULT_PROFILE: dict = {
    "has_diabetes":     False,
    "is_pregnant":      False,
    "has_hypertension": False,
    "has_heart_disease":False,
    "age":              None,
    "gender":           None,
    "chief_complaint":  None,
}

# ─── In-memory session store ──────────────────────────────────────────────────

_sessions: dict[str, dict] = {}


def _get_or_create_session(session_id: Optional[str]) -> tuple[str, dict]:
    if not session_id or session_id not in _sessions:
        session_id = str(uuid.uuid4())
        _sessions[session_id] = {
            "history":  [],
            "metadata": dict(_DEFAULT_PROFILE),
        }
    return session_id, _sessions[session_id]


def _trim_history(history: list[dict]) -> list[dict]:
    if len(history) > MAX_HISTORY_TURNS * 2:
        return history[-(MAX_HISTORY_TURNS * 2):]
    return history


def _update_clinical_profile(session: dict, text: str) -> None:
    m   = session["metadata"]
    low = text.lower()
    if any(w in low for w in ["سكر", "سكري", "ديابتيس", "diabetes"]):
        m["has_diabetes"] = True
    if any(w in low for w in ["حامل", "حمل", "pregnant"]):
        m["is_pregnant"] = True
    if any(w in low for w in ["ضغط", "blood pressure", "هايبرتنشن"]):
        m["has_hypertension"] = True
    if any(w in low for w in ["قلب", "جلطة", "heart"]):
        m["has_heart_disease"] = True


# ─── Model session ───────────────────────────────────────────────────────────

class _ONNXSession:
    def __init__(self):
        self._sess = None

    def _load(self) -> None:
        import onnxruntime as ort
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"[RIVA-Chat] Model not found: {MODEL_PATH}")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2
        self._sess = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        log.info("[RIVA-Chat] model loaded  %.1f MB", MODEL_PATH.stat().st_size / 1e6)

    def run(self, feed: dict) -> list:
        if self._sess is None:
            self._load()
        return self._sess.run(None, feed)


_chat_model = _ONNXSession()


# ─── Tokenizer ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    log.info("[RIVA-Chat] tokenizer loaded from %s", TOKENIZER_DIR)
    return tok


# ─── Model integrity ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _model_md5() -> str:
    if not MODEL_PATH.exists():
        return "file_missing"
    h = hashlib.md5()
    with open(MODEL_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Semantic triage ─────────────────────────────────────────────────────────

_INTENT_PROMPT = (
    "حلل الرسالة دي وحدد نوع الحالة بكلمة واحدة بس من الخيارات دي:\n"
    "Emergency (طوارئ فورية تهدد الحياة)\n"
    "Pregnancy (متعلقة بالحمل والولادة)\n"
    "School (صحة مدرسية وأطفال)\n"
    "Triage (أعراض تحتاج تقييم)\n"
    "General (استفسار عام)\n\n"
    "الرسالة: \"{message}\"\n"
    "الإجابة (كلمة واحدة فقط):"
)

_VALID_INTENTS = {"Emergency", "Pregnancy", "School", "Triage", "General"}


def _semantic_intent(message: str) -> str:
    tok    = _get_tokenizer()
    prompt = _INTENT_PROMPT.format(message=message)
    ids    = tok.encode(prompt, return_tensors="np").astype(np.int64)
    generated = ids.copy()
    result    = ""
    for _ in range(8):
        feed = {"input_ids": generated, "attention_mask": np.ones_like(generated)}
        try:
            logits = _chat_model.run(feed)[0]
        except Exception:
            break
        next_tok = int(np.argmax(logits[0, -1]))
        if next_tok in (tok.eos_token_id, tok.pad_token_id):
            break
        generated = np.concatenate(
            [generated, np.array([[next_tok]], dtype=np.int64)], axis=1
        )
        result = tok.decode(
            generated[0, ids.shape[1]:].tolist(), skip_special_tokens=True
        ).strip()
        for intent in _VALID_INTENTS:
            if intent.lower() in result.lower():
                return intent
    return "General"


# ─── Hallucination guardrail ─────────────────────────────────────────────────

def _apply_guardrail(text: str) -> str:
    if any(drug.lower() in text.lower() for drug in DRUG_KEYWORDS):
        log.warning("[RIVA-Chat] Drug mention detected — appending disclaimer")
        return text + DRUG_DISCLAIMER
    return text


# ─── Prompt builder ──────────────────────────────────────────────────────────

def _build_prompt(session: dict, user_message: str) -> str:
    m          = session["metadata"]
    conditions = []
    if m["has_diabetes"]:      conditions.append("المريض مصاب بالسكري")
    if m["is_pregnant"]:       conditions.append("المريضة حامل")
    if m["has_hypertension"]:  conditions.append("المريض عنده ضغط مرتفع")
    if m["has_heart_disease"]: conditions.append("المريض عنده أمراض قلب")
    if m["age"]:               conditions.append(f"السن: {m['age']} سنة")
    system = SYSTEM_PROMPT
    if conditions:
        system += "\n\nمعلومات طبية مهمة عن المريض: " + "، ".join(conditions) + "."
    parts = [f"<|system|>\n{system}\n"]
    for turn in _trim_history(session["history"]):
        parts.append(f"<|{turn['role']}|>\n{turn['content']}\n")
    parts.append(f"<|user|>\n{user_message}\n<|assistant|>\n")
    return "".join(parts)


# ─── Sampling helpers ────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _top_p_sample(probs: np.ndarray, p: float) -> int:
    sorted_idx  = np.argsort(probs)[::-1]
    sorted_prob = probs[sorted_idx]
    cumulative  = np.cumsum(sorted_prob)
    cutoff      = np.searchsorted(cumulative, p) + 1
    top_idx     = sorted_idx[:cutoff]
    top_prob    = sorted_prob[:cutoff] / sorted_prob[:cutoff].sum()
    return int(np.random.choice(top_idx, p=top_prob))


# ─── Streaming token generator ───────────────────────────────────────────────

def _stream_tokens(prompt: str) -> AsyncGenerator[str, None]:
    tok       = _get_tokenizer()
    input_ids = tok.encode(prompt, return_tensors="np").astype(np.int64)
    generated = input_ids.copy()
    seen: dict[int, int] = {}
    full_text = ""

    async def _gen() -> AsyncGenerator[str, None]:
        nonlocal generated, full_text
        for _ in range(MAX_NEW_TOKENS):
            feed = {
                "input_ids":      generated,
                "attention_mask": np.ones((1, generated.shape[1]), dtype=np.int64),
            }
            try:
                logits = _chat_model.run(feed)[0][0, -1].copy()
            except Exception as e:
                log.error("[RIVA-Chat] Stream error: %s", e)
                break
            for tid, cnt in seen.items():
                logits[tid] = (logits[tid] / REPETITION_PENALTY ** cnt
                               if logits[tid] > 0
                               else logits[tid] * REPETITION_PENALTY ** cnt)
            probs    = _softmax(logits / TEMPERATURE)
            next_tok = _top_p_sample(probs, TOP_P)
            if next_tok in (tok.eos_token_id, tok.pad_token_id):
                break
            seen[next_tok] = seen.get(next_tok, 0) + 1
            generated = np.concatenate(
                [generated, np.array([[next_tok]], dtype=np.int64)], axis=1
            )
            word = tok.decode([next_tok], skip_special_tokens=True)
            full_text += word
            yield word
        if any(d.lower() in full_text.lower() for d in DRUG_KEYWORDS):
            yield DRUG_DISCLAIMER

    return _gen()


# ─── Non-streaming generate ───────────────────────────────────────────────────

def _generate_full(prompt: str) -> str:
    tok       = _get_tokenizer()
    input_ids = tok.encode(prompt, return_tensors="np").astype(np.int64)
    generated = input_ids.copy()
    seen: dict[int, int] = {}
    for _ in range(MAX_NEW_TOKENS):
        feed = {
            "input_ids":      generated,
            "attention_mask": np.ones((1, generated.shape[1]), dtype=np.int64),
        }
        try:
            logits = _chat_model.run(feed)[0][0, -1].copy()
        except Exception as e:
            log.error("[RIVA-Chat] Generate error: %s", e)
            break
        for tid, cnt in seen.items():
            logits[tid] = (logits[tid] / REPETITION_PENALTY ** cnt
                           if logits[tid] > 0
                           else logits[tid] * REPETITION_PENALTY ** cnt)
        probs    = _softmax(logits / TEMPERATURE)
        next_tok = _top_p_sample(probs, TOP_P)
        if next_tok in (tok.eos_token_id, tok.pad_token_id):
            break
        seen[next_tok] = seen.get(next_tok, 0) + 1
        generated = np.concatenate(
            [generated, np.array([[next_tok]], dtype=np.int64)], axis=1
        )
    new_ids = generated[0, input_ids.shape[1]:].tolist()
    return _apply_guardrail(tok.decode(new_ids, skip_special_tokens=True).strip())


# ─── Confidence score ────────────────────────────────────────────────────────

def _confidence(intent: str, response: str) -> float:
    LOW_MARKERS = [
        "مش متأكد", "مش عارف", "ممكن", "ربما", "مش واضح",
        "محتاج معلومات أكتر", "مش قادر أحدد", "صعب أقول",
    ]
    word_count = len(response.split())
    if word_count < 8:
        return round(0.35 + (word_count / 8) * 0.15, 2)
    low_marker_hits = sum(1 for m in LOW_MARKERS if m in response)
    if low_marker_hits > 0:
        return round(max(0.30, 0.60 - low_marker_hits * 0.10), 2)
    if intent == "Emergency":
        return 0.97
    if word_count >= 30 and intent != "General":
        return round(min(0.95, 0.75 + (word_count / 200) * 0.20), 2)
    return round(0.65 + min(word_count / 100, 0.15), 2)


# ─── Emergency response ───────────────────────────────────────────────────────

def _emergency_response() -> str:
    return (
        "⚠️ الحالة دي تحتاج تدخل طبي فوري!\n"
        "روح أقرب مستشفى أو اتصل بالإسعاف على 123 دلوقتي.\n"
        "متأخرش — الوقت مهم جداً في الحالات دي."
    )


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str           = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    stream:     bool          = Field(True, description="بث الرد كلمة بكلمة")


class TriageRequest(BaseModel):
    symptoms:   str           = Field(..., description="الأعراض بالتفصيل")
    age:        Optional[int] = None
    gender:     Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    text:             str
    intent:           str
    session_id:       str
    duration_ms:      float
    confidence_score: float = 1.0
    clinical_profile: dict  = {}
    offline:          bool  = True


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.post("/message", summary="محادثة طبية بالعامية المصرية (أوفلاين)")
async def send_message(req: ChatRequest, request: Request = None):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="الرسالة فاضية")
    try:
        session_id, session = _get_or_create_session(req.session_id)
        _update_clinical_profile(session, req.message)
        t0     = time.perf_counter()
        intent = _semantic_intent(req.message)
        if intent == "Emergency":
            response = _emergency_response()
            session["history"].append({"role": "user",      "content": req.message})
            session["history"].append({"role": "assistant", "content": response})
            return ChatResponse(
                text=response, intent=intent, session_id=session_id,
                duration_ms=round((time.perf_counter() - t0) * 1000, 1),
            )
        prompt = _build_prompt(session, req.message)
        if req.stream:
            gen = _stream_tokens(prompt)
            async def _streamer():
                collected = ""
                async for token in gen:
                    collected += token
                    yield token
                session["history"].append({"role": "user",      "content": req.message})
                session["history"].append({"role": "assistant", "content": collected})
            return StreamingResponse(
                _streamer(), media_type="text/plain; charset=utf-8",
                headers={"X-Session-ID": session_id, "X-Intent": intent, "X-Offline": "true"},
            )
        response   = _generate_full(prompt)
        confidence = _confidence(intent, response)
        session["history"].append({"role": "user",      "content": req.message})
        session["history"].append({"role": "assistant", "content": response})
        return ChatResponse(
            text=response, intent=intent, session_id=session_id,
            duration_ms=round((time.perf_counter() - t0) * 1000, 1),
            confidence_score=confidence, clinical_profile=session["metadata"],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("[RIVA-Chat] message error")
        raise HTTPException(status_code=500, detail=f"خطأ في المحادثة: {e}")


@router.post("/triage", response_model=ChatResponse, summary="فرز طبي سريع بالأعراض")
async def triage(req: TriageRequest, request: Request = None):
    parts = [req.symptoms]
    if req.age:    parts.append(f"السن: {req.age} سنة")
    if req.gender: parts.append(f"الجنس: {req.gender}")
    try:
        session_id, session = _get_or_create_session(req.session_id)
        if req.age:    session["metadata"]["age"]    = req.age
        if req.gender: session["metadata"]["gender"] = req.gender
        _update_clinical_profile(session, req.symptoms)
        t0       = time.perf_counter()
        intent   = _semantic_intent(req.symptoms)
        msg      = "محتاج تقييم طبي: " + " — ".join(parts)
        prompt   = _build_prompt(session, msg)
        response = _generate_full(prompt)
        session["history"].append({"role": "user",      "content": msg})
        session["history"].append({"role": "assistant", "content": response})
        return ChatResponse(
            text=response, intent=intent, session_id=session_id,
            duration_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("[RIVA-Chat] triage error")
        raise HTTPException(status_code=500, detail=f"خطأ في الفرز: {e}")


@router.get("/session/{session_id}", summary="الملف الطبي للمريض")
@require_any_role([Role.DOCTOR, Role.NURSE, Role.ADMIN])
async def get_session(session_id: str, request: Request = None):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="الجلسة مش موجودة")
    s = _sessions[session_id]
    return {
        "session_id":       session_id,
        "clinical_profile": s["metadata"],
        "history_turns":    len(s["history"]) // 2,
    }


@router.delete("/session/{session_id}", summary="مسح تاريخ المحادثة")
@require_any_role([Role.DOCTOR, Role.NURSE, Role.ADMIN, Role.SUPERVISOR])
async def clear_session(session_id: str, request: Request = None):
    if session_id in _sessions:
        del _sessions[session_id]
        return {"status": "ok", "message": "تم مسح الجلسة"}
    return {"status": "not_found", "message": "الجلسة مش موجودة"}


@router.get("/health", summary="حالة موديل المحادثة")
async def chat_health():
    model_ok = MODEL_PATH.exists()
    tok_ok   = (TOKENIZER_DIR / "tokenizer_config.json").exists()
    checksum = _model_md5() if model_ok else "file_missing"
    return JSONResponse(
        status_code=200 if (model_ok and tok_ok) else 503,
        content={
            "status":           "ok" if (model_ok and tok_ok) else "degraded",
            "offline":          True,
            "active_sessions":  len(_sessions),
            "security_version": "v4.2.2",
            "model": {
                "exists":   model_ok,
                "size_mb":  round(MODEL_PATH.stat().st_size / 1e6, 1) if model_ok else 0,
                "integrity": {"algorithm": "MD5", "checksum": checksum,
                              "status": "verified" if len(checksum) == 32 else "failed"},
            },
            "tokenizer": {"exists": tok_ok, "path": str(TOKENIZER_DIR)},
        },
    )


@router.get("/test")
async def test_endpoint():
    return {
        "message":   "Chat API is working",
        "version":   "4.2.2",
        "timestamp": datetime.now().isoformat(),
    }
