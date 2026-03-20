"""
orchestrator.py
===============
RIVA Health Platform — Voice-Chat Integration Bridge + Smart Router
--------------------------------------------------------------------
الجسر الحقيقي بين الصوت والشات مع التوجيه الذكي للـ 17 صفحة.

Pipeline:
    صوت / نص
        → transcribe_audio()   [voice.py]   — الأذن  (لو صوت)
        → chat()               [chat.py]    — العقل
        → _smart_route()                    — التوجيه الذكي
        → target_page                       — الصفحة الصح من الـ 17

الـ 17 صفحة:
    01_home.html              — الرئيسية
    02_chatbot.html           — الشات بوت
    03_triage.html            — الفرز الطبي
    04_result.html            — النتيجة + QR
    05_history.html           — السجل الطبي
    06_pregnancy.html         — داشبورد صحة الأم
    07_school.html            — داشبورد الصحة المدرسية
    08_offline.html           — صفحة بدون نت
    09_doctor_dashboard.html  — لوحة الدكتور
    10_mother_dashboard.html  — لوحة الأم
    11_school_dashboard.html  — لوحة المدرسة
    12_ai_explanation.html    — شرح قرارات الـ AI
    13_readmission.html       — خطر إعادة الإدخال
    14_los_dashboard.html     — مدة الإقامة بالمستشفى
    15_combined_dashboard.html— داشبورد موحد
    16_doctor_notes.html      — ملاحظات الدكتور
    17_sustainability.html    — الاستدامة

Endpoints:
    POST /consult/voice        — صوت → نص → رد → صفحة
    POST /consult/voice/base64 — base64 → نص → رد → صفحة
    POST /consult/text         — نص مباشر → رد → صفحة
    GET  /consult/health       — حالة الـ pipeline

Author : GODA EMAD
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from .voice import transcribe_audio
from .chat  import chat

log = logging.getLogger("riva.orchestrator")

router = APIRouter(prefix="/consult", tags=["orchestrator"])


# ─── Smart Routing Table — الـ 17 صفحة ──────────────────────────────────────
#
# المنطق:
#   intent  → الصفحة الأنسب للحالة
#   بعض الـ intents عندها sub-intents بناءً على الـ clinical_profile

_ROUTE_TABLE: dict[str, str] = {
    # حالات طوارئ → نتيجة فورية + QR
    "Emergency":          "04_result.html",

    # أعراض تحتاج فرز → صفحة الفرز
    "Triage":             "03_triage.html",

    # صحة الأم والحمل → داشبورد الأم
    "Pregnancy":          "10_mother_dashboard.html",

    # صحة مدرسية → داشبورد المدرسة
    "School":             "11_school_dashboard.html",

    # دكتور / طاقم طبي → لوحة الدكتور
    "Doctor":             "09_doctor_dashboard.html",

    # مريض سبق إدخاله → خطر إعادة الإدخال
    "Readmission":        "13_readmission.html",

    # سؤال عن مدة علاج / إقامة
    "LOS":                "14_los_dashboard.html",

    # طلب شرح قرار الـ AI
    "Explanation":        "12_ai_explanation.html",

    # عرض سجل مريض
    "History":            "05_history.html",

    # داشبورد موحد (حالات متعددة / مزمنة)
    "Combined":           "15_combined_dashboard.html",

    # دكتور يكتب ملاحظات
    "DoctorNotes":        "16_doctor_notes.html",

    # استفسار عام → الشات
    "General":            "02_chatbot.html",
}

# Sub-routing بناءً على الـ clinical_profile
# لو المريض سكري + طارئ → نتيجة مباشرة
# لو المريض حامل + أعراض → داشبورد الأم مش الفرز العادي
_PROFILE_OVERRIDE: list[tuple[dict, str, str]] = [
    # (profile_conditions,          original_intent, override_page)
    ({"is_pregnant": True},         "Triage",        "10_mother_dashboard.html"),
    ({"is_pregnant": True},         "General",       "06_pregnancy.html"),
    ({"has_diabetes": True,
      "has_heart_disease": True},   "Triage",        "15_combined_dashboard.html"),
]


def _smart_route(intent: str, clinical_profile: dict) -> str:
    """
    Determines the correct target page from the 17 pages
    based on chat intent + patient clinical profile.

    Priority:
        1. Profile overrides  (حامل + أعراض → داشبورد الأم)
        2. Intent table       (الجدول الأساسي)
        3. Fallback           (02_chatbot.html)
    """
    # 1. Check profile overrides first
    for conditions, original_intent, override_page in _PROFILE_OVERRIDE:
        if intent == original_intent:
            if all(clinical_profile.get(k) == v for k, v in conditions.items()):
                log.info(
                    "[RIVA-Router] Profile override: %s + %s → %s",
                    intent, conditions, override_page,
                )
                return override_page

    # 2. Standard intent routing
    page = _ROUTE_TABLE.get(intent, "02_chatbot.html")
    log.info("[RIVA-Router] %s → %s", intent, page)
    return page


# ─── Empathy prefix ──────────────────────────────────────────────────────────

LONG_RECORDING_S  = 60
MULTI_CHUNK_THRESH = 2


def _empathy_prefix(stt: dict) -> str:
    duration_s = stt.get("duration_ms", 0) / 1000
    chunks     = stt.get("chunks", 1)
    if duration_s >= LONG_RECORDING_S:
        return (
            "أنا سمعت شكوتك الطويلة ومهتم جداً بكل التفاصيل اللي قلتها. "
            "هحاول أساعدك بأفضل طريقة ممكنة.\n\n"
        )
    if chunks >= MULTI_CHUNK_THRESH:
        return "شكراً على الشرح الكامل، ده بيساعدني أفهم الموضوع أحسن.\n\n"
    return ""


# ─── Core pipeline ───────────────────────────────────────────────────────────

def process_consultation(
    user_text:  Optional[str] = None,
    audio_bytes: Optional[bytes] = None,
    session_id: Optional[str] = None,
    language:   str = "ar",
) -> dict:
    """
    Universal consultation pipeline — accepts text OR audio.

    Stage 1 — الأذن   : transcribe (لو صوت)
    Stage 2 — التعاطف : empathy prefix
    Stage 3 — العقل   : chat() → intent + clinical_profile
    Stage 4 — التوجيه : _smart_route() → target_page من الـ 17
    """
    t0  = time.perf_counter()
    stt = {}

    # Stage 1: STT (لو في صوت)
    if audio_bytes:
        log.info("[RIVA-Orch] Stage 1: STT …")
        stt       = transcribe_audio(audio_bytes, language=language)
        user_text = stt.get("text", "")
        if not user_text:
            raise ValueError("الصوت مفيش كلام واضح — اطلب من المريض يكرر.")

    if not user_text:
        raise ValueError("لازم ترسل نص أو صوت.")

    log.info("[RIVA-Orch] Input: '%s...'", user_text[:60])

    # Stage 2: Empathy prefix
    prefix = _empathy_prefix(stt)

    # Stage 3: Chat → intent + profile
    log.info("[RIVA-Orch] Stage 2: chat inference …")
    chat_result      = chat(user_message=user_text, session_id=session_id)
    intent           = chat_result["intent"]
    session_id       = chat_result["session_id"]
    clinical_profile = chat_result.get("clinical_profile", {})
    final_response   = prefix + chat_result["text"]

    # Stage 4: Smart routing → target page
    target_page = _smart_route(intent, clinical_profile)

    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    log.info(
        "[RIVA-Orch] Done  intent=%s  page=%s  total=%.0f ms",
        intent, target_page, total_ms,
    )

    return {
        "patient_said":      user_text,
        "riva_response":     final_response,
        "intent":            intent,
        "target_page":       target_page,        # ← الصفحة الصح من الـ 17
        "session_id":        session_id,
        "clinical_profile":  clinical_profile,
        "stt_duration_ms":   stt.get("duration_ms", 0),
        "chat_duration_ms":  chat_result["duration_ms"],
        "total_duration_ms": total_ms,
        "audio_chunks":      stt.get("chunks", 0),
        "offline":           True,
        "multimodal":        bool(audio_bytes),
    }


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class TextConsultRequest(BaseModel):
    message:    str           = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    language:   str           = "ar"


class Base64ConsultRequest(BaseModel):
    audio_base64: str           = Field(..., description="الصوت base64")
    session_id:   Optional[str] = None
    language:     str           = "ar"


class ConsultationResponse(BaseModel):
    patient_said:      str
    riva_response:     str
    intent:            str
    target_page:       str     # ← الصفحة من الـ 17 اللي المريض يروحها
    session_id:        str
    clinical_profile:  dict
    stt_duration_ms:   float
    chat_duration_ms:  float
    total_duration_ms: float
    audio_chunks:      int
    offline:           bool = True
    multimodal:        bool = False


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.post(
    "/text",
    response_model=ConsultationResponse,
    summary="استشارة نصية → رد + توجيه ذكي",
    description=(
        "يقبل رسالة نصية ويرجع الرد الطبي + اسم الصفحة الصح "
        "من الـ 17 صفحة بناءً على حالة المريض."
    ),
)
async def text_consultation(req: TextConsultRequest):
    try:
        result = process_consultation(
            user_text=req.message,
            session_id=req.session_id,
            language=req.language,
        )
        return ConsultationResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("[RIVA-Orch] text consultation error")
        raise HTTPException(status_code=500, detail=f"خطأ: {e}")


@router.post(
    "/voice",
    response_model=ConsultationResponse,
    summary="استشارة صوتية → نص → رد + توجيه ذكي",
    description=(
        "يقبل ملف صوتي، يحوّله لنص، يولّد رد طبي، "
        "ويرجع اسم الصفحة الصح من الـ 17 صفحة."
    ),
)
async def voice_consultation(
    file:       UploadFile    = File(...),
    session_id: Optional[str] = None,
    language:   str           = "ar",
):
    if not file.content_type or not any(
        t in file.content_type for t in ("audio", "octet-stream", "video")
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="الملف لازم يكون صوتي",
        )
    try:
        audio_bytes = await file.read()
        result      = process_consultation(
            audio_bytes=audio_bytes,
            session_id=session_id,
            language=language,
        )
        return ConsultationResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("[RIVA-Orch] voice consultation error")
        raise HTTPException(status_code=500, detail=f"خطأ: {e}")


@router.post(
    "/voice/base64",
    response_model=ConsultationResponse,
    summary="استشارة صوتية base64 (للـ PWA)",
)
async def voice_consultation_base64(req: Base64ConsultRequest):
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="base64 غير صحيح")
    try:
        result = process_consultation(
            audio_bytes=audio_bytes,
            session_id=req.session_id,
            language=req.language,
        )
        return ConsultationResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("[RIVA-Orch] base64 consultation error")
        raise HTTPException(status_code=500, detail=f"خطأ: {e}")


@router.get("/routes", summary="جدول التوجيه الكامل للـ 17 صفحة")
async def get_route_table():
    """Returns the full routing table so the frontend knows all possible redirects."""
    return {
        "route_table":    _ROUTE_TABLE,
        "total_pages":    17,
        "profile_overrides": len(_PROFILE_OVERRIDE),
        "description": "intent → target_page mapping for all 17 RIVA pages",
    }


@router.get("/health", summary="حالة الـ pipeline كاملة")
async def orchestrator_health():
    from .voice import ENCODER_PATH, DECODER_PATH, DECODER_PAST_PATH
    from .chat  import MODEL_PATH as CHAT_MODEL_PATH

    voice_ok = all(p.exists() for p in [ENCODER_PATH, DECODER_PATH, DECODER_PAST_PATH])
    chat_ok  = CHAT_MODEL_PATH.exists()

    return {
        "status":     "ok" if (voice_ok and chat_ok) else "degraded",
        "offline":    True,
        "multimodal": True,
        "routing": {
            "total_pages":       17,
            "intents_mapped":    len(_ROUTE_TABLE),
            "profile_overrides": len(_PROFILE_OVERRIDE),
            "route_table":       _ROUTE_TABLE,
        },
        "pipeline": {
            "voice_layer": "ok" if voice_ok else "missing",
            "chat_layer":  "ok" if chat_ok  else "missing",
            "bridge":      "ok",
        },
    }
