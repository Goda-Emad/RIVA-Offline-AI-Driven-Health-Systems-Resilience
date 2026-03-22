"""
__init__.py
===========
RIVA Health Platform - API Routes Package
حزمة مسارات API لمنصة RIVA الصحية
===============================================

يستورد جميع الـ 16 API Route ويجعلها متاحة للتطبيق الرئيسي.

Author: GODA EMAD
Version: 4.2.1
"""

from .chat import router as chat_router
from .combined import router as combined_router
from .explain import router as explain_router
from .family_links import router as family_links_router
from .history import router as history_router
from .interactions import router as interactions_router
from .los import router as los_router
from .orchestrator import router as orchestrator_router
from .pregnancy import router as pregnancy_router
from .prescriptions import router as prescriptions_router
from .readmission import router as readmission_router
from .school import router as school_router
from .school_links import router as school_links_router
from .sentiment import router as sentiment_router
from .triage import router as triage_router
from .voice import router as voice_router

__all__ = [
    "chat_router",
    "combined_router",
    "explain_router",
    "family_links_router",
    "history_router",
    "interactions_router",
    "los_router",
    "orchestrator_router",
    "pregnancy_router",
    "prescriptions_router",
    "readmission_router",
    "school_router",
    "school_links_router",
    "sentiment_router",
    "triage_router",
    "voice_router",
]

# قائمة جميع الـ routers للتسجيل في main.py
ALL_ROUTERS = [
    chat_router,
    combined_router,
    explain_router,
    family_links_router,
    history_router,
    interactions_router,
    los_router,
    orchestrator_router,
    pregnancy_router,
    prescriptions_router,
    readmission_router,
    school_router,
    school_links_router,
    sentiment_router,
    triage_router,
    voice_router,
]

# معلومات عن الـ routes
ROUTES_INFO = {
    "chat": {"router": chat_router, "prefix": "/chat", "tags": ["chat"], "description": "محادثة طبية بالعامية المصرية"},
    "combined": {"router": combined_router, "prefix": "/api/predict", "tags": ["Combined Predictions"], "description": "التنبؤات المتكاملة"},
    "explain": {"router": explain_router, "prefix": "/api/v1", "tags": ["Medical Explanations"], "description": "شرح قرارات الذكاء الاصطناعي"},
    "family_links": {"router": family_links_router, "prefix": "/api/v1", "tags": ["Family Links & Genetics"], "description": "شجرة العائلة والمخاطر الوراثية"},
    "history": {"router": history_router, "prefix": "/api/v1", "tags": ["Medical History"], "description": "التاريخ الطبي الكامل"},
    "interactions": {"router": interactions_router, "prefix": "/interactions", "tags": ["Drug Interactions"], "description": "فحص تداخلات الأدوية"},
    "los": {"router": los_router, "prefix": "/api/predict/los", "tags": ["LOS Prediction"], "description": "التنبؤ بمدة الإقامة"},
    "orchestrator": {"router": orchestrator_router, "prefix": "/consult", "tags": ["orchestrator"], "description": "العقل المدبر"},
    "pregnancy": {"router": pregnancy_router, "prefix": "/api/predict/pregnancy", "tags": ["Pregnancy Risk"], "description": "تقييم مخاطر الحمل"},
    "prescriptions": {"router": prescriptions_router, "prefix": "/prescriptions", "tags": ["Prescriptions"], "description": "الوصفات الطبية والتوقيع الإلكتروني"},
    "readmission": {"router": readmission_router, "prefix": "/api/predict/readmission", "tags": ["Readmission Prediction"], "description": "التنبؤ بإعادة الدخول"},
    "school": {"router": school_router, "prefix": "/api/school", "tags": ["School Health"], "description": "تحليل صحة الطلاب"},
    "school_links": {"router": school_links_router, "prefix": "/api/school-links", "tags": ["School Links"], "description": "ربط الطلاب بالمدارس"},
    "sentiment": {"router": sentiment_router, "prefix": "/api/sentiment", "tags": ["Sentiment Analysis"], "description": "تحليل المشاعر"},
    "triage": {"router": triage_router, "prefix": "/api/triage", "tags": ["Triage"], "description": "تصنيف حالات الطوارئ"},
    "voice": {"router": voice_router, "prefix": "/voice", "tags": ["voice"], "description": "تحويل الصوت إلى نص"},
}

def get_routes_stats():
    """إرجاع إحصائيات عن جميع الـ routes"""
    return {
        "total_routers": len(ALL_ROUTERS),
        "total_endpoints": sum(len(router.routes) for router in ALL_ROUTERS if hasattr(router, 'routes')),
        "routes_info": ROUTES_INFO
    }
