"""
__init__.py
===========
RIVA Health Platform — API Package Initializer
-----------------------------------------------
يقوم بتهيئة مجلد API بالكامل ويوفر واجهة موحدة لاستيراد جميع الـ routes والتبعيات

Author: GODA EMAD
"""

from .dependencies import (
    # Security
    get_current_user,
    get_current_active_user,
    get_doctor_user,
    get_admin_user,
    
    # Database
    get_db,
    get_db_loader_instance,
    get_patient_context,
    get_patient_context_with_auth,
    
    # AI Models
    get_readmission_predictor,
    get_los_predictor,
    get_triage_engine,
    get_sentiment_analyzer,
    get_school_health_analyzer,
    get_drug_interaction_checker,
    get_prescription_generator,
    
    # Voice
    get_voice_encoder,
    
    # Orchestrator
    get_orchestrator,
    
    # Monitoring
    log_request,
    get_system_status,
    
    # School
    get_school_context,
    
    # Roles
    Role,
)

from .routes import (
    # All routers
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
    
    # All routers list
    ALL_ROUTERS,
    ROUTES_INFO,
    get_routes_stats,
)

__all__ = [
    # Dependencies
    "get_current_user",
    "get_current_active_user",
    "get_doctor_user",
    "get_admin_user",
    "get_db",
    "get_db_loader_instance",
    "get_patient_context",
    "get_patient_context_with_auth",
    "get_readmission_predictor",
    "get_los_predictor",
    "get_triage_engine",
    "get_sentiment_analyzer",
    "get_school_health_analyzer",
    "get_drug_interaction_checker",
    "get_prescription_generator",
    "get_voice_encoder",
    "get_orchestrator",
    "log_request",
    "get_system_status",
    "get_school_context",
    "Role",
    
    # Routes
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
    "ALL_ROUTERS",
    "ROUTES_INFO",
    "get_routes_stats",
]

# معلومات عن الـ API Package
PACKAGE_INFO = {
    "name": "RIVA Health Platform API",
    "version": "4.2.1",
    "description": "الواجهة الخلفية لمنصة RIVA الصحية",
    "routers_count": 16,
    "total_endpoints": None,  # سيتم حسابه عند التشغيل
    "dependencies_count": 18,
    "security_version": "v4.2.1",
    "features": [
        "🔐 Multi-layer security with JWT",
        "🎤 Voice-to-text with Whisper ONNX",
        "💬 Medical chatbot with Egyptian dialect",
        "🧠 AI predictions (Readmission, LOS, Triage)",
        "📋 Medical history management",
        "💊 Drug interactions checker",
        "📝 E-prescriptions with digital signature",
        "👶 Pregnancy risk assessment",
        "📚 School health analytics",
        "👨‍👩‍👧 Family tree & genetic risks",
        "😊 Sentiment analysis with Egyptian lexicon",
        "🎯 Smart routing to 17 dashboard pages",
        "🗄️ Encrypted data storage (13 databases)",
    ],
}


def get_package_info():
    """إرجاع معلومات عن حزمة الـ API"""
    import inspect
    from .routes import ALL_ROUTERS
    
    total_endpoints = 0
    for router in ALL_ROUTERS:
        if hasattr(router, 'routes'):
            total_endpoints += len(router.routes)
    
    PACKAGE_INFO["total_endpoints"] = total_endpoints
    return PACKAGE_INFO


# دالة لتسجيل جميع الـ routes في التطبيق الرئيسي
def register_all_routers(app):
    """
    تسجيل جميع الـ routers في تطبيق FastAPI
    
    Usage:
        from src.api import register_all_routers
        register_all_routers(app)
    """
    from .routes import ALL_ROUTERS
    
    for router in ALL_ROUTERS:
        app.include_router(router)
    
    # إضافة نقطة نهاية للإحصائيات
    @app.get("/api/info")
    async def api_info():
        return get_package_info()
    
    return app
