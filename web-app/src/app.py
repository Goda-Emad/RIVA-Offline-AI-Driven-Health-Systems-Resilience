"""
app.py
======
RIVA Health Platform - Main Application Entry Point
نقطة الدخول الرئيسية لتطبيق RIVA Health Platform

الوظائف:
- تهيئة تطبيق FastAPI
- تسجيل جميع الـ Routes (16 API)
- خدمة الملفات الثابتة (Static Files)
- خدمة صفحات HTML (Templates)
- خدمة PWA (manifest, service-worker, offline-fallback)
- إدارة CORS
- تكامل مع Service Worker

المسار: web-app/src/app.py

الإصدار: 4.2.1
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pathlib import Path
import logging
import sys
import os

# إضافة المسار الرئيسي
sys.path.append(str(Path(__file__).parent.parent.parent))

# استيراد الـ Routes
from src.api.routes import (
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
    voice_router
)

# استيراد التبعيات
from src.api.dependencies import get_system_status

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# 1. تهيئة التطبيق
# ──────────────────────────────────────────────────────────

app = FastAPI(
    title="RIVA Health Platform",
    description="منصة صحية ذكية تعمل بدون إنترنت - Offline AI-Driven Health System",
    version="4.2.1",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# ──────────────────────────────────────────────────────────
# 2. إعداد CORS
# ──────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────
# 3. إعداد المسارات
# ──────────────────────────────────────────────────────────

# الحصول على المسار الأساسي
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent  # الوصول إلى جذر المشروع
PUBLIC_DIR = PROJECT_ROOT / "web-app" / "public"  # مجلد public للملفات العامة
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "templates" / "static"

# التأكد من وجود المجلدات
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

# إعداد قوالب Jinja2
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# خدمة الملفات الثابتة (CSS, JS, Assets)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# خدمة الملفات العامة (PWA)
app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")

# ──────────────────────────────────────────────────────────
# 4. تسجيل الـ Routes (16 API)
# ──────────────────────────────────────────────────────────

routers = [
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
    voice_router
]

for router in routers:
    app.include_router(router)
    logger.info(f"✅ Router registered: {router.prefix}")

# ──────────────────────────────────────────────────────────
# 5. صفحات HTML (الـ 17 صفحة)
# ──────────────────────────────────────────────────────────

PAGES = [
    ("01_home.html", "الرئيسية"),
    ("02_chatbot.html", "المساعد الطبي"),
    ("03_triage.html", "الفرز الطبي"),
    ("04_result.html", "نتيجة التقييم"),
    ("05_history.html", "التاريخ الطبي"),
    ("06_pregnancy.html", "متابعة الحمل"),
    ("07_school.html", "الصحة المدرسية"),
    ("08_offline.html", "وضع عدم الاتصال"),
    ("09_doctor_dashboard.html", "لوحة الدكتور"),
    ("10_mother_dashboard.html", "لوحة الأم"),
    ("11_school_dashboard.html", "لوحة المدرسة"),
    ("12_ai_explanation.html", "شرح الذكاء الاصطناعي"),
    ("13_readmission.html", "التنبؤ بإعادة الدخول"),
    ("14_los_dashboard.html", "التنبؤ بمدة الإقامة"),
    ("15_combined_dashboard.html", "داشبورد متكامل"),
    ("16_doctor_notes.html", "ملاحظات الطبيب"),
    ("17_sustainability.html", "الاستدامة")
]

for filename, title in PAGES:
    @app.get(f"/{filename}", response_class=HTMLResponse)
    async def serve_page(request: Request, f=filename, t=title):
        try:
            return templates.TemplateResponse(
                name=f,
                context={"request": request, "title": t}
            )
        except Exception as e:
            logger.error(f"Failed to serve {f}: {e}")
            return HTMLResponse(content=f"<h1>404 - Page not found</h1><p>{f}</p>", status_code=404)

# الصفحة الرئيسية
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        name="01_home.html",
        context={"request": request, "title": "ريفا | الصحة للجميع"}
    )

# ──────────────────────────────────────────────────────────
# 6. PWA Routes (manifest, service-worker, offline-fallback)
# ──────────────────────────────────────────────────────────

@app.get("/manifest.json")
async def manifest():
    """PWA Manifest"""
    manifest_path = PUBLIC_DIR / "manifest.json"
    if manifest_path.exists():
        return FileResponse(manifest_path, media_type="application/json")
    return JSONResponse({"error": "manifest not found"}, status_code=404)

@app.get("/service-worker.js")
async def service_worker():
    """Service Worker"""
    sw_path = PUBLIC_DIR / "service-worker.js"
    if sw_path.exists():
        return FileResponse(sw_path, media_type="application/javascript")
    return HTMLResponse(content="// Service Worker not found", status_code=404)

@app.get("/offline-fallback.html")
async def offline_fallback():
    """صفحة Offline Fallback"""
    fallback_path = PUBLIC_DIR / "offline-fallback.html"
    if fallback_path.exists():
        return FileResponse(fallback_path, media_type="text/html")
    return HTMLResponse(content="<h1>Offline</h1><p>You are offline</p>", status_code=503)

# ──────────────────────────────────────────────────────────
# 7. صفحات إضافية (تسجيل الدخول، الملف الشخصي)
# ──────────────────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(
        name="login.html",
        context={"request": request, "title": "تسجيل الدخول - ريفا"}
    )

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse(
        name="register.html",
        context={"request": request, "title": "إنشاء حساب - ريفا"}
    )

@app.get("/profile", response_class=HTMLResponse)
async def profile_page(request: Request):
    return templates.TemplateResponse(
        name="profile.html",
        context={"request": request, "title": "الملف الشخصي - ريفا"}
    )

# ──────────────────────────────────────────────────────────
# 8. نقاط نهاية API إضافية
# ──────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "RIVA Health Platform",
        "version": "4.2.1",
        "offline": True,
        "multimodal": True,
        "api_endpoints": len(routers),
        "pages": len(PAGES)
    }

@app.get("/api/system-status")
async def system_status():
    return await get_system_status()

@app.get("/api/routes")
async def list_routes():
    routes = []
    for route in app.routes:
        routes.append({
            "path": route.path,
            "methods": list(route.methods) if route.methods else [],
            "name": route.name
        })
    return {"routes": routes, "total": len(routes)}

# ──────────────────────────────────────────────────────────
# 9. معالجة الأخطاء
# ──────────────────────────────────────────────────────────

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return templates.TemplateResponse(
        name="404.html",
        context={"request": request, "path": request.url.path},
        status_code=404
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return templates.TemplateResponse(
        name="500.html",
        context={"request": request, "error": str(exc)},
        status_code=500
    )

# ──────────────────────────────────────────────────────────
# 10. بدء التشغيل
# ──────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 50)
    logger.info("🚀 RIVA Health Platform v4.2.1")
    logger.info("=" * 50)
    logger.info(f"📁 Static files: {STATIC_DIR}")
    logger.info(f"📁 Templates: {TEMPLATES_DIR}")
    logger.info(f"📁 Public (PWA): {PUBLIC_DIR}")
    logger.info(f"🔌 API Endpoints: {len(routers)}")
    logger.info(f"📄 HTML Pages: {len(PAGES)}")
    logger.info("=" * 50)
    logger.info("✅ Server is ready to accept connections")
    logger.info("=" * 50)

# ──────────────────────────────────────────────────────────
# 11. تشغيل التطبيق
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


