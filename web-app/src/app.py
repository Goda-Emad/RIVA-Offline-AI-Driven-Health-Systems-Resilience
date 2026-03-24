"""
app.py
======
RIVA Health Platform - Main Application Entry Point
الإصدار: 4.2.2 - Fixed Templates Path
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

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.routes import (
    chat_router, combined_router, explain_router, family_links_router,
    history_router, interactions_router, los_router, orchestrator_router,
    pregnancy_router, prescriptions_router, readmission_router, school_router,
    school_links_router, sentiment_router, triage_router, voice_router
)
from src.api.dependencies import get_system_status

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
    description="منصة صحية ذكية تعمل بدون إنترنت",
    version="4.2.2",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────
# 2. إعداد المسارات
# ──────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent
PUBLIC_DIR   = PROJECT_ROOT / "web-app" / "public"

TEMPLATES_DIR = BASE_DIR / "templates"
PAGES_DIR     = BASE_DIR / "templates" / "pages"
STATIC_DIR    = BASE_DIR / "templates" / "static"
# إنشاء المجلدات لو مش موجودة
for d in [TEMPLATES_DIR, PAGES_DIR, STATIC_DIR, PUBLIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(PAGES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")
# ──────────────────────────────────────────────────────────
# 3. تسجيل الـ Routes
# ──────────────────────────────────────────────────────────

routers = [
    chat_router, combined_router, explain_router, family_links_router,
    history_router, interactions_router, los_router, orchestrator_router,
    pregnancy_router, prescriptions_router, readmission_router, school_router,
    school_links_router, sentiment_router, triage_router, voice_router
]

for router in routers:
    if router.prefix == "/chat":
        app.include_router(router, prefix="/api/chat", tags=["Chat"])
    else:
        app.include_router(router)
    logger.info(f"✅ Router registered: {router.prefix}")
# ──────────────────────────────────────────────────────────
# 4. صفحات HTML (الـ 17 صفحة)
# ──────────────────────────────────────────────────────────

PAGES = [
    ("01_home.html",             "الرئيسية"),
    ("02_chatbot.html",          "المساعد الطبي"),
    ("03_triage.html",           "الفرز الطبي"),
    ("04_result.html",           "نتيجة التقييم"),
    ("05_history.html",          "التاريخ الطبي"),
    ("06_pregnancy.html",        "متابعة الحمل"),
    ("07_school.html",           "الصحة المدرسية"),
    ("08_offline.html",          "وضع عدم الاتصال"),
    ("09_doctor_dashboard.html", "لوحة الدكتور"),
    ("10_mother_dashboard.html", "لوحة الأم"),
    ("11_school_dashboard.html", "لوحة المدرسة"),
    ("12_ai_explanation.html",   "شرح الذكاء الاصطناعي"),
    ("13_readmission.html",      "التنبؤ بإعادة الدخول"),
    ("14_los_dashboard.html",    "التنبؤ بمدة الإقامة"),
    ("15_combined_dashboard.html","داشبورد متكامل"),
    ("16_doctor_notes.html",     "ملاحظات الطبيب"),
    ("17_sustainability.html",   "الاستدامة"),
]

for filename, title in PAGES:
    @app.get(f"/{filename}", response_class=HTMLResponse)
    async def serve_page(request: Request, f=filename, t=title):
        try:
            return templates.TemplateResponse(
                name=f, context={"request": request, "title": t}
            )
        except Exception as e:
            logger.error(f"Failed to serve {f}: {e}")
            return HTMLResponse(
                content=f"<h1>404 - Page not found</h1><p>{f}</p>", status_code=404
            )

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        name="01_home.html",
        context={"request": request, "title": "ريفا | الصحة للجميع"}
    )

# ──────────────────────────────────────────────────────────
# 5. صفحات إضافية
# ──────────────────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    try:
        return templates.TemplateResponse(
            name="login.html", context={"request": request, "title": "تسجيل الدخول"}
        )
    except Exception:
        return HTMLResponse("<h1>Login page not found</h1>", status_code=404)

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    try:
        return templates.TemplateResponse(
            name="register.html", context={"request": request, "title": "إنشاء حساب"}
        )
    except Exception:
        return HTMLResponse("<h1>Register page not found</h1>", status_code=404)

@app.get("/profile", response_class=HTMLResponse)
async def profile_page(request: Request):
    try:
        return templates.TemplateResponse(
            name="profile.html", context={"request": request, "title": "الملف الشخصي"}
        )
    except Exception:
        return HTMLResponse("<h1>Profile page not found</h1>", status_code=404)

# ──────────────────────────────────────────────────────────
# 6. PWA Routes
# ──────────────────────────────────────────────────────────

@app.get("/manifest.json")
async def manifest():
    p = PUBLIC_DIR / "manifest.json"
    return FileResponse(p, media_type="application/json") if p.exists() else JSONResponse({"error": "not found"}, status_code=404)

@app.get("/service-worker.js")
async def service_worker():
    p = PUBLIC_DIR / "service-worker.js"
    return FileResponse(p, media_type="application/javascript") if p.exists() else HTMLResponse("// not found", status_code=404)

@app.get("/offline-fallback.html")
async def offline_fallback():
    p = PUBLIC_DIR / "offline-fallback.html"
    return FileResponse(p, media_type="text/html") if p.exists() else HTMLResponse("<h1>Offline</h1>", status_code=503)

# ──────────────────────────────────────────────────────────
# 7. API Routes
# ──────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "RIVA Health Platform",
        "version": "4.2.2",
        "offline": True,
        "api_endpoints": len(routers),
        "pages": len(PAGES),
        "templates_dir": str(PAGES_DIR),
    }

@app.get("/api/system-status")
async def system_status():
    return await get_system_status()

@app.get("/api/routes")
async def list_routes():
    return {
        "routes": [
            {"path": r.path, "methods": list(r.methods or []), "name": r.name}
            for r in app.routes
        ],
        "total": len(app.routes),
    }

# ──────────────────────────────────────────────────────────
# 8. معالجة الأخطاء
# ──────────────────────────────────────────────────────────

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    try:
        return templates.TemplateResponse(
            name="404.html",
            context={"request": request, "path": request.url.path},
            status_code=404
        )
    except Exception:
        return HTMLResponse("<h1>404 - Not Found</h1>", status_code=404)

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    try:
        return templates.TemplateResponse(
            name="500.html",
            context={"request": request, "error": str(exc)},
            status_code=500
        )
    except Exception:
        return HTMLResponse("<h1>500 - Internal Server Error</h1>", status_code=500)

# ──────────────────────────────────────────────────────────
# 9. Startup
# ──────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 50)
    logger.info("🚀 RIVA Health Platform v4.2.2")
    logger.info(f"📁 Pages dir  : {PAGES_DIR}")
    logger.info(f"📁 Static dir : {STATIC_DIR}")
    logger.info(f"📁 Public dir : {PUBLIC_DIR}")
    logger.info(f"🔌 Routers    : {len(routers)}")
    logger.info(f"📄 Pages      : {len(PAGES)}")
    logger.info("✅ Server ready")
    logger.info("=" * 50)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
