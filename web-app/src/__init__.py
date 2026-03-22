"""
__init__.py
===========
RIVA Health Platform - Web Application Package
حزمة تطبيق الويب الرئيسي لمنصة RIVA الصحية
===============================================

هذا الملف هو نقطة الدخول لحزمة التطبيق بالكامل.
يستورد جميع المكونات ويوفر واجهة موحدة للتطبيق الرئيسي.

Author: GODA EMAD
Version: 4.2.1
"""

import logging
from pathlib import Path

# ✅ إنشاء مجلد السجلات إذا لم يكن موجوداً (قبل إعداد logging)
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / 'app.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

# معلومات الحزمة
__version__ = '4.2.1'
__author__ = 'GODA EMAD'
__description__ = 'RIVA Health Platform - Offline AI-Driven Health System'

# تصدير المكونات الرئيسية
from .api import register_all_routers, get_package_info as get_api_info
from .templates import static

# قائمة المكونات المتاحة للتصدير
__all__ = [
    # الإصدار والمعلومات
    '__version__',
    '__author__',
    '__description__',
    
    # API
    'register_all_routers',
    'get_api_info',
    
    # Static
    'static',
]

# معلومات الحزمة الكاملة
PACKAGE_INFO = {
    "name": "RIVA Health Platform",
    "version": __version__,
    "author": __author__,
    "description": __description__,
    "features": [
        "🤖 AI-Powered Medical Chatbot",
        "🎤 Voice-to-Text (Whisper ONNX)",
        "📊 Readmission & LOS Prediction",
        "🚑 Smart Triage System",
        "👶 Pregnancy Risk Assessment",
        "📚 School Health Analytics",
        "💊 Drug Interaction Checker",
        "📝 E-Prescriptions with Digital Signature",
        "🔐 Secure Encrypted Storage",
        "📱 PWA with Offline Support",
        "🌐 RTL Arabic Support"
    ],
    "pages_count": 17,
    "api_endpoints": 16,
    "security_version": "v4.2.1"
}


def get_package_info():
    """الحصول على معلومات الحزمة الكاملة"""
    return PACKAGE_INFO


def init_app(app):
    """
    تهيئة التطبيق الرئيسي
    
    Args:
        app: FastAPI application instance
    """
    logger.info("Initializing RIVA Web Application v%s", __version__)
    
    # تسجيل جميع الـ routes
    register_all_routers(app)
    
    logger.info("Application initialized successfully")
    return app


logger.info("RIVA Web Application package loaded (v%s)", __version__)
