"""
===============================================================================
explanation_generator.py
مولد الشروحات الطبية
===============================================================================
"""

from typing import Dict, List, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """مولد الشروحات الطبية"""
    
    def __init__(self, language: str = 'ar'):
        self.language = language
        logger.info("✅ ExplanationGenerator initialized")
    
    def generate_readmission_explanation(self, probability: float, features: Dict) -> Dict:
        """توليد شرح readmission"""
        
        # تحديد مستوى الخطر
        if probability < 0.3:
            risk_level = "منخفض"
            color = "🟢"
        elif probability < 0.7:
            risk_level = "متوسط"
            color = "🟡"
        else:
            risk_level = "مرتفع"
            color = "🔴"
        
        # الملخص
        summary = f"{color} احتمالية إعادة الدخول: {risk_level} ({probability:.1%})"
        
        # التوصيات
        recommendations = []
        if probability < 0.3:
            recommendations = [
                "متابعة روتينية خلال شهر",
                "تأكيد على الالتزام بالعلاج"
            ]
        elif probability < 0.7:
            recommendations = [
                "متابعة خلال أسبوع",
                "مراجعة الأدوية",
                "تثقيف المريض"
            ]
        else:
            recommendations = [
                "متابعة دقيقة خلال 48 ساعة",
                "مراجعة مع أخصائي",
                "متابعة منزلية"
            ]
        
        return {
            'summary': summary,
            'recommendations': recommendations,
            'probability': probability,
            'risk_level': risk_level,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_los_explanation(self, days: float, features: Dict) -> Dict:
        """توليد شرح LOS"""
        
        # تحديد الفئة
        if days < 3:
            category = "🟢 قصيرة"
        elif days < 7:
            category = "🟡 متوسطة"
        elif days < 14:
            category = "🟠 طويلة"
        else:
            category = "🔴 طويلة جداً"
        
        # الملخص
        summary = f"📅 مدة الإقامة المتوقعة: {days} أيام - {category}"
        
        # التوصيات
        recommendations = []
        if days < 3:
            recommendations = [
                "تجهيز تقرير الخروج",
                "وصف الأدوية اللازمة"
            ]
        elif days < 7:
            recommendations = [
                "تحضير خطة خروج مبكر",
                "تثقيف المريض للعناية الذاتية"
            ]
        else:
            recommendations = [
                "تجهيز خطة رعاية موسعة",
                "توفير دعم نفسي",
                "مراجعة التغذية"
            ]
        
        return {
            'summary': summary,
            'recommendations': recommendations,
            'days': days,
            'category': category,
            'timestamp': datetime.now().isoformat()
        }


# اختبار
if __name__ == "__main__":
    print("="*50)
    print("🔧 اختبار ExplanationGenerator")
    print("="*50)
    
    gen = ExplanationGenerator()
    
    # اختبار readmission
    result1 = gen.generate_readmission_explanation(0.75, {})
    print("\n📝", result1['summary'])
    for rec in result1['recommendations']:
        print("   •", rec)
    
    # اختبار LOS
    result2 = gen.generate_los_explanation(8.5, {})
    print("\n📝", result2['summary'])
    for rec in result2['recommendations']:
        print("   •", rec)
    
    print("\n✅ تم الانتهاء")
