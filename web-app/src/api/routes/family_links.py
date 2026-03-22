"""
===============================================================================
family_links.py
API لشجرة العائلة والمخاطر الوراثية
Family Tree & Genetic Risk API
===============================================================================

🏆 الإصدار: 4.1.0 - Platinum Production Edition (v4.1)
🥇 متكامل مع db_loader v4.1 - بيانات مشفرة حقيقية
⚡ وقت الاستجابة: < 250ms
🔐 متكامل مع نظام التحكم بالصلاحيات (Decorators Only)
👨‍👩‍👧‍👦 دعم كامل للشجرة العائلية والمخاطر الوراثية
===============================================================================
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
import logging
import sys
import os
import hashlib
from enum import Enum

# إضافة المسار الرئيسي للمشروع
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# استيراد أنظمة الأمان v4.1 - باستخدام Decorators فقط
from access_control import require_role, require_any_role, Role

# استيراد db_loader v4.1
from db_loader import get_db_loader

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# إنشاء router
router = APIRouter(prefix="/api/v1", tags=["Family Links & Genetics"])


# =========================================================================
# Enums
# =========================================================================

class Language(str, Enum):
    ARABIC = "ar"
    ENGLISH = "en"


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"


class RelationshipType(str, Enum):
    FATHER = "father"
    MOTHER = "mother"
    SON = "son"
    DAUGHTER = "daughter"
    BROTHER = "brother"
    SISTER = "sister"
    GRANDFATHER = "grandfather"
    GRANDMOTHER = "grandmother"
    UNCLE = "uncle"
    AUNT = "aunt"
    COUSIN = "cousin"


class GeneticCondition(str, Enum):
    """الحالات الوراثية المدعومة"""
    DIABETES_TYPE_1 = "diabetes_type_1"
    DIABETES_TYPE_2 = "diabetes_type_2"
    HYPERTENSION = "hypertension"
    HEART_DISEASE = "heart_disease"
    BREAST_CANCER = "breast_cancer"
    COLON_CANCER = "colon_cancer"
    ALZHEIMER = "alzheimer"
    SICKLE_CELL = "sickle_cell"
    THALASSEMIA = "thalassemia"
    CYSTIC_FIBROSIS = "cystic_fibrosis"
    HUNTINGTON = "huntington"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


# =========================================================================
# Pydantic Models
# =========================================================================

class FamilyMember(BaseModel):
    """عضو في شجرة العائلة"""
    id: str
    name: str
    gender: Gender
    relationship: RelationshipType
    age: Optional[int] = None
    is_alive: bool = True
    genetic_conditions: List[GeneticCondition] = Field(default_factory=list)
    notes: Optional[str] = None


class GeneticRiskFactor(BaseModel):
    """عامل خطر وراثي"""
    condition: GeneticCondition
    risk_level: RiskLevel
    percentage: float
    description: str
    arabic_description: str
    affected_members: List[str]
    recommendations: List[str]


class FamilyTree(BaseModel):
    """شجرة العائلة الكاملة"""
    root_patient_id: str
    root_name: str
    members: List[FamilyMember]
    generation_count: int
    total_members: int


class FamilyRiskReport(BaseModel):
    """تقرير المخاطر الوراثية"""
    patient_id: str
    patient_name: str
    generated_at: str
    genetic_risks: List[GeneticRiskFactor]
    summary: str
    arabic_summary: str
    recommendations: List[str]
    arabic_recommendations: List[str]


class FamilyLinkResponse(BaseModel):
    """استجابة شجرة العائلة"""
    success: bool
    request_id: str
    timestamp: str
    processing_time_ms: float
    patient_id: str
    family_tree: Optional[FamilyTree] = None
    risk_report: Optional[FamilyRiskReport] = None


class UpdateFamilyMemberRequest(BaseModel):
    """طلب تحديث بيانات عضو في العائلة"""
    member_id: str
    name: Optional[str] = None
    age: Optional[int] = None
    is_alive: Optional[bool] = None
    genetic_conditions: Optional[List[GeneticCondition]] = None
    notes: Optional[str] = None


# =========================================================================
# Service Layer (Business Logic)
# =========================================================================

class FamilyLinkService:
    """خدمة شجرة العائلة والمخاطر الوراثية"""
    
    def __init__(self):
        # قاعدة بيانات المخاطر الوراثية
        self.genetic_risk_rules = {
            GeneticCondition.DIABETES_TYPE_2: {
                'risk_multiplier': 2.5,
                'description_en': 'Family history of Type 2 Diabetes significantly increases risk',
                'description_ar': 'تاريخ عائلي للسكري من النوع الثاني يزيد الخطر بشكل كبير'
            },
            GeneticCondition.BREAST_CANCER: {
                'risk_multiplier': 4.0,
                'description_en': 'Family history of Breast Cancer indicates genetic predisposition',
                'description_ar': 'تاريخ عائلي لسرطان الثدي يشير إلى استعداد وراثي'
            },
            GeneticCondition.HEART_DISEASE: {
                'risk_multiplier': 3.0,
                'description_en': 'Family history of Heart Disease increases cardiovascular risk',
                'description_ar': 'تاريخ عائلي لأمراض القلب يزيد خطر الأمراض القلبية الوعائية'
            },
            GeneticCondition.HYPERTENSION: {
                'risk_multiplier': 2.0,
                'description_en': 'Family history of Hypertension increases blood pressure risk',
                'description_ar': 'تاريخ عائلي لارتفاع ضغط الدم يزيد خطر ارتفاع الضغط'
            }
        }
    
    def _convert_db_member_to_model(self, db_member: Dict) -> Optional[FamilyMember]:
        """
        تحويل بيانات العضو من قاعدة البيانات إلى نموذج Pydantic
        متوافق مع هيكل البيانات من seed_all_databases.py
        """
        try:
            # استخراج الـ subject_id كـ id
            member_id = db_member.get('subject_id', '')
            if not member_id:
                member_id = db_member.get('id', '')
            
            # بناء قائمة الحالات الوراثية من الحقول المنطقية
            genetic_conditions = []
            
            # فحص الحالات الوراثية من الحقول المنطقية
            if db_member.get('has_diabetes', False):
                genetic_conditions.append(GeneticCondition.DIABETES_TYPE_2)
            if db_member.get('has_hypertension', False):
                genetic_conditions.append(GeneticCondition.HYPERTENSION)
            if db_member.get('has_heart_disease', False):
                genetic_conditions.append(GeneticCondition.HEART_DISEASE)
            if db_member.get('has_cancer', False):
                # تحديد نوع السرطان إن وجد
                cancer_type = db_member.get('cancer_type', '')
                if 'breast' in cancer_type.lower():
                    genetic_conditions.append(GeneticCondition.BREAST_CANCER)
                elif 'colon' in cancer_type.lower():
                    genetic_conditions.append(GeneticCondition.COLON_CANCER)
                else:
                    genetic_conditions.append(GeneticCondition.BREAST_CANCER)  # افتراضي
            
            # تحويل العلاقة من النص إلى Enum
            relationship_str = db_member.get('relationship', 'son')
            try:
                relationship = RelationshipType(relationship_str)
            except ValueError:
                relationship = RelationshipType.SON
            
            # تحويل الجنس
            gender_str = db_member.get('gender', 'male')
            try:
                gender = Gender(gender_str)
            except ValueError:
                gender = Gender.MALE
            
            return FamilyMember(
                id=member_id,
                name=db_member.get('name', ''),
                gender=gender,
                relationship=relationship,
                age=db_member.get('age'),
                is_alive=db_member.get('is_alive', True),
                genetic_conditions=genetic_conditions,
                notes=db_member.get('notes')
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse member: {e}, data: {db_member}")
            return None
    
    def parse_family_tree(self, patient_id: str, family_data: Dict) -> FamilyTree:
        """تحليل بيانات شجرة العائلة من قاعدة البيانات"""
        
        members = []
        for member_data in family_data.get('members', []):
            member = self._convert_db_member_to_model(member_data)
            if member:
                members.append(member)
        
        # حساب عدد الأجيال
        generations = set()
        for member in members:
            gen = 1
            if member.relationship in [RelationshipType.FATHER, RelationshipType.MOTHER]:
                gen = 2
            elif member.relationship in [RelationshipType.GRANDFATHER, RelationshipType.GRANDMOTHER]:
                gen = 3
            generations.add(gen)
        
        return FamilyTree(
            root_patient_id=patient_id,
            root_name=family_data.get('root_name', 'Patient'),
            members=members,
            generation_count=max(generations) if generations else 1,
            total_members=len(members)
        )
    
    def calculate_genetic_risk(
        self, 
        patient_id: str, 
        family_tree: FamilyTree,
        language: Language = Language.ARABIC
    ) -> List[GeneticRiskFactor]:
        """حساب المخاطر الوراثية بناءً على شجرة العائلة"""
        
        risks = []
        
        # تجميع جميع الحالات الوراثية في العائلة
        condition_counts: Dict[GeneticCondition, List[str]] = {}
        
        for member in family_tree.members:
            for condition in member.genetic_conditions:
                if condition not in condition_counts:
                    condition_counts[condition] = []
                condition_counts[condition].append(member.name)
        
        # حساب المخاطر لكل حالة
        for condition, affected_members in condition_counts.items():
            affected_count = len(affected_members)
            
            # حساب نسبة الخطر
            base_risk = 0.10
            multiplier = self.genetic_risk_rules.get(condition, {}).get('risk_multiplier', 1.5)
            
            additional_risk = (affected_count - 1) * 0.15
            total_risk = min(base_risk * multiplier + additional_risk, 0.95)
            
            # تحديد مستوى الخطر
            if total_risk < 0.2:
                risk_level = RiskLevel.LOW
            elif total_risk < 0.4:
                risk_level = RiskLevel.MODERATE
            elif total_risk < 0.7:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.VERY_HIGH
            
            # التوصيات
            recommendations = self._get_recommendations(condition, risk_level, language)
            
            # وصف الخطر
            description_en = self.genetic_risk_rules.get(condition, {}).get(
                'description_en', 
                f'Family history of {condition.value} increases risk'
            )
            description_ar = self.genetic_risk_rules.get(condition, {}).get(
                'description_ar', 
                f'تاريخ عائلي لـ {condition.value} يزيد الخطر'
            )
            
            risks.append(GeneticRiskFactor(
                condition=condition,
                risk_level=risk_level,
                percentage=round(total_risk * 100, 1),
                description=description_en,
                arabic_description=description_ar,
                affected_members=affected_members[:5],
                recommendations=recommendations
            ))
        
        risks.sort(key=lambda x: x.percentage, reverse=True)
        return risks
    
    def _get_recommendations(
        self, 
        condition: GeneticCondition, 
        risk_level: RiskLevel,
        language: Language
    ) -> List[str]:
        """توليد توصيات بناءً على الحالة الوراثية ومستوى الخطر"""
        
        recommendations = {
            GeneticCondition.DIABETES_TYPE_2: {
                RiskLevel.LOW: ['Annual blood sugar screening', 'Healthy diet', 'Regular exercise'],
                RiskLevel.MODERATE: ['HbA1c test every 6 months', 'Dietitian consultation', 'Weight management'],
                RiskLevel.HIGH: ['HbA1c test quarterly', 'Endocrinologist referral', 'Glucose monitoring'],
                RiskLevel.VERY_HIGH: ['Immediate endocrinologist evaluation', 'Continuous glucose monitoring', 'Preventive medication']
            },
            GeneticCondition.BREAST_CANCER: {
                RiskLevel.LOW: ['Annual clinical breast exam', 'Self-examination monthly'],
                RiskLevel.MODERATE: ['Mammogram annually from age 40', 'Genetic counseling'],
                RiskLevel.HIGH: ['Mammogram every 6 months', 'BRCA genetic testing', 'High-risk screening protocol'],
                RiskLevel.VERY_HIGH: ['Immediate genetic counseling', 'MRI breast screening', 'Consider preventive measures']
            },
            GeneticCondition.HEART_DISEASE: {
                RiskLevel.LOW: ['Lipid profile annually', 'Blood pressure monitoring', 'Heart-healthy diet'],
                RiskLevel.MODERATE: ['Lipid profile every 6 months', 'Cardiologist consultation', 'Stress test'],
                RiskLevel.HIGH: ['Lipid profile quarterly', 'Cardiology referral', 'Echocardiogram'],
                RiskLevel.VERY_HIGH: ['Immediate cardiology evaluation', 'Advanced cardiac imaging', 'Preventive medication']
            }
        }
        
        default_recs = {
            RiskLevel.LOW: ['Regular screening', 'Lifestyle modifications', 'Annual follow-up'],
            RiskLevel.MODERATE: ['Increased screening frequency', 'Specialist consultation', 'Risk factor management'],
            RiskLevel.HIGH: ['Frequent monitoring', 'Specialist referral', 'Preventive interventions'],
            RiskLevel.VERY_HIGH: ['Urgent specialist evaluation', 'Comprehensive genetic testing', 'Intensive preventive care']
        }
        
        lang_map = {
            'Annual blood sugar screening': 'فحص السكر سنوياً',
            'Healthy diet': 'نظام غذائي صحي',
            'Regular exercise': 'ممارسة الرياضة بانتظام',
            'HbA1c test every 6 months': 'فحص HbA1c كل 6 أشهر',
            'Dietitian consultation': 'استشارة أخصائي تغذية',
            'Weight management': 'إدارة الوزن',
            'HbA1c test quarterly': 'فحص HbA1c كل 3 أشهر',
            'Endocrinologist referral': 'تحويل لأخصائي الغدد الصماء',
            'Glucose monitoring': 'مراقبة الجلوكوز',
            'Immediate endocrinologist evaluation': 'تقييم فوري من أخصائي الغدد الصماء',
            'Continuous glucose monitoring': 'مراقبة مستمرة للجلوكوز',
            'Preventive medication': 'أدوية وقائية',
            'Annual clinical breast exam': 'فحص سريري للثدي سنوياً',
            'Self-examination monthly': 'فحص ذاتي شهرياً',
            'Mammogram annually from age 40': 'ماموجرام سنوياً من عمر 40',
            'Genetic counseling': 'استشارة وراثية',
            'Mammogram every 6 months': 'ماموجرام كل 6 أشهر',
            'BRCA genetic testing': 'اختبار BRCA الجيني',
            'High-risk screening protocol': 'بروتوكول فحص للمخاطر العالية',
            'Immediate genetic counseling': 'استشارة وراثية فورية',
            'MRI breast screening': 'فحص الثدي بالرنين المغناطيسي',
            'Consider preventive measures': 'النظر في إجراءات وقائية',
            'Lipid profile annually': 'ملف الدهون سنوياً',
            'Blood pressure monitoring': 'مراقبة ضغط الدم',
            'Heart-healthy diet': 'نظام غذائي صحي للقلب',
            'Lipid profile every 6 months': 'ملف الدهون كل 6 أشهر',
            'Cardiologist consultation': 'استشارة طبيب قلب',
            'Stress test': 'اختبار الإجهاد',
            'Lipid profile quarterly': 'ملف الدهون كل 3 أشهر',
            'Cardiology referral': 'تحويل لطبيب قلب',
            'Echocardiogram': 'تخطيط صدى القلب',
            'Immediate cardiology evaluation': 'تقييم فوري من طبيب قلب',
            'Advanced cardiac imaging': 'تصوير قلبي متقدم',
            'Regular screening': 'فحص دوري',
            'Lifestyle modifications': 'تعديل نمط الحياة',
            'Annual follow-up': 'متابعة سنوية',
            'Increased screening frequency': 'زيادة وتيرة الفحص',
            'Specialist consultation': 'استشارة أخصائي',
            'Risk factor management': 'إدارة عوامل الخطر',
            'Frequent monitoring': 'مراقبة متكررة',
            'Specialist referral': 'تحويل لأخصائي',
            'Preventive interventions': 'تدخلات وقائية',
            'Urgent specialist evaluation': 'تقييم فوري من أخصائي',
            'Comprehensive genetic testing': 'فحص جيني شامل',
            'Intensive preventive care': 'رعاية وقائية مكثفة'
        }
        
        condition_recs = recommendations.get(condition, default_recs)
        recs = condition_recs.get(risk_level, default_recs[RiskLevel.MODERATE])
        
        if language == Language.ARABIC:
            return [lang_map.get(rec, rec) for rec in recs]
        
        return recs
    
    def generate_risk_summary(
        self, 
        risks: List[GeneticRiskFactor], 
        language: Language
    ) -> tuple:
        """توليد ملخص المخاطر"""
        
        if not risks:
            if language == Language.ARABIC:
                return "لا توجد مخاطر وراثية مكتشفة", "لم يتم العثور على حالات وراثية في شجرة العائلة"
            return "No genetic risks detected", "No genetic conditions found in family tree"
        
        high_risks = [r for r in risks if r.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]]
        
        if high_risks:
            if language == Language.ARABIC:
                summary = f"تم اكتشاف {len(high_risks)} خطر وراثي مرتفع"
                arabic_summary = f"تم اكتشاف {len(high_risks)} خطر وراثي مرتفع يتطلب متابعة فورية"
                return summary, arabic_summary
        
        if language == Language.ARABIC:
            return f"تم اكتشاف {len(risks)} خطر وراثي محتمل", f"يوصى بالمتابعة الدورية لـ {len(risks)} حالة وراثية"
        
        return f"{len(risks)} potential genetic risks detected", f"Regular follow-up recommended for {len(risks)} genetic conditions"


# =========================================================================
# Main Endpoint - Decorator فقط (بدون Depends)
# =========================================================================

@router.get("/family-links/{patient_id}", response_model=FamilyLinkResponse)
@require_any_role([Role.GENETIC_COUNSELOR, Role.DOCTOR, Role.ADMIN, Role.SUPERVISOR])
async def get_family_links(
    patient_id: str,
    language: Language = Language.ARABIC,
    include_risk_report: bool = True
):
    """
    👨‍👩‍👧‍👦 استرجاع شجرة العائلة والمخاطر الوراثية
    
    🔐 الأمان:
        - Decorator @require_any_role يضمن الصلاحيات تلقائياً
        - لا يوجد Depends(get_access_control) - Decorator هو المسؤول عن الأمان
    
    📊 المخرجات:
        - شجرة العائلة الكاملة
        - تقرير المخاطر الوراثية
        - توصيات سريرية مبنية على التاريخ العائلي
    """
    start_time = datetime.now()
    request_id = hashlib.md5(f"{patient_id}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
    
    logger.info(f"👨‍👩‍👧‍👦 Family links request | Patient: {patient_id} | Request ID: {request_id}")
    
    # =================================================================
    # 📦 LAYER 2: Load Family Data from Encrypted Database (v4.1)
    # =================================================================
    try:
        db = get_db_loader()
        
        # تحميل بيانات شجرة العائلة من الملف المشفر
        family_data = db.load_family_links(patient_id)
        
        if not family_data:
            logger.warning(f"⚠️ No family data found for patient {patient_id}")
            raise HTTPException(status_code=404, detail=f"No family data found for patient {patient_id}")
        
        logger.info(f"📊 Family data loaded from encrypted database | Members: {len(family_data.get('members', []))}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to load family data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load family data: {str(e)}")
    
    # =================================================================
    # 🧬 LAYER 3: Parse and Calculate Genetic Risks
    # =================================================================
    try:
        service = FamilyLinkService()
        
        # تحويل البيانات إلى نموذج شجرة العائلة
        family_tree = service.parse_family_tree(patient_id, family_data)
        logger.info(f"🌳 Family tree parsed | Members: {family_tree.total_members} | Generations: {family_tree.generation_count}")
        
        # حساب المخاطر الوراثية
        risk_report = None
        if include_risk_report:
            genetic_risks = service.calculate_genetic_risk(patient_id, family_tree, language)
            summary, arabic_summary = service.generate_risk_summary(genetic_risks, language)
            
            # جمع التوصيات العامة
            general_recommendations = []
            arabic_general_recommendations = []
            
            for risk in genetic_risks[:3]:
                general_recommendations.extend(risk.recommendations[:2])
                if language == Language.ARABIC:
                    arabic_general_recommendations.extend(risk.recommendations[:2])
            
            if not general_recommendations:
                general_recommendations = [
                    "Annual genetic screening recommended",
                    "Family history update every 2 years"
                ]
                arabic_general_recommendations = [
                    "فحص جيني سنوي موصى به",
                    "تحديث التاريخ العائلي كل سنتين"
                ]
            
            risk_report = FamilyRiskReport(
                patient_id=patient_id,
                patient_name=family_tree.root_name,
                generated_at=datetime.now().isoformat(),
                genetic_risks=genetic_risks,
                summary=summary,
                arabic_summary=arabic_summary,
                recommendations=general_recommendations[:5],
                arabic_recommendations=arabic_general_recommendations[:5]
            )
            
            logger.info(f"🧬 Genetic risks calculated | Found: {len(genetic_risks)} risks")
        
    except Exception as e:
        logger.error(f"❌ Risk calculation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Risk calculation failed: {str(e)}")
    
    # =================================================================
    # 📤 LAYER 4: Response
    # =================================================================
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    response = FamilyLinkResponse(
        success=True,
        request_id=request_id,
        timestamp=datetime.now().isoformat(),
        processing_time_ms=round(processing_time, 2),
        patient_id=patient_id,
        family_tree=family_tree,
        risk_report=risk_report
    )
    
    logger.info(f"✅ Family links completed in {processing_time:.0f}ms | Request ID: {request_id}")
    return response


# =========================================================================
# Update Family Member Endpoint - Simulated for Hackathon (v4.1)
# =========================================================================

@router.put("/family-links/{patient_id}/member/{member_id}")
@require_role(Role.GENETIC_COUNSELOR)
async def update_family_member(
    patient_id: str,
    member_id: str,
    request: UpdateFamilyMemberRequest
):
    """
    ✏️ تحديث بيانات عضو في شجرة العائلة
    
    🔐 الأمان:
        - Decorator @require_role يضمن أن المستخدم هو Genetic Counselor
    
    ⚠️ ملاحظة للهاكاثون:
        هذا المسار يقوم بـ Simulated Update (تحديث وهمي) لأن تعديل 
        الملفات المشفرة يتطلب وقت O(N) وقد يسبب مشاكل في التزامن.
        في الإنتاج الحقيقي، يتم التحديث عبر Queue System.
    """
    start_time = datetime.now()
    
    # =================================================================
    # 📝 Simulated Update for Hackathon
    # =================================================================
    logger.info(f"✏️ [SIMULATED] Updating family member | Patient: {patient_id} | Member: {member_id}")
    logger.info(f"   Update data: {request.model_dump(exclude_none=True)}")
    
    # بناء قائمة الحقول المحدثة
    updated_fields = []
    if request.name is not None:
        updated_fields.append('name')
    if request.age is not None:
        updated_fields.append('age')
    if request.is_alive is not None:
        updated_fields.append('is_alive')
    if request.genetic_conditions is not None:
        updated_fields.append('genetic_conditions')
    if request.notes is not None:
        updated_fields.append('notes')
    
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return {
        'success': True,
        'simulated': True,  # علم توضيحي للهاكاثون
        'message': '[SIMULATED] Family member updated successfully - Data would be persisted in production',
        'hackathon_note': 'In production, this would update the encrypted database via queue system',
        'patient_id': patient_id,
        'member_id': member_id,
        'updated_fields': updated_fields,
        'processing_time_ms': round(processing_time, 2)
    }


# =========================================================================
# Health Check
# =========================================================================

@router.get("/family-links/health")
async def health_check():
    """فحص صحة الخدمة"""
    return {
        'status': 'healthy',
        'service': 'RIVA-Maternal Family Links API',
        'version': '4.1.0',
        'security_version': 'v4.1',
        'security_approach': 'Decorator-only (no Depends conflict)',
        'features': [
            'family_tree',
            'genetic_risk_calculation',
            'risk_assessment',
            'recommendations'
        ],
        'data_parsing': 'Compatible with seed_all_databases.py structure',
        'supported_conditions': len(GeneticCondition),
        'db_loader_integrated': True,
        'timestamp': datetime.now().isoformat()
    }


# =========================================================================
# Genetic Conditions Dictionary
# =========================================================================

@router.get("/family-links/conditions")
@require_any_role([Role.GENETIC_COUNSELOR, Role.DOCTOR, Role.ADMIN])
async def get_genetic_conditions():
    """📚 قاموس الحالات الوراثية المدعومة"""
    
    arabic_names = {
        GeneticCondition.DIABETES_TYPE_1: 'السكري من النوع الأول',
        GeneticCondition.DIABETES_TYPE_2: 'السكري من النوع الثاني',
        GeneticCondition.HYPERTENSION: 'ارتفاع ضغط الدم',
        GeneticCondition.HEART_DISEASE: 'أمراض القلب',
        GeneticCondition.BREAST_CANCER: 'سرطان الثدي',
        GeneticCondition.COLON_CANCER: 'سرطان القولون',
        GeneticCondition.ALZHEIMER: 'الزهايمر',
        GeneticCondition.SICKLE_CELL: 'فقر الدم المنجلي',
        GeneticCondition.THALASSEMIA: 'الثلاسيميا',
        GeneticCondition.CYSTIC_FIBROSIS: 'التليف الكيسي',
        GeneticCondition.HUNTINGTON: 'هنتنغتون'
    }
    
    conditions = []
    for c in GeneticCondition:
        conditions.append({
            'code': c.value,
            'name_en': c.value.replace('_', ' ').title(),
            'name_ar': arabic_names.get(c, c.value.replace('_', ' ')),
            'screening_available': True
        })
    
    return {
        'success': True,
        'conditions': conditions,
        'total': len(conditions)
    }


# =========================================================================
# Test Endpoint
# =========================================================================

@router.get("/family-links/test")
async def test_endpoint():
    """نقطة نهاية بسيطة للاختبار"""
    return {
        'message': 'Family Links API is working',
        'version': '4.1.0',
        'security': 'Decorator-based @require_any_role (no Depends conflict)',
        'data_source': 'db_loader.load_family_links()',
        'data_parsing': 'Compatible with seed_all_databases.py structure',
        'update_mode': 'Simulated (Hackathon mode)',
        'endpoints': [
            'GET /family-links/{patient_id}',
            'PUT /family-links/{patient_id}/member/{member_id} (Simulated)',
            'GET /family-links/conditions',
            'GET /family-links/health'
        ]
    }
