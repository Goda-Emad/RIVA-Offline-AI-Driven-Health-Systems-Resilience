# Medical Review Workflow
## RIVA Health Platform — دليل مراجعة القرارات الطبية

> **النسخة:** 2.0 | **آخر تحديث:** مارس 2026 | **المؤلف:** GODA EMAD

---

## 1. نظرة عامة

منظومة RIVA تعمل على مبدأ **Human-in-the-Loop** — الذكاء الاصطناعي يقترح، والطبيب يقرر.

هذا الملف يوثّق المسار الكامل لمراجعة القرارات الطبية من لحظة استقبال صوت المريض حتى تسجيل النتيجة النهائية وتحسين الموديل.

```
المريض يتكلم
      ↓
  RIVA تسمع وتفهم  (voice.py + chat.py)
      ↓
  Orchestrator يوجّه  (orchestrator.py)
      ↓
  ┌─────────────────────────────────────┐
  │   confidence ≥ 0.55                 │
  │   → الصفحة المناسبة مباشرة         │
  └─────────────────────────────────────┘
  ┌─────────────────────────────────────┐
  │   confidence < 0.55                 │
  │   → 12_ai_explanation.html          │
  │   → الدكتور يراجع                  │
  └─────────────────────────────────────┘
      ↓
  Doctor Validation Layer
  (clinical_override_log + doctor_feedback_handler)
      ↓
  Retraining Signal (لو في نمط سلبي)
      ↓
  RIVA تتحسن في الدورة الجاية
```

---

## 2. مستويات الثقة وقرارات التوجيه

| مستوى الثقة | النطاق | الإجراء |
|---|---|---|
| **عالي** | ≥ 0.80 | توجيه مباشر للصفحة المناسبة |
| **متوسط** | 0.55 – 0.79 | توجيه مع إشعار للدكتور في الـ dashboard |
| **منخفض** | < 0.55 | توجيه لـ `12_ai_explanation.html` للمراجعة |
| **طوارئ** | أي قيمة | `04_result.html` فوراً + تنبيه الدكتور |

---

## 3. مسارات المراجعة الطبية

### 3.1 المسار العادي (High Confidence)

```
المريض: "عندي صداع من امبارح"
    ↓
RIVA: intent=Triage, confidence=0.82
    ↓
→ 03_triage.html  (مباشرة، بدون تدخل)
    ↓
الدكتور يشوف الحالة في 09_doctor_dashboard.html
    ↓
يضغط ✓ "الرد صح"  → submit_validation()
```

### 3.2 مسار المراجعة (Low Confidence)

```
المريض: "تعبان شوية"  (غير واضح)
    ↓
RIVA: intent=General, confidence=0.38
    ↓
→ 12_ai_explanation.html
    ↓
RIVA تعرض: "بنيت قراري على: لا يوجد معلومات طبية محددة"
    ↓
المريض يصحّح: "أنا سكري وعندي ألم في الصدر"
    ↓
RIVA تعيد التقييم: intent=Emergency, confidence=0.94
    ↓
→ 04_result.html + تنبيه فوري للدكتور
```

### 3.3 مسار Override الطبي

```
RIVA: intent=Triage, confidence=0.71
      → "راجع طبيب في الأسبوع الجاي"
    ↓
الدكتور يفحص المريض سريرياً
    ↓
يكتشف: ألم صدر ينتشر للذراع اليسرى
    ↓
يضغط "تجاوز قرار AI" في 09_doctor_dashboard.html
    ↓
log_override(
    override_reason = AI_UNDERESTIMATED,
    severity        = CRITICAL,
    doctor_decision = "إدخال طارئ فوري",
    reason_notes    = "أعراض جلطة قلبية"
)
    ↓
record_hash يُحفظ → Chain of Trust مضمون
    ↓
Signal فوري للفريق التقني → RIVA تتعلم
```

---

## 4. صلاحيات الدكتور في RIVA

### 4.1 أنواع التدخل المتاحة

| النوع | الوظيفة | الملف |
|---|---|---|
| **Validation** ✓ | تأكيد صحة رد AI | `doctor_feedback_handler.submit_validation()` |
| **Rating** ⭐ | تقييم 1-5 نجوم | `doctor_feedback_handler.submit_rating()` |
| **Correction** ✏️ | تصحيح الرد | `doctor_feedback_handler.submit_correction()` |
| **Override** 🔄 | تجاوز القرار كلياً | `clinical_override_log.log_override()` |
| **Flag** 🚨 | إبلاغ عن خطر | `doctor_feedback_handler.submit_flag()` |

### 4.2 أوزان التأثير على إعادة التدريب

| تخصص الدكتور | الوزن |
|---|---|
| استشاري (Consultant) | 3.0× |
| أخصائي (Specialist) | 2.0× |
| ممارس عام (GP) | 1.0× |
| مقيم (Resident) | 0.8× |
| متدرب (Intern) | 0.5× |

> مثال: Flag من استشاري = 3.0 × 3.0 = **9.0 نقطة** → يولّد Retraining Signal فوري

---

## 5. Chain of Trust — ضمان سلامة السجلات

كل سجل override يُختم بـ `record_hash` مشتق من بياناته الثابتة.

### كيف يعمل:

```python
# عند الحفظ
record_hash = SHA256(
    override_id + timestamp + doctor_id + patient_id_hash +
    ai_intent + ai_confidence + doctor_decision + override_reason
)

# عند المراجعة
is_intact = (stored_hash == recompute_hash(record))
```

### في الـ dashboard:
```
✓  1,247 سجل سليم
⚠️  0   سجل معدّل
→  الحالة: CLEAN
```

### لو اكتُشف تعديل:
```
🚨 3 سجلات تم العبث بها
→  تنبيه فوري لمدير النظام
→  السجلات المشبوهة تُعزل
→  يبدأ تحقيق أمني
```

---

## 6. حماية بيانات المريض (HIPAA / GDPR)

| البيانات | كيف تُعالَج |
|---|---|
| `patient_id` الخام | **لا يُخزَّن أبداً** |
| `patient_id_hash` | SHA-256 + SALT من `.env` |
| اسم المريض | غير مطلوب في السجلات |
| الأعراض | تُخزَّن مشفّرة في `patients.encrypted` |
| تاريخ الميلاد | السن فقط (عدد صحيح) |

### الـ SALT:
```bash
# في .env
PATIENT_ID_SALT=your-secret-salt-min-32-chars
```

> ⚠️ بدون الـ SALT، لا يمكن استرداد أي `patient_id` من الهاش حتى بالقوة الغاشمة.

---

## 7. Retraining Loop — دورة التحسين المستمر

### متى تُطلق إشارة إعادة التدريب؟

```
حالة 1: Flag فوري (أي دكتور)
    → Priority: CRITICAL
    → تنبيه Slack فوري

حالة 2: تراكم نقاط سلبية
    → النقاط = Σ (feedback_weight × specialty_weight)
    → عند تجاوز threshold → Priority: HIGH/MEDIUM/LOW

حالة 3: تقييم متكرر ≤ 2 نجوم على نفس intent
    → بعد NEGATIVE_THRESHOLD ردود → Signal
```

### أولويات إعادة التدريب:

| الأولوية | الإجراء | الوقت المستهدف |
|---|---|---|
| **CRITICAL** | تدريب طارئ + مراجعة يدوية | < 24 ساعة |
| **HIGH** | إعادة تدريب في الدورة القادمة | < أسبوع |
| **MEDIUM** | إضافة للـ backlog | < شهر |
| **LOW** | مراجعة في التحديث الدوري | < ربع سنة |

---

## 8. متغيرات البيئة المطلوبة

```bash
# في .env

# PII Protection
PATIENT_ID_SALT=change-this-to-random-32-char-string

# Storage (اختر واحد)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key

# أو
VERCEL_KV_URL=https://your-kv.kv.vercel-storage.com
VERCEL_KV_TOKEN=your-vercel-kv-token

# Alerts
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# AI Confidence
LOW_CONFIDENCE_THRESH=0.55
```

---

## 9. API Reference السريع

### clinical_override_log.py
```python
# تسجيل override
override_id = log_override(
    doctor_id       = "DR-001",
    patient_id      = "PT-4892",
    session_id      = "session-uuid",
    ai_intent       = "Triage",
    ai_confidence   = 0.42,
    ai_suggestion   = "راجع طبيب...",
    doctor_decision = "إدخال فوري",
    override_reason = OverrideReason.AI_UNDERESTIMATED,
    severity        = Severity.CRITICAL,
    reason_notes    = "أعراض جلطة",
)

# تحديث النتيجة (O(1) - بدون إعادة كتابة الملف)
update_outcome(override_id, Outcome.IMPROVED)

# تحقق من سلامة السجلات
report = verify_audit_log()
# {"total": 150, "intact": 150, "tampered": 0, "status": "clean"}
```

### doctor_feedback_handler.py
```python
# تقييم
submit_rating(doctor_id, patient_id, session_id,
              ai_intent, ai_confidence, ai_response,
              rating=4, doctor_specialty="specialist")

# تصحيح
submit_correction(doctor_id, patient_id, session_id,
                  ai_intent, ai_confidence, ai_response,
                  corrected_response="الرد الصحيح هو...")

# إبلاغ عن خطر
submit_flag(doctor_id, patient_id, session_id,
            ai_intent, ai_confidence, ai_response,
            flag_reason=FlagReason.MISSED_EMERGENCY,
            flag_notes="المريض عنده أعراض خطيرة أُغفلت",
            doctor_specialty="consultant")

# إحصائيات
summary = get_summary()
signals = get_signals(priority=RetrainingPriority.CRITICAL)
```

---

## 10. الربط مع الـ Frontend

في `09_doctor_dashboard.html`:
```javascript
// الدكتور يضغط "تجاوز قرار AI"
async function submitOverride(sessionId, patientId) {
    const res = await fetch('/doctor/override', {
        method: 'POST',
        body: JSON.stringify({
            session_id:      sessionId,
            patient_id:      patientId,
            doctor_decision: document.getElementById('decision').value,
            override_reason: document.getElementById('reason').value,
            severity:        document.getElementById('severity').value,
            reason_notes:    document.getElementById('notes').value,
        })
    });
    const data = await res.json();
    showToast(`✓ تم التسجيل — ID: ${data.override_id}`);
}
```

في `12_ai_explanation.html`:
```javascript
// المريض يصحّح معلوماته
async function saveCorrection(sessionId, updatedProfile) {
    await fetch(`/chat/session/${sessionId}/profile`, {
        method: 'PATCH',
        body:   JSON.stringify(updatedProfile)
    });
    // RIVA ستعيد التقييم في الطلب التالي
    window.location.href = '02_chatbot.html';
}
```

---

## 11. قائمة التحقق قبل النشر

- [ ] `PATIENT_ID_SALT` تم تعيينه في `.env` (مش القيمة الافتراضية)
- [ ] قاعدة بيانات السحابة متصلة (Supabase أو Vercel KV)
- [ ] `SLACK_WEBHOOK_URL` مُعيَّن للتنبيهات الحرجة
- [ ] `httpx` مضاف في `requirements-ai.txt`
- [ ] `verify_audit_log()` تُشغَّل في CI/CD pipeline
- [ ] الدكاترة حاصلون على `doctor_specialty` صحيح في ملفاتهم
- [ ] `LOW_CONFIDENCE_THRESH` مضبوط حسب احتياج العيادة

---

*RIVA Health Platform — Offline AI-Driven Health Systems Resilience*
*Harvard HSIL Hackathon 2026*
