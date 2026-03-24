/**
 * explainability.js
 * =================
 * RIVA Health Platform - AI Explainability Module
 * وحدة شرح قرارات الذكاء الاصطناعي
 * 
 * المسؤوليات:
 * - توليد شروحات SHAP values من الـ Backend
 * - عرض أهم العوامل المؤثرة في القرار
 * - توفير تفسيرات طبية بالعربية للمرضى
 * - دعم مستويات الشرح (Patient / Clinical / Expert)
 * 
 * المسار: web-app/src/static/js/explainability.js
 * 
 * التحسينات:
 * - فصل دوال العرض في كلاس منفصل (AIExplainabilityUI)
 * - حماية من XSS باستخدام DOMPurify
 * - تحسين أداء الترجمة باستخدام Static Property
 * - Clean Code & Separation of Concerns
 */

// ──────────────────────────────────────────────────────────
// 0. تحميل DOMPurify للحماية من XSS
// ──────────────────────────────────────────────────────────

let purifyLoaded = false;
let purifyInstance = null;

async function loadDOMPurify() {
    if (purifyLoaded) return purifyInstance;
    
    return new Promise((resolve) => {
        if (typeof window.DOMPurify !== 'undefined') {
            purifyInstance = window.DOMPurify;
            purifyLoaded = true;
            resolve(purifyInstance);
            return;
        }
        
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/dompurify@3.0.6/dist/purify.min.js';
        script.onload = () => {
            purifyInstance = window.DOMPurify;
            purifyLoaded = true;
            resolve(purifyInstance);
        };
        document.head.appendChild(script);
    });
}

// ──────────────────────────────────────────────────────────
// 1. كلاس الترجمة (Static Translations)
// ──────────────────────────────────────────────────────────

class FeatureTranslations {
    static translations = {
        'age': 'السن',
        'has_diabetes': 'السكري',
        'has_hypertension': 'ضغط الدم',
        'has_heart_failure': 'فشل القلب',
        'has_kidney_disease': 'أمراض الكلى',
        'medication_count': 'عدد الأدوية',
        'symptom_count': 'عدد الأعراض',
        'previous_readmission': 'دخول سابق للمستشفى',
        'glucose': 'مستوى السكر',
        'blood_pressure': 'ضغط الدم',
        'bmi': 'مؤشر كتلة الجسم',
        'heart_rate': 'نبض القلب',
        'oxygen_saturation': 'نسبة الأكسجين',
        'temperature': 'درجة الحرارة',
        'pulse_pressure': 'الضغط النبضي',
        'bp_ratio': 'نسبة الضغط',
        'temp_fever': 'الحمى'
    };
    
    static patientTranslations = {
        'احتمالية إعادة الدخول': 'فرصة رجوعك للمستشفى تاني',
        'مدة الإقامة المتوقعة': 'العدد المتوقع لأيامك في المستشفى',
        'الخطر مرتفع': 'الوضع محتاج اهتمام',
        'الخطر متوسط': 'الوضع مستقر',
        'الخطر منخفض': 'الوضع كويس',
        'فشل القلب': 'مشكلة في القلب',
        'السكري': 'مرض السكر',
        'ارتفاع ضغط الدم': 'ضغط الدم العالي',
        'متابعة هاتفية': 'هنتكلم معاك تلفون',
        'مراجعة طبية': 'روح للدكتور',
        'مراجعة الأدوية': 'راجع الأدوية اللي بتاخدها',
        'متابعة مكثفة': 'هنتابع معاك كتير',
        'تدخل طبي فوري': 'روح المستشفى دلوقتي',
        'فحوصات دورية': 'اعمل التحاليل بانتظام',
        'تخطيط للخروج': 'تجهيز للخروج من المستشفى',
        'يزيد خطر إعادة الدخول': 'عشان كدا لازم نتابع معاك',
        'يؤثر على التئام الجروح': 'عشان كدا الجرح محتاج وقت',
        'يزيد خطر العدوى': 'نخلي بالنا من الالتهابات',
        'يحتاج متابعة مكثفة': 'نتابع معاك عن قرب',
        'يحتاج تدخل فوري': 'لازم نتصرف بسرعة'
    };
    
    static get(featureName) {
        return this.translations[featureName] || featureName;
    }
    
    static simplify(text) {
        let simplified = text;
        for (const [key, value] of Object.entries(this.patientTranslations)) {
            if (text.includes(key)) {
                simplified = value;
                break;
            }
        }
        return simplified;
    }
}

// ──────────────────────────────────────────────────────────
// 2. كلاس عرض واجهة المستخدم (UI Layer)
// ──────────────────────────────────────────────────────────

class AIExplainabilityUI {
    constructor() {
        this.stylesInjected = false;
    }
    
    async sanitize(text) {
        await loadDOMPurify();
        return purifyInstance ? purifyInstance.sanitize(text) : text;
    }
    
    async render(elementId, explanation) {
        const element = document.getElementById(elementId);
        if (!element) {
            console.warn(`[AIExplainabilityUI] Element ${elementId} not found`);
            return;
        }
        
        if (!explanation) {
            element.innerHTML = await this.renderEmpty();
            return;
        }
        
        // بناء العناصر باستخدام DOM API (آمن من XSS)
        const container = document.createElement('div');
        container.className = 'explanation-container';
        
        // الملخص
        const summaryDiv = await this.createSummarySection(explanation);
        container.appendChild(summaryDiv);
        
        // أهم العوامل المؤثرة
        if (explanation.topFactors && explanation.topFactors.length > 0) {
            const factorsDiv = await this.createFactorsSection(explanation.topFactors);
            container.appendChild(factorsDiv);
        }
        
        // التوصيات
        if (explanation.recommendations && explanation.recommendations.length > 0) {
            const recommendationsDiv = await this.createRecommendationsSection(explanation.recommendations);
            container.appendChild(recommendationsDiv);
        }
        
        // درجة الثقة
        if (explanation.confidenceScore) {
            const confidenceDiv = await this.createConfidenceSection(explanation);
            container.appendChild(confidenceDiv);
        }
        
        // تنظيف المحتوى القديم وإضافة الجديد
        element.innerHTML = '';
        element.appendChild(container);
        
        // إضافة الأنماط
        this.injectStyles();
    }
    
    async createSummarySection(explanation) {
        const div = document.createElement('div');
        div.className = 'explanation-summary';
        
        const title = document.createElement('h4');
        title.textContent = '📋 ملخص القرار';
        div.appendChild(title);
        
        const summaryText = await this.sanitize(explanation.summary);
        const paragraph = document.createElement('p');
        paragraph.textContent = summaryText;
        div.appendChild(paragraph);
        
        return div;
    }
    
    async createFactorsSection(topFactors) {
        const div = document.createElement('div');
        div.className = 'explanation-factors';
        
        const title = document.createElement('h4');
        title.textContent = '🔍 أهم العوامل المؤثرة';
        div.appendChild(title);
        
        const factorsList = document.createElement('div');
        factorsList.className = 'factors-list';
        
        for (const factor of topFactors.slice(0, 5)) {
            const factorItem = await this.createFactorItem(factor);
            factorsList.appendChild(factorItem);
        }
        
        div.appendChild(factorsList);
        return div;
    }
    
    async createFactorItem(factor) {
        const item = document.createElement('div');
        const impactClass = factor.impact === 'positive' ? 'positive-impact' : 'negative-impact';
        item.className = `factor-item ${impactClass}`;
        
        // اسم العامل
        const nameDiv = document.createElement('div');
        nameDiv.className = 'factor-name';
        
        const iconSpan = document.createElement('span');
        iconSpan.className = 'factor-icon';
        iconSpan.textContent = factor.impact === 'positive' ? '⬆️' : '⬇️';
        nameDiv.appendChild(iconSpan);
        
        const nameStrong = document.createElement('strong');
        const displayName = await this.sanitize(factor.nameAr || factor.name);
        nameStrong.textContent = displayName;
        nameDiv.appendChild(nameStrong);
        
        item.appendChild(nameDiv);
        
        // شريط التأثير
        const impactDiv = document.createElement('div');
        impactDiv.className = 'factor-impact';
        
        const barContainer = document.createElement('div');
        barContainer.className = 'factor-bar-container';
        
        const bar = document.createElement('div');
        bar.className = 'factor-bar';
        const barWidth = Math.abs(factor.shapValue * 100);
        bar.style.width = `${barWidth}%`;
        barContainer.appendChild(bar);
        
        impactDiv.appendChild(barContainer);
        
        const valueSpan = document.createElement('span');
        valueSpan.className = 'factor-value';
        valueSpan.textContent = `${barWidth.toFixed(1)}%`;
        impactDiv.appendChild(valueSpan);
        
        item.appendChild(impactDiv);
        
        // المعنى السريري
        if (factor.clinicalMeaning) {
            const meaningDiv = document.createElement('div');
            meaningDiv.className = 'factor-meaning';
            meaningDiv.textContent = await this.sanitize(factor.clinicalMeaning);
            item.appendChild(meaningDiv);
        }
        
        return item;
    }
    
    async createRecommendationsSection(recommendations) {
        const div = document.createElement('div');
        div.className = 'explanation-recommendations';
        
        const title = document.createElement('h4');
        title.textContent = '💡 التوصيات';
        div.appendChild(title);
        
        const list = document.createElement('ul');
        list.className = 'recommendations-list';
        
        for (const rec of recommendations) {
            const li = document.createElement('li');
            li.textContent = await this.sanitize(rec);
            list.appendChild(li);
        }
        
        div.appendChild(list);
        return div;
    }
    
    async createConfidenceSection(explanation) {
        const div = document.createElement('div');
        div.className = 'explanation-confidence';
        
        const title = document.createElement('h4');
        title.textContent = '🎯 درجة الثقة';
        div.appendChild(title);
        
        const confidencePercent = (explanation.confidenceScore * 100).toFixed(0);
        const confidenceClass = explanation.confidenceScore > 0.7 ? 'high' : 
                                explanation.confidenceScore > 0.4 ? 'medium' : 'low';
        
        const meter = document.createElement('div');
        meter.className = 'confidence-meter';
        
        const bar = document.createElement('div');
        bar.className = `confidence-bar ${confidenceClass}`;
        bar.style.width = `${confidencePercent}%`;
        
        const valueSpan = document.createElement('span');
        valueSpan.className = 'confidence-value';
        valueSpan.textContent = `${confidencePercent}%`;
        
        meter.appendChild(bar);
        meter.appendChild(valueSpan);
        div.appendChild(meter);
        
        if (explanation.isFallback) {
            const note = document.createElement('p');
            note.className = 'fallback-note';
            note.textContent = '⚠️ هذا شرح افتراضي لعدم توفر البيانات الكافية';
            div.appendChild(note);
        }
        
        return div;
    }
    
    async renderEmpty() {
        const sanitized = await this.sanitize('لا يوجد شرح متاح حالياً');
        return `<p class="text-muted">${sanitized}</p>`;
    }
    
    injectStyles() {
        if (this.stylesInjected) return;
        if (document.getElementById('riva-explainability-styles')) return;
        
        const styles = `
            <style id="riva-explainability-styles">
                .explanation-container {
                    background: var(--white, #ffffff);
                    border-radius: 12px;
                    padding: 20px;
                    margin: 15px 0;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }
                
                .explanation-summary p {
                    line-height: 1.6;
                    color: var(--dark, #202124);
                }
                
                .factors-list {
                    margin: 15px 0;
                }
                
                .factor-item {
                    background: var(--light, #f8f9fa);
                    border-radius: 8px;
                    padding: 12px;
                    margin-bottom: 10px;
                    transition: all 0.2s ease;
                }
                
                .factor-item:hover {
                    transform: translateX(-4px);
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }
                
                .factor-name {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    margin-bottom: 8px;
                }
                
                .factor-icon {
                    font-size: 14px;
                }
                
                .factor-bar-container {
                    flex: 1;
                    height: 8px;
                    background: var(--gray-lighter, #e8eaed);
                    border-radius: 4px;
                    overflow: hidden;
                }
                
                .factor-bar {
                    height: 100%;
                    background: var(--primary, #1a73e8);
                    border-radius: 4px;
                    transition: width 0.3s ease;
                }
                
                .positive-impact .factor-bar {
                    background: var(--danger, #ea4335);
                }
                
                .negative-impact .factor-bar {
                    background: var(--success, #34a853);
                }
                
                .factor-impact {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    margin: 8px 0;
                }
                
                .factor-value {
                    font-size: 12px;
                    font-weight: 500;
                    min-width: 45px;
                }
                
                .factor-meaning {
                    font-size: 12px;
                    color: var(--gray, #5f6368);
                    margin-top: 6px;
                }
                
                .recommendations-list {
                    list-style: none;
                    padding: 0;
                }
                
                .recommendations-list li {
                    padding: 8px 0 8px 24px;
                    position: relative;
                }
                
                .recommendations-list li:before {
                    content: "✓";
                    position: absolute;
                    right: 0;
                    color: var(--success, #34a853);
                    font-weight: bold;
                }
                
                .confidence-meter {
                    background: var(--gray-lighter, #e8eaed);
                    border-radius: 20px;
                    height: 30px;
                    position: relative;
                    overflow: hidden;
                    margin: 10px 0;
                }
                
                .confidence-bar {
                    height: 100%;
                    transition: width 0.5s ease;
                    display: flex;
                    align-items: center;
                    justify-content: flex-end;
                    padding-right: 10px;
                }
                
                .confidence-bar.high {
                    background: linear-gradient(90deg, var(--success, #34a853), var(--success, #34a853));
                }
                
                .confidence-bar.medium {
                    background: linear-gradient(90deg, var(--warning, #fbbc04), var(--warning, #fbbc04));
                }
                
                .confidence-bar.low {
                    background: linear-gradient(90deg, var(--danger, #ea4335), var(--danger, #ea4335));
                }
                
                .confidence-value {
                    position: absolute;
                    right: 10px;
                    top: 50%;
                    transform: translateY(-50%);
                    font-size: 12px;
                    font-weight: bold;
                    color: var(--dark, #202124);
                    background: rgba(255,255,255,0.8);
                    padding: 2px 8px;
                    border-radius: 20px;
                }
                
                .fallback-note {
                    font-size: 12px;
                    color: var(--warning, #fbbc04);
                    margin-top: 10px;
                    font-style: italic;
                }
                
                @media (max-width: 768px) {
                    .explanation-container {
                        padding: 15px;
                    }
                    
                    .factor-impact {
                        flex-direction: column;
                        align-items: flex-start;
                    }
                    
                    .factor-bar-container {
                        width: 100%;
                    }
                }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
        this.stylesInjected = true;
    }
}

// ──────────────────────────────────────────────────────────
// 3. الكلاس الرئيسي (Business Logic)
// ──────────────────────────────────────────────────────────

class AIExplainability {
    constructor() {
        this.apiClient = window.rivaClient || null;
        this.ui = new AIExplainabilityUI();
        this.currentExplanation = null;
        this.explanationLevel = 'clinical';
        this.cache = new Map();
        
        this.init();
    }

    // ──────────────────────────────────────────────────────────
    // 1. التهيئة
    // ──────────────────────────────────────────────────────────

    init() {
        console.log('[AIExplainability] Initialized');
        
        if (!this.apiClient) {
            window.addEventListener('riva-client-ready', () => {
                this.apiClient = window.rivaClient;
                console.log('[AIExplainability] API Client connected');
            });
        }
    }

    // ──────────────────────────────────────────────────────────
    // 2. جلب شرح القرار
    // ──────────────────────────────────────────────────────────

    async getExplanation(patientId, predictionType, features, options = {}) {
        const cacheKey = `${patientId}_${predictionType}_${this.explanationLevel}`;
        
        if (this.cache.has(cacheKey) && !options.forceRefresh) {
            console.log('[AIExplainability] Using cached explanation');
            return this.cache.get(cacheKey);
        }
        
        try {
            let response;
            
            if (options.simple) {
                response = await this.apiClient.getSimpleExplanation(
                    patientId,
                    predictionType,
                    features
                );
            } else {
                response = await this.apiClient.generateExplanation(
                    patientId,
                    predictionType,
                    features
                );
            }
            
            if (response && response.success) {
                this.currentExplanation = this.formatExplanation(response, this.explanationLevel);
                this.cache.set(cacheKey, this.currentExplanation);
                
                window.dispatchEvent(new CustomEvent('riva-explanation-updated', {
                    detail: { explanation: this.currentExplanation }
                }));
                
                return this.currentExplanation;
            }
            
            throw new Error(response?.detail || 'Failed to get explanation');
            
        } catch (error) {
            console.error('[AIExplainability] Failed to get explanation:', error);
            return this.getFallbackExplanation(predictionType);
        }
    }

    // ──────────────────────────────────────────────────────────
    // 3. تنسيق الشرح حسب مستوى المستخدم
    // ──────────────────────────────────────────────────────────

    formatExplanation(rawExplanation, level = 'clinical') {
        const formatted = {
            level: level,
            summary: '',
            topFactors: [],
            clinicalImplications: [],
            recommendations: [],
            confidenceScore: 0,
            rawShapValues: rawExplanation.full_shap_values || {}
        };
        
        if (rawExplanation.top_features && rawExplanation.top_features.length > 0) {
            formatted.topFactors = rawExplanation.top_features.map(factor => ({
                name: factor.feature_name,
                nameAr: FeatureTranslations.get(factor.feature_name),
                shapValue: factor.shap_value,
                impact: factor.impact_direction,
                clinicalMeaning: factor.clinical_meaning
            }));
        }
        
        if (level === 'patient') {
            formatted.summary = FeatureTranslations.simplify(rawExplanation.summary || '');
            formatted.clinicalImplications = this.simplifyImplications(rawExplanation.clinical_recommendations || []);
            formatted.recommendations = this.simplifyRecommendations(rawExplanation.recommendations || []);
        } else if (level === 'expert') {
            formatted.summary = rawExplanation.summary || '';
            formatted.clinicalImplications = rawExplanation.clinical_recommendations || [];
            formatted.recommendations = rawExplanation.recommendations || [];
            formatted.shapValues = rawExplanation.full_shap_values || {};
        } else {
            formatted.summary = rawExplanation.summary || '';
            formatted.clinicalImplications = rawExplanation.clinical_recommendations || [];
            formatted.recommendations = rawExplanation.recommendations || [];
        }
        
        formatted.confidenceScore = rawExplanation.confidence_score || 0.85;
        
        return formatted;
    }

    // ──────────────────────────────────────────────────────────
    // 4. تبسيط التوصيات للمرضى
    // ──────────────────────────────────────────────────────────

    simplifyRecommendations(recommendations) {
        return recommendations.map(rec => FeatureTranslations.simplify(rec));
    }

    // ──────────────────────────────────────────────────────────
    // 5. تبسيط الآثار السريرية للمرضى
    // ──────────────────────────────────────────────────────────

    simplifyImplications(implications) {
        return implications.map(imp => FeatureTranslations.simplify(imp));
    }

    // ──────────────────────────────────────────────────────────
    // 6. شرح افتراضي في حالة الفشل
    // ──────────────────────────────────────────────────────────

    getFallbackExplanation(predictionType) {
        const fallback = {
            level: this.explanationLevel,
            summary: '',
            topFactors: [],
            clinicalImplications: [],
            recommendations: [],
            confidenceScore: 0.5,
            isFallback: true
        };
        
        if (predictionType === 'readmission') {
            fallback.summary = 'بناءً على البيانات المتاحة، هناك عوامل متعددة تؤثر على احتمالية إعادة الدخول. يوصى بمتابعة الحالة عن كثب.';
            fallback.recommendations = ['متابعة دورية', 'مراجعة الأدوية'];
        } else if (predictionType === 'los') {
            fallback.summary = 'مدة الإقامة المتوقعة تعتمد على الحالة السريرية واستجابة المريض للعلاج.';
            fallback.recommendations = ['تقييم يومي', 'تخطيط للخروج المبكر'];
        } else {
            fallback.summary = 'تم تحليل الحالة بناءً على المعايير السريرية المتاحة.';
            fallback.recommendations = ['استشارة الطبيب المعالج'];
        }
        
        return fallback;
    }

    // ──────────────────────────────────────────────────────────
    // 7. تغيير مستوى الشرح
    // ──────────────────────────────────────────────────────────

    setExplanationLevel(level) {
        if (!['patient', 'clinical', 'expert'].includes(level)) {
            console.warn(`[AIExplainability] Invalid level: ${level}`);
            return false;
        }
        
        this.explanationLevel = level;
        
        if (this.currentExplanation) {
            this.currentExplanation = this.formatExplanation(
                this.currentExplanation,
                level
            );
            
            window.dispatchEvent(new CustomEvent('riva-explanation-level-changed', {
                detail: { level: level, explanation: this.currentExplanation }
            }));
        }
        
        console.log(`[AIExplainability] Explanation level changed to: ${level}`);
        return true;
    }

    // ──────────────────────────────────────────────────────────
    // 8. عرض الشرح في عنصر HTML (واجهة موحدة)
    // ──────────────────────────────────────────────────────────

    async renderExplanation(elementId, explanation = null) {
        const data = explanation || this.currentExplanation;
        await this.ui.render(elementId, data);
    }

    // ──────────────────────────────────────────────────────────
    // 9. مسح الـ Cache
    // ──────────────────────────────────────────────────────────

    clearCache() {
        this.cache.clear();
        console.log('[AIExplainability] Cache cleared');
    }

    // ──────────────────────────────────────────────────────────
    // 10. الحصول على إحصائيات الشرح
    // ──────────────────────────────────────────────────────────

    getStats() {
        return {
            cacheSize: this.cache.size,
            currentLevel: this.explanationLevel,
            hasExplanation: this.currentExplanation !== null,
            isFallback: this.currentExplanation?.isFallback || false
        };
    }
}

// ──────────────────────────────────────────────────────────
// 11. تصدير نسخة واحدة (Singleton)
// ──────────────────────────────────────────────────────────

// إنشاء نسخة واحدة
const aiExplainability = new AIExplainability();

// تخزين في window للاستخدام العادي
window.aiExplainability = aiExplainability;
window.rivaExplainability = aiExplainability;

// ES Module export
export default aiExplainability;
export { aiExplainability };
