/**
 * explanation_viewer.js
 * =====================
 * RIVA Health Platform - AI Explanation Viewer
 * عارض شروحات الذكاء الاصطناعي (SHAP values)
 * 
 * المسؤوليات:
 * - عرض شروحات SHAP values بشكل تفاعلي
 * - دعم مستويات الشرح (Patient / Clinical / Expert)
 * - عرض أهم العوامل المؤثرة في القرار
 * - تكامل مع واجهة المستخدم للدكتور والمريض
 * 
 * المسار: web-app/src/static/js/explanation_viewer.js
 * 
 * التحسينات:
 * - فصل المهام (Separation of Concerns)
 * - حماية من XSS باستخدام DOMPurify
 * - استخدام Static Translations
 * - Clean Code Architecture
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
// 1. ترجمات الميزات (Static)
// ──────────────────────────────────────────────────────────

const FeatureTranslations = {
    translations: {
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
        'temperature': 'درجة الحرارة'
    },
    
    get(featureName) {
        return this.translations[featureName] || featureName;
    }
};

// ──────────────────────────────────────────────────────────
// 2. كلاس عرض واجهة المستخدم (UI Layer)
// ──────────────────────────────────────────────────────────

class ExplanationViewerUI {
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
            console.warn(`[ExplanationViewer] Element ${elementId} not found`);
            return;
        }
        
        if (!explanation) {
            element.innerHTML = await this.renderEmpty();
            return;
        }
        
        const container = document.createElement('div');
        container.className = 'explanation-viewer';
        
        // Header with level selector
        const header = await this.createHeader(explanation.level);
        container.appendChild(header);
        
        // Summary section
        const summarySection = await this.createSummarySection(explanation);
        container.appendChild(summarySection);
        
        // Top factors section
        if (explanation.topFactors && explanation.topFactors.length > 0) {
            const factorsSection = await this.createFactorsSection(explanation.topFactors);
            container.appendChild(factorsSection);
        }
        
        // Recommendations section
        if (explanation.recommendations && explanation.recommendations.length > 0) {
            const recommendationsSection = await this.createRecommendationsSection(explanation.recommendations);
            container.appendChild(recommendationsSection);
        }
        
        // Confidence section
        if (explanation.confidenceScore) {
            const confidenceSection = await this.createConfidenceSection(explanation);
            container.appendChild(confidenceSection);
        }
        
        // SHAP values section (for expert level)
        if (explanation.level === 'expert' && explanation.shapValues) {
            const shapSection = await this.createShapSection(explanation.shapValues);
            container.appendChild(shapSection);
        }
        
        element.innerHTML = '';
        element.appendChild(container);
        
        this.injectStyles();
    }
    
    async createHeader(level) {
        const header = document.createElement('div');
        header.className = 'explanation-header';
        
        const title = document.createElement('h3');
        title.textContent = '🧠 شرح قرار الذكاء الاصطناعي';
        header.appendChild(title);
        
        const levelSelector = document.createElement('div');
        levelSelector.className = 'level-selector';
        
        const levels = [
            { value: 'patient', label: '👤 للمريض', icon: '👤' },
            { value: 'clinical', label: '👨‍⚕️ سريري', icon: '👨‍⚕️' },
            { value: 'expert', label: '🔬 متقدم', icon: '🔬' }
        ];
        
        levels.forEach(lvl => {
            const btn = document.createElement('button');
            btn.className = `level-btn ${level === lvl.value ? 'active' : ''}`;
            btn.setAttribute('data-level', lvl.value);
            btn.textContent = lvl.label;
            
            btn.addEventListener('click', () => {
                window.dispatchEvent(new CustomEvent('riva-change-explanation-level', {
                    detail: { level: lvl.value }
                }));
            });
            
            levelSelector.appendChild(btn);
        });
        
        header.appendChild(levelSelector);
        
        return header;
    }
    
    async createSummarySection(explanation) {
        const section = document.createElement('div');
        section.className = 'explanation-summary';
        
        const title = document.createElement('h4');
        title.textContent = '📋 ملخص القرار';
        section.appendChild(title);
        
        const text = await this.sanitize(explanation.summary);
        const paragraph = document.createElement('p');
        paragraph.textContent = text;
        section.appendChild(paragraph);
        
        return section;
    }
    
    async createFactorsSection(topFactors) {
        const section = document.createElement('div');
        section.className = 'explanation-factors';
        
        const title = document.createElement('h4');
        title.textContent = '🔍 أهم العوامل المؤثرة';
        section.appendChild(title);
        
        const list = document.createElement('div');
        list.className = 'factors-list';
        
        for (const factor of topFactors.slice(0, 5)) {
            const item = await this.createFactorItem(factor);
            list.appendChild(item);
        }
        
        section.appendChild(list);
        return section;
    }
    
    async createFactorItem(factor) {
        const item = document.createElement('div');
        const impactClass = factor.impact === 'positive' ? 'positive-impact' : 'negative-impact';
        item.className = `factor-item ${impactClass}`;
        
        // Name
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
        
        // Impact bar
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
        
        // Clinical meaning
        if (factor.clinicalMeaning) {
            const meaningDiv = document.createElement('div');
            meaningDiv.className = 'factor-meaning';
            meaningDiv.textContent = await this.sanitize(factor.clinicalMeaning);
            item.appendChild(meaningDiv);
        }
        
        return item;
    }
    
    async createRecommendationsSection(recommendations) {
        const section = document.createElement('div');
        section.className = 'explanation-recommendations';
        
        const title = document.createElement('h4');
        title.textContent = '💡 التوصيات';
        section.appendChild(title);
        
        const list = document.createElement('ul');
        list.className = 'recommendations-list';
        
        for (const rec of recommendations) {
            const li = document.createElement('li');
            li.textContent = await this.sanitize(rec);
            list.appendChild(li);
        }
        
        section.appendChild(list);
        return section;
    }
    
    async createConfidenceSection(explanation) {
        const section = document.createElement('div');
        section.className = 'explanation-confidence';
        
        const title = document.createElement('h4');
        title.textContent = '🎯 درجة الثقة';
        section.appendChild(title);
        
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
        section.appendChild(meter);
        
        if (explanation.isFallback) {
            const note = document.createElement('p');
            note.className = 'fallback-note';
            note.textContent = '⚠️ هذا شرح افتراضي لعدم توفر البيانات الكافية';
            section.appendChild(note);
        }
        
        return section;
    }
    
    async createShapSection(shapValues) {
        const section = document.createElement('div');
        section.className = 'explanation-shap';
        
        const title = document.createElement('h4');
        title.textContent = '📊 قيم SHAP (المتقدمة)';
        section.appendChild(title);
        
        const container = document.createElement('div');
        container.className = 'shap-container';
        
        const entries = Object.entries(shapValues).slice(0, 10);
        
        for (const [key, value] of entries) {
            const row = document.createElement('div');
            row.className = 'shap-row';
            
            const nameSpan = document.createElement('span');
            nameSpan.className = 'shap-name';
            nameSpan.textContent = FeatureTranslations.get(key);
            row.appendChild(nameSpan);
            
            const barContainer = document.createElement('div');
            barContainer.className = 'shap-bar-container';
            
            const bar = document.createElement('div');
            bar.className = 'shap-bar';
            const absValue = Math.abs(value);
            const barWidth = Math.min(absValue * 100, 100);
            bar.style.width = `${barWidth}%`;
            bar.style.backgroundColor = value > 0 ? '#ea4335' : '#34a853';
            barContainer.appendChild(bar);
            
            row.appendChild(barContainer);
            
            const valueSpan = document.createElement('span');
            valueSpan.className = 'shap-value';
            valueSpan.textContent = value.toFixed(3);
            row.appendChild(valueSpan);
            
            container.appendChild(row);
        }
        
        section.appendChild(container);
        return section;
    }
    
    async renderEmpty() {
        const sanitized = await this.sanitize('لا يوجد شرح متاح حالياً');
        return `<p class="text-muted">${sanitized}</p>`;
    }
    
    injectStyles() {
        if (this.stylesInjected) return;
        if (document.getElementById('riva-explanation-viewer-styles')) return;
        
        const styles = `
            <style id="riva-explanation-viewer-styles">
                .explanation-viewer {
                    background: var(--white, #ffffff);
                    border-radius: 16px;
                    padding: 24px;
                    margin: 16px 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                }
                
                .explanation-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    flex-wrap: wrap;
                    gap: 16px;
                    margin-bottom: 24px;
                    padding-bottom: 16px;
                    border-bottom: 1px solid var(--gray-lighter, #e8eaed);
                }
                
                .explanation-header h3 {
                    margin: 0;
                    font-size: 18px;
                    font-weight: 600;
                }
                
                .level-selector {
                    display: flex;
                    gap: 8px;
                }
                
                .level-btn {
                    padding: 6px 12px;
                    border: 1px solid var(--gray-lighter, #e8eaed);
                    border-radius: 20px;
                    background: transparent;
                    cursor: pointer;
                    font-size: 12px;
                    transition: all 0.2s ease;
                }
                
                .level-btn:hover {
                    background: var(--light, #f8f9fa);
                }
                
                .level-btn.active {
                    background: var(--primary, #1a73e8);
                    border-color: var(--primary, #1a73e8);
                    color: white;
                }
                
                .explanation-summary p {
                    line-height: 1.6;
                    margin: 12px 0;
                }
                
                .factors-list {
                    margin: 16px 0;
                }
                
                .factor-item {
                    background: var(--light, #f8f9fa);
                    border-radius: 12px;
                    padding: 12px 16px;
                    margin-bottom: 12px;
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
                    margin-bottom: 10px;
                }
                
                .factor-icon {
                    font-size: 14px;
                }
                
                .factor-impact {
                    display: flex;
                    align-items: center;
                    gap: 12px;
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
                
                .factor-value {
                    font-size: 12px;
                    font-weight: 500;
                    min-width: 45px;
                }
                
                .factor-meaning {
                    font-size: 12px;
                    color: var(--gray, #5f6368);
                    margin-top: 8px;
                    padding-top: 8px;
                    border-top: 1px dashed var(--gray-lighter, #e8eaed);
                }
                
                .recommendations-list {
                    list-style: none;
                    padding: 0;
                    margin: 12px 0;
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
                    height: 32px;
                    position: relative;
                    overflow: hidden;
                    margin: 12px 0;
                }
                
                .confidence-bar {
                    height: 100%;
                    transition: width 0.5s ease;
                    display: flex;
                    align-items: center;
                    justify-content: flex-end;
                    padding-right: 12px;
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
                    right: 12px;
                    top: 50%;
                    transform: translateY(-50%);
                    font-size: 12px;
                    font-weight: bold;
                    color: var(--dark, #202124);
                    background: rgba(255,255,255,0.9);
                    padding: 2px 8px;
                    border-radius: 20px;
                }
                
                .shap-container {
                    background: var(--light, #f8f9fa);
                    border-radius: 12px;
                    padding: 16px;
                    margin-top: 16px;
                }
                
                .shap-row {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    margin-bottom: 12px;
                }
                
                .shap-name {
                    width: 100px;
                    font-size: 12px;
                    font-weight: 500;
                }
                
                .shap-bar-container {
                    flex: 1;
                    height: 24px;
                    background: var(--gray-lighter, #e8eaed);
                    border-radius: 4px;
                    overflow: hidden;
                }
                
                .shap-bar {
                    height: 100%;
                    transition: width 0.3s ease;
                }
                
                .shap-value {
                    width: 50px;
                    font-size: 11px;
                    text-align: right;
                    font-family: monospace;
                }
                
                .fallback-note {
                    font-size: 12px;
                    color: var(--warning, #fbbc04);
                    margin-top: 12px;
                    font-style: italic;
                }
                
                @media (max-width: 768px) {
                    .explanation-viewer {
                        padding: 16px;
                    }
                    
                    .explanation-header {
                        flex-direction: column;
                        align-items: flex-start;
                    }
                    
                    .factor-impact {
                        flex-direction: column;
                        align-items: flex-start;
                    }
                    
                    .factor-bar-container {
                        width: 100%;
                    }
                    
                    .shap-row {
                        flex-wrap: wrap;
                    }
                    
                    .shap-name {
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

class ExplanationViewer {
    constructor() {
        this.ui = new ExplanationViewerUI();
        this.currentExplanation = null;
        this.currentElementId = null;
        
        this.setupEventListeners();
        this.init();
    }
    
    init() {
        console.log('[ExplanationViewer] Initialized');
    }
    
    setupEventListeners() {
        window.addEventListener('riva-change-explanation-level', async (event) => {
            const { level } = event.detail;
            if (this.currentExplanation) {
                this.currentExplanation.level = level;
                await this.refresh();
            }
        });
    }
    
    async loadExplanation(patientId, predictionType, features, options = {}) {
        try {
            // استخدام الـ AIExplainability module
            if (window.aiExplainability) {
                const explanation = await window.aiExplainability.getExplanation(
                    patientId,
                    predictionType,
                    features,
                    options
                );
                
                this.currentExplanation = explanation;
                return explanation;
            }
            
            // Fallback if AIExplainability not available
            console.warn('[ExplanationViewer] AIExplainability not available');
            return this.getFallbackExplanation(predictionType);
            
        } catch (error) {
            console.error('[ExplanationViewer] Failed to load explanation:', error);
            return this.getFallbackExplanation(predictionType);
        }
    }
    
    async display(elementId, patientId, predictionType, features, options = {}) {
        this.currentElementId = elementId;
        
        const explanation = await this.loadExplanation(patientId, predictionType, features, options);
        
        if (explanation) {
            await this.ui.render(elementId, explanation);
        }
    }
    
    async refresh() {
        if (this.currentElementId && this.currentExplanation) {
            await this.ui.render(this.currentElementId, this.currentExplanation);
        }
    }
    
    getFallbackExplanation(predictionType) {
        const fallback = {
            level: 'clinical',
            summary: '',
            topFactors: [],
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
    
    clear() {
        this.currentExplanation = null;
        if (this.currentElementId) {
            const element = document.getElementById(this.currentElementId);
            if (element) {
                element.innerHTML = '';
            }
        }
    }
}

// ──────────────────────────────────────────────────────────
// 4. تصدير نسخة واحدة (Singleton)
// ──────────────────────────────────────────────────────────

const explanationViewer = new ExplanationViewer();

window.explanationViewer = explanationViewer;

export default explanationViewer;
