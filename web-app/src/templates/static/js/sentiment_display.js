/**
 * sentiment_display.js
 * ====================
 * RIVA Health Platform - Sentiment Analysis Display Module
 * وحدة عرض نتائج تحليل المشاعر
 * 
 * المسؤوليات:
 * - عرض نتائج تحليل المشاعر بشكل مرئي
 * - إظهار تنبيهات الطوارئ (Emergency Alerts)
 * - عرض تدرج المشاعر (Sentiment Gauge)
 * - إظهار الكلمات المفتاحية المستخرجة
 * - تكامل مع صفحة تحليل المشاعر ولوحة الدكتور
 * 
 * المسار: web-app/src/static/js/sentiment_display.js
 * 
 * التحسينات:
 * - دعم RTL للغة العربية
 * - عرض رسوم بيانية متحركة
 * - تنبيهات صوتية مع زر تفعيل (User Interaction Required)
 * - تكامل مع نظام الترياج
 * - معالجة النصوص الطويلة (word-break)
 * 
 * الإصدار: 4.2.1
 */

// ──────────────────────────────────────────────────────────
// 0. التحقق من تحميل ApexCharts
// ──────────────────────────────────────────────────────────

let apexChartsLoaded = false;
let apexChartsQueue = [];

function waitForApexCharts() {
    return new Promise((resolve) => {
        if (typeof ApexCharts !== 'undefined') {
            apexChartsLoaded = true;
            resolve();
            return;
        }
        
        const checkInterval = setInterval(() => {
            if (typeof ApexCharts !== 'undefined') {
                clearInterval(checkInterval);
                apexChartsLoaded = true;
                apexChartsQueue.forEach(cb => cb());
                apexChartsQueue = [];
                resolve();
            }
        }, 100);
        
        setTimeout(() => {
            clearInterval(checkInterval);
            if (!apexChartsLoaded) {
                console.error('[SentimentDisplay] ApexCharts failed to load');
                resolve();
            }
        }, 10000);
    });
}

// ──────────────────────────────────────────────────────────
// 1. كلاس Sentiment Display
// ──────────────────────────────────────────────────────────

class SentimentDisplay {
    constructor() {
        this.charts = {};
        this.initialized = false;
        this.isRTL = document.documentElement.dir === 'rtl' || document.body.dir === 'rtl';
        this.audioContext = null;
        this.emergencySound = null;
        this.soundEnabled = false;
        this.userInteracted = false;
        
        this.colors = {
            very_positive: '#34a853',
            positive: '#4caf50',
            neutral: '#fbbc04',
            negative: '#ea4335',
            very_negative: '#c5221f',
            emergency: '#d32f2f',
            background: '#ffffff',
            text: '#202124'
        };
        
        this.sentimentLabels = {
            ar: {
                very_positive: 'إيجابي جداً',
                positive: 'إيجابي',
                neutral: 'محايد',
                negative: 'سلبي',
                very_negative: 'سلبي جداً'
            },
            en: {
                very_positive: 'Very Positive',
                positive: 'Positive',
                neutral: 'Neutral',
                negative: 'Negative',
                very_negative: 'Very Negative'
            }
        };
        
        this.darkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
        
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            this.darkMode = e.matches;
            this.updateColors();
            this.updateAllCharts();
        });
        
        // إضافة مستمع للتفاعل الأول للمستخدم
        this.setupUserInteractionListener();
        
        this.init();
    }
    
    async init() {
        console.log('[SentimentDisplay] Initializing...');
        await waitForApexCharts();
        this.initialized = true;
        this.loadEmergencySound();
        this.injectStyles();
        console.log('[SentimentDisplay] Initialized', { isRTL: this.isRTL });
    }
    
    setupUserInteractionListener() {
        const enableSound = () => {
            if (!this.userInteracted) {
                this.userInteracted = true;
                console.log('[SentimentDisplay] User interaction detected, sound enabled');
                this.showSoundEnabledToast();
                document.removeEventListener('click', enableSound);
                document.removeEventListener('touchstart', enableSound);
                document.removeEventListener('keydown', enableSound);
            }
        };
        
        document.addEventListener('click', enableSound);
        document.addEventListener('touchstart', enableSound);
        document.addEventListener('keydown', enableSound);
    }
    
    loadEmergencySound() {
        try {
            this.emergencySound = new Audio('/static/assets/audio/emergency-alert.mp3');
            this.emergencySound.volume = 0.7;
            this.emergencySound.load();
        } catch (error) {
            console.warn('[SentimentDisplay] Failed to load emergency sound:', error);
        }
    }
    
    showSoundEnabledToast() {
        const toast = document.createElement('div');
        toast.className = 'sound-enabled-toast';
        toast.innerHTML = `
            <div class="toast-content">
                <span>🔊 ${this.isRTL ? 'تم تفعيل التنبيهات الصوتية' : 'Sound alerts enabled'}</span>
            </div>
        `;
        document.body.appendChild(toast);
        setTimeout(() => {
            toast.classList.add('fade-out');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
    
    showSoundDisabledWarning(containerId) {
        const container = this._getElement(containerId);
        if (!container) return;
        
        const warningHtml = `
            <div class="sound-warning">
                <span class="warning-icon">🔇</span>
                <span class="warning-text">${this.isRTL ? 'لتفعيل التنبيهات الصوتية، اضغط على أي مكان في الصفحة' : 'Click anywhere to enable sound alerts'}</span>
            </div>
        `;
        container.insertAdjacentHTML('afterbegin', warningHtml);
        setTimeout(() => {
            const warning = container.querySelector('.sound-warning');
            if (warning) warning.remove();
        }, 5000);
    }
    
    playEmergencySound() {
        if (this.userInteracted && this.emergencySound && this.soundEnabled) {
            this.emergencySound.play().catch(e => console.warn('Audio play failed:', e));
        }
    }
    
    enableSound() {
        this.soundEnabled = true;
        console.log('[SentimentDisplay] Sound enabled by user');
    }
    
    disableSound() {
        this.soundEnabled = false;
        console.log('[SentimentDisplay] Sound disabled');
    }
    
    updateColors() {
        if (this.darkMode) {
            this.colors.background = '#1e1e1e';
            this.colors.text = '#e8eaed';
        } else {
            this.colors.background = '#ffffff';
            this.colors.text = '#202124';
        }
    }
    
    _getElement(elementId) {
        const element = document.getElementById(elementId);
        if (!element) {
            console.warn(`[SentimentDisplay] Element "${elementId}" not found`);
            return null;
        }
        return element;
    }
    
    _isReady() {
        if (!this.initialized) {
            console.warn('[SentimentDisplay] Not initialized yet');
            return false;
        }
        if (typeof ApexCharts === 'undefined') {
            console.error('[SentimentDisplay] ApexCharts not loaded');
            return false;
        }
        return true;
    }
    
    _formatNumber(value, decimals = 1) {
        if (this.isRTL) {
            const arabicNumbers = {
                '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤',
                '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩'
            };
            const formatted = value.toFixed(decimals);
            return formatted.replace(/[0-9]/g, d => arabicNumbers[d]);
        }
        return value.toFixed(decimals);
    }
    
    getSentimentColor(level) {
        const colorMap = {
            'very_positive': this.colors.very_positive,
            'positive': this.colors.positive,
            'neutral': this.colors.neutral,
            'negative': this.colors.negative,
            'very_negative': this.colors.very_negative
        };
        return colorMap[level] || this.colors.neutral;
    }
    
    getSentimentLabel(level, language = 'ar') {
        return this.sentimentLabels[language][level] || level;
    }
    
    // ──────────────────────────────────────────────────────────
    // 2. رسم مؤشر المشاعر (Sentiment Gauge)
    // ──────────────────────────────────────────────────────────
    
    async createSentimentGauge(elementId, sentimentScore, options = {}) {
        if (!this._isReady()) return null;
        
        const element = this._getElement(elementId);
        if (!element) return null;
        
        const validScore = Math.max(-1, Math.min(1, sentimentScore || 0));
        const percentage = ((validScore + 1) / 2) * 100;
        
        let color = this.colors.very_negative;
        if (validScore >= 0.7) color = this.colors.very_positive;
        else if (validScore >= 0.2) color = this.colors.positive;
        else if (validScore >= -0.2) color = this.colors.neutral;
        else if (validScore >= -0.7) color = this.colors.negative;
        
        const theme = {
            background: this.colors.background,
            textColor: this.colors.text,
            gridColor: this.darkMode ? '#3c4043' : '#e8eaed'
        };
        
        const defaultOptions = {
            chart: {
                type: 'radialBar',
                height: options.height || 300,
                background: theme.background,
                toolbar: { show: false }
            },
            plotOptions: {
                radialBar: {
                    startAngle: -135,
                    endAngle: 135,
                    hollow: {
                        margin: 0,
                        size: '65%',
                        background: 'transparent'
                    },
                    track: {
                        background: theme.gridColor,
                        strokeWidth: '67%',
                        margin: 0
                    },
                    dataLabels: {
                        name: {
                            show: true,
                            fontSize: '14px',
                            fontWeight: '500',
                            color: theme.textColor,
                            offsetY: -10
                        },
                        value: {
                            show: true,
                            fontSize: '28px',
                            fontWeight: 'bold',
                            color: color,
                            formatter: (val) => this._formatNumber(validScore, 2),
                            offsetY: 10
                        }
                    }
                }
            },
            colors: [color],
            stroke: { lineCap: 'round' },
            labels: [options.label || (this.isRTL ? 'درجة المشاعر' : 'Sentiment Score')],
            title: {
                text: options.title || '',
                align: 'center',
                style: {
                    fontSize: '16px',
                    fontWeight: 'bold',
                    color: theme.textColor
                }
            },
            tooltip: {
                theme: this.darkMode ? 'dark' : 'light',
                y: {
                    formatter: (val) => this._formatNumber(val, 2)
                }
            }
        };
        
        const finalOptions = { ...defaultOptions, ...options };
        finalOptions.series = [percentage];
        
        try {
            if (this.charts[elementId]) {
                this.destroyChart(elementId);
            }
            
            this.charts[elementId] = new ApexCharts(element, finalOptions);
            await this.charts[elementId].render();
            return this.charts[elementId];
        } catch (error) {
            console.error('[SentimentDisplay] Failed to create sentiment gauge:', error);
            return null;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 3. عرض تنبيه الطوارئ (Emergency Alert)
    // ──────────────────────────────────────────────────────────
    
    showEmergencyAlert(containerId, alertData, options = {}) {
        const container = this._getElement(containerId);
        if (!container) return;
        
        const { level, keywords, sentimentScore, recommendations } = alertData;
        
        const alertLevel = level === 'critical' ? 'critical' : 'high';
        const alertTitle = level === 'critical' 
            ? (this.isRTL ? '🚨 حالة حرجة - تدخل فوري مطلوب' : '🚨 Critical - Immediate Action Required')
            : (this.isRTL ? '⚠️ حالة طارئة - متابعة عاجلة' : '⚠️ Emergency - Urgent Attention');
        
        // تشغيل الصوت فقط إذا كان المستخدم تفاعل والصوت مفعل
        if (level === 'critical' && this.userInteracted && this.soundEnabled) {
            this.playEmergencySound();
        } else if (level === 'critical' && !this.userInteracted) {
            this.showSoundDisabledWarning(containerId);
        }
        
        const alertHtml = `
            <div class="sentiment-emergency-alert alert-${alertLevel}">
                <div class="alert-header">
                    <span class="alert-icon">${level === 'critical' ? '🔴' : '🟠'}</span>
                    <span class="alert-title">${alertTitle}</span>
                    <button class="alert-close" onclick="this.closest('.sentiment-emergency-alert').remove()">✕</button>
                </div>
                <div class="alert-body">
                    <div class="alert-keywords">
                        <strong>${this.isRTL ? 'كلمات مفتاحية:' : 'Keywords:'}</strong>
                        <div class="keyword-list">
                            ${keywords.map(k => `<span class="keyword emergency-keyword">${this.escapeHtml(k)}</span>`).join('')}
                        </div>
                    </div>
                    <div class="alert-score">
                        <strong>${this.isRTL ? 'درجة المشاعر:' : 'Sentiment Score:'}</strong>
                        <span class="score-value" style="color: ${this.getSentimentColor('very_negative')}">${this._formatNumber(sentimentScore, 2)}</span>
                    </div>
                    ${recommendations && recommendations.length > 0 ? `
                        <div class="alert-recommendations">
                            <strong>${this.isRTL ? 'التوصيات:' : 'Recommendations:'}</strong>
                            <ul class="recommendations-list">
                                ${recommendations.map(r => `<li>${this.escapeHtml(r)}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                    <div class="alert-actions">
                        <button class="alert-btn alert-btn-primary" onclick="window.location.href='/triage'">
                            ${this.isRTL ? '🚑 الانتقال للفرز الطبي' : '🚑 Go to Triage'}
                        </button>
                        <button class="alert-btn alert-btn-secondary" onclick="this.closest('.sentiment-emergency-alert').remove()">
                            ${this.isRTL ? 'تأكيد' : 'Dismiss'}
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        container.innerHTML = alertHtml;
        
        container.scrollIntoView({ behavior: 'smooth', block: 'start' });
        container.classList.add('alert-animate');
        setTimeout(() => container.classList.remove('alert-animate'), 500);
        
        window.dispatchEvent(new CustomEvent('riva-emergency-alert', {
            detail: { level, keywords, sentimentScore }
        }));
    }
    
    // ──────────────────────────────────────────────────────────
    // 4. عرض نتائج التحليل (Analysis Results)
    // ──────────────────────────────────────────────────────────
    
    displayResults(containerId, results, options = {}) {
        const container = this._getElement(containerId);
        if (!container) return;
        
        const sentimentLabel = this.getSentimentLabel(results.sentiment, options.language || 'ar');
        const sentimentColor = this.getSentimentColor(results.sentiment);
        
        const html = `
            <div class="sentiment-results-card">
                <div class="results-header">
                    <h3>${this.isRTL ? '📊 نتائج تحليل المشاعر' : '📊 Sentiment Analysis Results'}</h3>
                    ${options.showTimestamp ? `<span class="timestamp">${new Date().toLocaleString()}</span>` : ''}
                </div>
                
                <div class="results-main">
                    <div class="sentiment-badge" style="background: ${sentimentColor}20; border-right-color: ${sentimentColor}">
                        <span class="sentiment-label" style="color: ${sentimentColor}">${sentimentLabel}</span>
                        <span class="sentiment-score">${this._formatNumber(results.sentimentScore, 2)}</span>
                    </div>
                    
                    ${results.emergencyLevel !== 'none' ? `
                        <div class="emergency-badge emergency-${results.emergencyLevel}">
                            <span class="emergency-icon">${results.emergencyLevel === 'critical' ? '🔴' : '🟠'}</span>
                            <span class="emergency-text">
                                ${this.isRTL 
                                    ? (results.emergencyLevel === 'critical' ? 'حالة حرجة - تدخل فوري' : 'تنبيه طارئ - متابعة عاجلة')
                                    : (results.emergencyLevel === 'critical' ? 'Critical - Immediate Action' : 'Emergency - Urgent')}
                            </span>
                        </div>
                    ` : ''}
                </div>
                
                ${results.emergencyKeywords && results.emergencyKeywords.length > 0 ? `
                    <div class="keywords-section">
                        <h4>${this.isRTL ? '🚨 كلمات طارئة' : '🚨 Emergency Keywords'}</h4>
                        <div class="keywords-list">
                            ${results.emergencyKeywords.map(k => `<span class="keyword emergency-keyword">${this.escapeHtml(k)}</span>`).join('')}
                        </div>
                    </div>
                ` : ''}
                
                ${results.extractedSymptoms && results.extractedSymptoms.length > 0 ? `
                    <div class="symptoms-section">
                        <h4>${this.isRTL ? '🩺 الأعراض المستخرجة' : '🩺 Extracted Symptoms'}</h4>
                        <div class="symptoms-list">
                            ${results.extractedSymptoms.map(s => `<span class="symptom-tag">${this.escapeHtml(s)}</span>`).join('')}
                        </div>
                    </div>
                ` : ''}
                
                ${results.extractedMedications && results.extractedMedications.length > 0 ? `
                    <div class="medications-section">
                        <h4>${this.isRTL ? '💊 الأدوية المستخرجة' : '💊 Extracted Medications'}</h4>
                        <div class="medications-list">
                            ${results.extractedMedications.map(m => `<span class="medication-tag">${this.escapeHtml(m)}</span>`).join('')}
                        </div>
                    </div>
                ` : ''}
                
                ${results.recommendations && results.recommendations.length > 0 ? `
                    <div class="recommendations-section">
                        <h4>${this.isRTL ? '💡 التوصيات' : '💡 Recommendations'}</h4>
                        <ul class="recommendations-list">
                            ${results.recommendations.map(r => `<li>${this.escapeHtml(r)}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
                
                ${results.alerts && results.alerts.length > 0 ? `
                    <div class="alerts-section">
                        <h4>${this.isRTL ? '⚠️ التنبيهات' : '⚠️ Alerts'}</h4>
                        <div class="alerts-list">
                            ${results.alerts.map(a => `<div class="alert-item">${this.escapeHtml(a)}</div>`).join('')}
                        </div>
                    </div>
                ` : ''}
                
                ${options.showConfidence !== false ? `
                    <div class="confidence-section">
                        <div class="confidence-label">${this.isRTL ? 'درجة الثقة' : 'Confidence'}</div>
                        <div class="confidence-bar-container">
                            <div class="confidence-bar" style="width: ${(results.confidence || 0.85) * 100}%"></div>
                        </div>
                        <div class="confidence-value">${this._formatPercentage((results.confidence || 0.85) * 100)}</div>
                    </div>
                ` : ''}
            </div>
        `;
        
        container.innerHTML = html;
        
        container.classList.add('fade-in');
        setTimeout(() => container.classList.remove('fade-in'), 500);
    }
    
    // ──────────────────────────────────────────────────────────
    // 5. عرض الرسوم البيانية للمشاعر (Sentiment Timeline)
    // ──────────────────────────────────────────────────────────
    
    async createSentimentTimeline(elementId, timelineData, options = {}) {
        if (!this._isReady()) return null;
        
        const element = this._getElement(elementId);
        if (!element) return null;
        
        if (!timelineData || !timelineData.series || timelineData.series.length === 0) {
            console.warn('[SentimentDisplay] No timeline data provided');
            return null;
        }
        
        const theme = {
            background: this.colors.background,
            textColor: this.colors.text,
            gridColor: this.darkMode ? '#3c4043' : '#e8eaed'
        };
        
        const defaultOptions = {
            chart: {
                type: 'area',
                height: options.height || 300,
                toolbar: { show: true },
                background: theme.background,
                animations: {
                    enabled: true,
                    easing: 'easeinout',
                    speed: 800
                }
            },
            stroke: {
                curve: 'smooth',
                width: 2
            },
            fill: {
                type: 'gradient',
                gradient: {
                    shadeIntensity: 0.3,
                    opacityFrom: 0.4,
                    opacityTo: 0.1,
                    stops: [0, 100]
                }
            },
            colors: [this.colors.negative, this.colors.positive],
            grid: {
                borderColor: theme.gridColor,
                strokeDashArray: 5
            },
            xaxis: {
                categories: timelineData.categories || [],
                labels: {
                    style: { colors: theme.textColor },
                    rotate: -45
                },
                title: {
                    text: options.xTitle || (this.isRTL ? 'الوقت' : 'Time'),
                    style: { color: theme.textColor }
                }
            },
            yaxis: {
                labels: {
                    style: { colors: theme.textColor },
                    formatter: (val) => this._formatNumber(val, 2)
                },
                title: {
                    text: options.yTitle || (this.isRTL ? 'درجة المشاعر' : 'Sentiment Score'),
                    style: { color: theme.textColor }
                },
                min: -1,
                max: 1
            },
            title: {
                text: options.title || (this.isRTL ? 'تطور المشاعر عبر الزمن' : 'Sentiment Timeline'),
                align: 'center',
                style: {
                    fontSize: '16px',
                    fontWeight: 'bold',
                    color: theme.textColor
                }
            },
            tooltip: {
                theme: this.darkMode ? 'dark' : 'light',
                y: {
                    formatter: (val) => this._formatNumber(val, 2)
                }
            },
            legend: {
                labels: { colors: theme.textColor },
                position: 'top'
            }
        };
        
        const finalOptions = { ...defaultOptions, ...options };
        finalOptions.series = timelineData.series;
        
        try {
            if (this.charts[elementId]) {
                this.destroyChart(elementId);
            }
            
            this.charts[elementId] = new ApexCharts(element, finalOptions);
            await this.charts[elementId].render();
            return this.charts[elementId];
        } catch (error) {
            console.error('[SentimentDisplay] Failed to create sentiment timeline:', error);
            return null;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 6. عرض كلمات مفتاحية عالية التأثير
    // ──────────────────────────────────────────────────────────
    
    displayKeyPhrases(containerId, phrases, options = {}) {
        const container = this._getElement(containerId);
        if (!container) return;
        
        const sortedPhrases = [...phrases].sort((a, b) => b.impact - a.impact);
        
        const html = `
            <div class="key-phrases-card">
                <h4>${this.isRTL ? '🔑 العبارات المفتاحية' : '🔑 Key Phrases'}</h4>
                <div class="phrases-list">
                    ${sortedPhrases.map(p => `
                        <div class="phrase-item" style="--impact: ${Math.min(p.impact * 100, 100)}%">
                            <span class="phrase-text">${this.escapeHtml(p.text)}</span>
                            <span class="phrase-impact" style="color: ${p.sentiment === 'negative' ? this.colors.negative : this.colors.positive}">
                                ${p.sentiment === 'negative' ? '⬇️' : '⬆️'} ${this._formatNumber(p.impact, 2)}
                            </span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        
        container.innerHTML = html;
    }
    
    // ──────────────────────────────────────────────────────────
    // 7. تحديث جميع الرسوم البيانية (Dark Mode)
    // ──────────────────────────────────────────────────────────
    
    updateAllCharts() {
        const theme = {
            background: this.colors.background,
            textColor: this.colors.text,
            gridColor: this.darkMode ? '#3c4043' : '#e8eaed'
        };
        
        Object.keys(this.charts).forEach(elementId => {
            const chart = this.charts[elementId];
            if (chart && chart.updateOptions) {
                try {
                    chart.updateOptions({
                        chart: { background: theme.background },
                        grid: { borderColor: theme.gridColor },
                        tooltip: { theme: this.darkMode ? 'dark' : 'light' },
                        title: { style: { color: theme.textColor } },
                        xaxis: { labels: { style: { colors: theme.textColor } } },
                        yaxis: { labels: { style: { colors: theme.textColor } } },
                        legend: { labels: { colors: theme.textColor } }
                    });
                } catch (error) {
                    console.error(`[SentimentDisplay] Failed to update chart ${elementId}:`, error);
                }
            }
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 8. تحديث مؤشر المشاعر
    // ──────────────────────────────────────────────────────────
    
    updateSentimentGauge(elementId, sentimentScore) {
        const chart = this.charts[elementId];
        if (!chart) return false;
        
        const validScore = Math.max(-1, Math.min(1, sentimentScore || 0));
        const percentage = ((validScore + 1) / 2) * 100;
        
        let color = this.colors.very_negative;
        if (validScore >= 0.7) color = this.colors.very_positive;
        else if (validScore >= 0.2) color = this.colors.positive;
        else if (validScore >= -0.2) color = this.colors.neutral;
        else if (validScore >= -0.7) color = this.colors.negative;
        
        try {
            chart.updateSeries([percentage]);
            chart.updateOptions({
                colors: [color],
                plotOptions: {
                    radialBar: {
                        dataLabels: {
                            value: { 
                                color: color,
                                formatter: () => this._formatNumber(validScore, 2)
                            }
                        }
                    }
                }
            });
            return true;
        } catch (error) {
            console.error('[SentimentDisplay] Failed to update sentiment gauge:', error);
            return false;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 9. حذف الرسم البياني
    // ──────────────────────────────────────────────────────────
    
    destroyChart(elementId) {
        const chart = this.charts[elementId];
        if (chart) {
            try {
                chart.destroy();
            } catch (error) {
                console.warn(`[SentimentDisplay] Error destroying chart ${elementId}:`, error);
            }
            delete this.charts[elementId];
            return true;
        }
        return false;
    }
    
    // ──────────────────────────────────────────────────────────
    // 10. دوال مساعدة
    // ──────────────────────────────────────────────────────────
    
    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    _formatPercentage(value) {
        return `${this._formatNumber(value)}%`;
    }
    
    // ──────────────────────────────────────────────────────────
    // 11. إضافة أنماط CSS (مكتملة مع دعم النصوص الطويلة)
    // ──────────────────────────────────────────────────────────
    
    injectStyles() {
        if (document.getElementById('riva-sentiment-styles')) return;
        
        const isDark = this.darkMode;
        
        const styles = `
            <style id="riva-sentiment-styles">
                /* الأنماط الأساسية مع دعم النصوص الطويلة */
                .sentiment-results-card,
                .key-phrases-card,
                .sentiment-emergency-alert {
                    word-wrap: break-word;
                    word-break: break-word;
                    overflow-wrap: break-word;
                }
                
                .results-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    flex-wrap: wrap;
                    gap: 12px;
                    margin-bottom: 20px;
                    padding-bottom: 12px;
                    border-bottom: 1px solid ${isDark ? '#5f6368' : '#e8eaed'};
                }
                
                .results-header h3 {
                    margin: 0;
                    font-size: 18px;
                    word-break: break-word;
                }
                
                .timestamp {
                    font-size: 12px;
                    color: ${isDark ? '#9aa0a6' : '#5f6368'};
                }
                
                .sentiment-emergency-alert {
                    background: ${isDark ? '#2d2e2e' : '#ffffff'};
                    border-radius: 16px;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                    overflow: hidden;
                    animation: slideDown 0.3s ease;
                }
                
                .alert-critical {
                    border-right: 5px solid #d32f2f;
                }
                
                .alert-high {
                    border-right: 5px solid #ea4335;
                }
                
                .alert-header {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    padding: 16px 20px;
                    background: ${isDark ? '#3c4043' : '#f8f9fa'};
                    border-bottom: 1px solid ${isDark ? '#5f6368' : '#e8eaed'};
                }
                
                .alert-icon {
                    font-size: 24px;
                    flex-shrink: 0;
                }
                
                .alert-title {
                    flex: 1;
                    font-weight: bold;
                    font-size: 16px;
                    color: ${isDark ? '#e8eaed' : '#202124'};
                    word-break: break-word;
                }
                
                .alert-close {
                    background: none;
                    border: none;
                    font-size: 20px;
                    cursor: pointer;
                    color: ${isDark ? '#9aa0a6' : '#5f6368'};
                    flex-shrink: 0;
                }
                
                .alert-body {
                    padding: 20px;
                }
                
                .alert-keywords, .alert-score, .alert-recommendations {
                    margin-bottom: 16px;
                }
                
                .keyword-list {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 8px;
                    margin-top: 8px;
                }
                
                .keyword {
                    background: ${isDark ? '#5f6368' : '#e8eaed'};
                    padding: 4px 12px;
                    border-radius: 20px;
                    font-size: 12px;
                    color: ${isDark ? '#e8eaed' : '#202124'};
                    word-break: break-word;
                }
                
                .emergency-keyword {
                    background: rgba(211, 47, 47, 0.2);
                    color: #d32f2f;
                    font-weight: bold;
                }
                
                .alert-actions {
                    display: flex;
                    gap: 12px;
                    margin-top: 20px;
                    flex-wrap: wrap;
                }
                
                .alert-btn {
                    padding: 8px 16px;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 14px;
                    font-weight: 500;
                    transition: all 0.2s ease;
                    word-break: break-word;
                }
                
                .alert-btn-primary {
                    background: #d32f2f;
                    color: white;
                }
                
                .alert-btn-primary:hover {
                    background: #b71c1c;
                    transform: translateY(-2px);
                }
                
                .alert-btn-secondary {
                    background: ${isDark ? '#5f6368' : '#e8eaed'};
                    color: ${isDark ? '#e8eaed' : '#202124'};
                }
                
                .sentiment-results-card {
                    background: ${isDark ? '#2d2e2e' : '#ffffff'};
                    border-radius: 16px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                }
                
                .results-main {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    flex-wrap: wrap;
                    gap: 16px;
                    margin-bottom: 20px;
                }
                
                .sentiment-badge {
                    display: inline-flex;
                    align-items: center;
                    gap: 12px;
                    padding: 8px 16px;
                    border-radius: 40px;
                    border-right: 3px solid;
                    flex-wrap: wrap;
                }
                
                .sentiment-label {
                    font-weight: bold;
                    font-size: 16px;
                    word-break: break-word;
                }
                
                .sentiment-score {
                    font-size: 20px;
                    font-weight: bold;
                }
                
                .emergency-badge {
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    padding: 8px 16px;
                    border-radius: 40px;
                    font-weight: bold;
                    flex-wrap: wrap;
                }
                
                .emergency-critical {
                    background: rgba(211, 47, 47, 0.2);
                    color: #d32f2f;
                }
                
                .emergency-high {
                    background: rgba(234, 67, 53, 0.2);
                    color: #ea4335;
                }
                
                .symptoms-section, .medications-section, .keywords-section {
                    margin-bottom: 20px;
                }
                
                .symptoms-section h4, .medications-section h4, .keywords-section h4 {
                    font-size: 14px;
                    margin-bottom: 10px;
                    color: ${isDark ? '#9aa0a6' : '#5f6368'};
                }
                
                .symptoms-list, .medications-list, .keywords-list {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 8px;
                }
                
                .symptom-tag, .medication-tag {
                    background: ${isDark ? '#3c4043' : '#e8eaed'};
                    padding: 6px 12px;
                    border-radius: 20px;
                    font-size: 13px;
                    word-break: break-word;
                }
                
                .recommendations-section {
                    background: rgba(26, 115, 232, 0.1);
                    border-radius: 12px;
                    padding: 16px;
                    margin: 16px 0;
                }
                
                .recommendations-list {
                    margin: 8px 0 0;
                    padding-right: 20px;
                }
                
                .recommendations-list li {
                    margin-bottom: 6px;
                    word-break: break-word;
                }
                
                .confidence-section {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    margin-top: 16px;
                    padding-top: 16px;
                    border-top: 1px solid ${isDark ? '#5f6368' : '#e8eaed'};
                    flex-wrap: wrap;
                }
                
                .confidence-bar-container {
                    flex: 1;
                    height: 8px;
                    background: ${isDark ? '#3c4043' : '#e8eaed'};
                    border-radius: 4px;
                    overflow: hidden;
                    min-width: 100px;
                }
                
                .confidence-bar {
                    height: 100%;
                    background: linear-gradient(90deg, #34a853, #1a73e8);
                    border-radius: 4px;
                    transition: width 0.5s ease;
                }
                
                .confidence-value {
                    font-size: 12px;
                    font-weight: bold;
                    min-width: 45px;
                }
                
                .key-phrases-card {
                    background: ${isDark ? '#2d2e2e' : '#ffffff'};
                    border-radius: 16px;
                    padding: 16px;
                    margin-top: 20px;
                }
                
                .key-phrases-card h4 {
                    margin-bottom: 16px;
                    word-break: break-word;
                }
                
                .phrase-item {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    gap: 12px;
                    padding: 12px;
                    margin-bottom: 8px;
                    background: ${isDark ? '#3c4043' : '#f8f9fa'};
                    border-radius: 8px;
                    transition: all 0.2s ease;
                    flex-wrap: wrap;
                }
                
                .phrase-item:hover {
                    transform: translateX(-4px);
                }
                
                .phrase-text {
                    font-weight: 500;
                    word-break: break-word;
                    flex: 1;
                }
                
                .phrase-impact {
                    font-size: 12px;
                    font-weight: bold;
                    flex-shrink: 0;
                }
                
                .sound-enabled-toast {
                    position: fixed;
                    bottom: 20px;
                    left: 20px;
                    background: ${isDark ? '#2d2e2e' : '#ffffff'};
                    border-radius: 8px;
                    padding: 12px 16px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                    z-index: 10001;
                    animation: slideInLeft 0.3s ease;
                    border-right: 3px solid #34a853;
                }
                
                .sound-warning {
                    background: rgba(251, 188, 4, 0.2);
                    border-radius: 8px;
                    padding: 8px 12px;
                    margin-bottom: 12px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    font-size: 12px;
                    color: ${isDark ? '#e8eaed' : '#202124'};
                    flex-wrap: wrap;
                }
                
                @keyframes slideDown {
                    from { opacity: 0; transform: translateY(-20px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                
                @keyframes slideInLeft {
                    from { opacity: 0; transform: translateX(-100%); }
                    to { opacity: 1; transform: translateX(0); }
                }
                
                .alert-animate {
                    animation: pulse 0.5s ease;
                }
                
                @keyframes pulse {
                    0%, 100% { box-shadow: 0 0 0 0 rgba(211, 47, 47, 0.4); }
                    50% { box-shadow: 0 0 0 8px rgba(211, 47, 47, 0); }
                }
                
                .fade-in {
                    animation: fadeIn 0.3s ease;
                }
                
                .fade-out {
                    animation: fadeOut 0.3s ease forwards;
                }
                
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                
                @keyframes fadeOut {
                    to { opacity: 0; transform: translateX(-100%); }
                }
                
                @media (max-width: 768px) {
                    .results-main {
                        flex-direction: column;
                        align-items: flex-start;
                    }
                    
                    .alert-actions {
                        flex-direction: column;
                    }
                    
                    .alert-btn {
                        width: 100%;
                        text-align: center;
                    }
                    
                    .confidence-section {
                        flex-direction: column;
                        align-items: flex-start;
                    }
                    
                    .confidence-bar-container {
                        width: 100%;
                    }
                    
                    .phrase-item {
                        flex-direction: column;
                        align-items: flex-start;
                    }
                    
                    .sound-enabled-toast {
                        left: 16px;
                        right: 16px;
                        bottom: 16px;
                    }
                }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }
}

// ──────────────────────────────────────────────────────────
// 12. تصدير نسخة واحدة (Singleton)
// ──────────────────────────────────────────────────────────

// إنشاء نسخة واحدة
const sentimentDisplay = new SentimentDisplay();

// تخزين في window للاستخدام العادي
window.sentimentDisplay = sentimentDisplay;
window.rivaSentimentDisplay = sentimentDisplay;

// ES Module export
export default sentimentDisplay;
export { sentimentDisplay };
