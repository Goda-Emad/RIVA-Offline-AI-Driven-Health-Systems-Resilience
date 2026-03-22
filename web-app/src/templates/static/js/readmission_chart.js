/**
 * readmission_chart.js
 * ====================
 * RIVA Health Platform - Readmission Risk Chart Module
 * وحدة الرسوم البيانية لخطر إعادة الدخول للمستشفى
 * 
 * المسؤوليات:
 * - رسم مؤشر خطر إعادة الدخول (Gauge Chart)
 * - عرض عوامل الخطر المؤثرة (SHAP values)
 * - رسم اتجاهات إعادة الدخول عبر الزمن
 * - عرض مقارنة بين المرضى
 * - تكامل مع صفحة 13_readmission.html
 * 
 * المسار: web-app/src/static/js/readmission_chart.js
 * 
 * التحسينات:
 * - دعم الوضع الليلي (Dark Mode)
 * - دعم RTL (Right-to-Left) للغة العربية
 * - تحديث ديناميكي للبيانات
 * - عرض هامش الثقة
 * - تصدير التقارير كـ PDF
 * - معالجة الأسماء الطويلة (Truncate)
 * 
 * الإصدار: 4.2.1
 */

// ──────────────────────────────────────────────────────────
// 0. التحقق من تحميل ApexCharts والمكتبات المساعدة
// ──────────────────────────────────────────────────────────

let apexChartsLoaded = false;
let html2canvasLoaded = false;
let jspdfLoaded = false;
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
                console.error('[ReadmissionChart] ApexCharts failed to load');
                resolve();
            }
        }, 10000);
    });
}

function loadHtml2Canvas() {
    return new Promise((resolve) => {
        if (typeof html2canvas !== 'undefined') {
            html2canvasLoaded = true;
            resolve();
            return;
        }
        
        const script = document.createElement('script');
        script.src = 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js';
        script.onload = () => {
            html2canvasLoaded = true;
            resolve();
        };
        script.onerror = () => {
            console.warn('[ReadmissionChart] html2canvas failed to load');
            resolve();
        };
        document.head.appendChild(script);
    });
}

function loadJSPDF() {
    return new Promise((resolve) => {
        if (typeof jspdf !== 'undefined') {
            jspdfLoaded = true;
            resolve();
            return;
        }
        
        const script = document.createElement('script');
        script.src = 'https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js';
        script.onload = () => {
            jspdfLoaded = true;
            resolve();
        };
        script.onerror = () => {
            console.warn('[ReadmissionChart] jspdf failed to load');
            resolve();
        };
        document.head.appendChild(script);
    });
}

// ──────────────────────────────────────────────────────────
// 1. كلاس Readmission Chart
// ──────────────────────────────────────────────────────────

class ReadmissionChart {
    constructor() {
        this.charts = {};
        this.initialized = false;
        this.isRTL = document.documentElement.dir === 'rtl' || document.body.dir === 'rtl';
        this.colors = {
            primary: '#1a73e8',
            secondary: '#34a853',
            warning: '#fbbc04',
            danger: '#ea4335',
            info: '#4285f4',
            dark: '#202124',
            gray: '#5f6368',
            light: '#f8f9fa',
            critical: '#c5221f'
        };
        
        this.darkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
        
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            this.darkMode = e.matches;
            this.updateAllCharts();
        });
        
        this.init();
    }
    
    async init() {
        console.log('[ReadmissionChart] Initializing...');
        await Promise.all([
            waitForApexCharts(),
            loadHtml2Canvas(),
            loadJSPDF()
        ]);
        this.initialized = true;
        this.injectStyles();
        console.log('[ReadmissionChart] Initialized', { isRTL: this.isRTL });
    }
    
    _getElement(elementId) {
        const element = document.getElementById(elementId);
        if (!element) {
            console.warn(`[ReadmissionChart] Element "${elementId}" not found`);
            return null;
        }
        return element;
    }
    
    _isReady() {
        if (!this.initialized) {
            console.warn('[ReadmissionChart] Not initialized yet');
            return false;
        }
        if (typeof ApexCharts === 'undefined') {
            console.error('[ReadmissionChart] ApexCharts not loaded');
            return false;
        }
        return true;
    }
    
    _truncateText(text, maxLength = 20) {
        if (!text) return '';
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength - 3) + '...';
    }
    
    _formatNumber(value, decimals = 1) {
        if (this.isRTL) {
            // تحويل الأرقام للعربية مع المحافظة على النقطة العشرية
            const arabicNumbers = {
                '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤',
                '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩'
            };
            const formatted = value.toFixed(decimals);
            return formatted.replace(/[0-9]/g, d => arabicNumbers[d]);
        }
        return value.toFixed(decimals);
    }
    
    _formatPercentage(value) {
        return `${this._formatNumber(value)}%`;
    }
    
    getThemeConfig() {
        if (this.darkMode) {
            return {
                background: '#1e1e1e',
                textColor: '#e8eaed',
                gridColor: '#3c4043',
                tooltipTheme: 'dark'
            };
        }
        return {
            background: '#ffffff',
            textColor: '#202124',
            gridColor: '#e8eaed',
            tooltipTheme: 'light'
        };
    }
    
    getLocaleConfig() {
        if (this.isRTL) {
            return {
                name: 'ar',
                options: {
                    months: ['يناير', 'فبراير', 'مارس', 'إبريل', 'مايو', 'يونيو', 'يوليو', 'أغسطس', 'سبتمبر', 'أكتوبر', 'نوفمبر', 'ديسمبر'],
                    shortMonths: ['يناير', 'فبراير', 'مارس', 'إبريل', 'مايو', 'يونيو', 'يوليو', 'أغسطس', 'سبتمبر', 'أكتوبر', 'نوفمبر', 'ديسمبر'],
                    days: ['الأحد', 'الإثنين', 'الثلاثاء', 'الأربعاء', 'الخميس', 'الجمعة', 'السبت'],
                    shortDays: ['أحد', 'إثنين', 'ثلاثاء', 'أربعاء', 'خميس', 'جمعة', 'سبت'],
                    toolbar: {
                        exportToSVG: 'تصدير SVG',
                        exportToPNG: 'تصدير PNG',
                        exportToCSV: 'تصدير CSV',
                        menu: 'القائمة',
                        selection: 'تحديد',
                        selectionZoom: 'تكبير التحديد',
                        zoomIn: 'تكبير',
                        zoomOut: 'تصغير',
                        pan: 'تحريك',
                        reset: 'إعادة تعيين'
                    }
                }
            };
        }
        return {};
    }
    
    // ──────────────────────────────────────────────────────────
    // 2. رسم مؤشر خطر إعادة الدخول (Risk Gauge)
    // ──────────────────────────────────────────────────────────
    
    async createRiskGauge(elementId, probability, options = {}) {
        if (!this._isReady()) return null;
        
        const element = this._getElement(elementId);
        if (!element) return null;
        
        const validProb = Math.max(0, Math.min(100, (probability || 0) * 100));
        const theme = this.getThemeConfig();
        
        let color = this.colors.secondary;
        if (validProb > 70) color = this.colors.critical;
        else if (validProb > 40) color = this.colors.warning;
        
        const defaultOptions = {
            chart: {
                type: 'radialBar',
                height: options.height || 350,
                background: theme.background,
                toolbar: { show: false },
                locales: [this.getLocaleConfig()],
                defaultLocale: this.isRTL ? 'ar' : 'en'
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
                            fontSize: '32px',
                            fontWeight: 'bold',
                            color: color,
                            formatter: (val) => this._formatPercentage(Math.round(val)),
                            offsetY: 10
                        }
                    }
                }
            },
            colors: [color],
            stroke: { lineCap: 'round' },
            labels: [options.label || (this.isRTL ? 'خطر إعادة الدخول' : 'Readmission Risk')],
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
                theme: theme.tooltipTheme,
                y: {
                    formatter: (val) => this._formatPercentage(Math.round(val))
                }
            },
            subtitle: {
                text: options.subtitle || (this.isRTL ? `الثقة: ${this._formatPercentage((options.confidence || 0.85) * 100)}` : `Confidence: ${this._formatPercentage((options.confidence || 0.85) * 100)}`),
                align: 'center',
                style: {
                    fontSize: '12px',
                    color: theme.textColor
                }
            }
        };
        
        const finalOptions = { ...defaultOptions, ...options };
        finalOptions.series = [validProb];
        
        try {
            if (this.charts[elementId]) {
                this.destroyChart(elementId);
            }
            
            this.charts[elementId] = new ApexCharts(element, finalOptions);
            await this.charts[elementId].render();
            return this.charts[elementId];
        } catch (error) {
            console.error('[ReadmissionChart] Failed to create risk gauge:', error);
            return null;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 3. رسم عوامل الخطر (SHAP Values - Horizontal Bar)
    // ──────────────────────────────────────────────────────────
    
    async createRiskFactorsChart(elementId, riskFactors, options = {}) {
        if (!this._isReady()) return null;
        
        const element = this._getElement(elementId);
        if (!element) return null;
        
        if (!riskFactors || riskFactors.length === 0) {
            console.warn('[ReadmissionChart] No risk factors provided');
            return null;
        }
        
        const theme = this.getThemeConfig();
        
        const sortedFactors = [...riskFactors].sort((a, b) => Math.abs(b.shapValue) - Math.abs(a.shapValue));
        const topFactors = sortedFactors.slice(0, options.topK || 8);
        
        const categories = topFactors.map(f => this._truncateText(f.nameAr || f.name, 25));
        const seriesData = topFactors.map(f => Math.abs(f.shapValue) * 100);
        
        const defaultOptions = {
            chart: {
                type: 'bar',
                height: options.height || 400,
                toolbar: { show: true },
                background: theme.background,
                locales: [this.getLocaleConfig()],
                defaultLocale: this.isRTL ? 'ar' : 'en'
            },
            plotOptions: {
                bar: {
                    horizontal: true,
                    barHeight: '70%',
                    borderRadius: 8,
                    dataLabels: { position: 'inside' }
                }
            },
            colors: [this.colors.primary],
            grid: {
                borderColor: theme.gridColor,
                strokeDashArray: 5,
                xaxis: { lines: { show: true } }
            },
            xaxis: {
                categories: categories,
                labels: {
                    style: { colors: theme.textColor, fontSize: '13px' },
                    trim: true,
                    maxHeight: 100
                },
                title: {
                    text: options.xTitle || (this.isRTL ? 'نسبة التأثير (%)' : 'Impact (%)'),
                    style: { color: theme.textColor }
                }
            },
            yaxis: {
                labels: {
                    style: { colors: theme.textColor, fontSize: '12px' }
                },
                title: {
                    text: options.yTitle || (this.isRTL ? 'عوامل الخطر' : 'Risk Factors'),
                    style: { color: theme.textColor }
                }
            },
            title: {
                text: options.title || (this.isRTL ? 'أهم عوامل الخطر المؤثرة' : 'Key Risk Factors'),
                align: 'center',
                style: {
                    fontSize: '16px',
                    fontWeight: 'bold',
                    color: theme.textColor
                }
            },
            tooltip: {
                theme: theme.tooltipTheme,
                y: {
                    formatter: (val) => this._formatPercentage(val)
                }
            },
            dataLabels: {
                enabled: true,
                formatter: (val) => this._formatPercentage(val),
                style: {
                    fontSize: '11px',
                    colors: [theme.textColor]
                }
            }
        };
        
        const finalOptions = { ...defaultOptions, ...options };
        finalOptions.series = [{
            name: this.isRTL ? 'التأثير' : 'Impact',
            data: seriesData
        }];
        
        try {
            if (this.charts[elementId]) {
                this.destroyChart(elementId);
            }
            
            this.charts[elementId] = new ApexCharts(element, finalOptions);
            await this.charts[elementId].render();
            return this.charts[elementId];
        } catch (error) {
            console.error('[ReadmissionChart] Failed to create risk factors chart:', error);
            return null;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 4. رسم اتجاهات إعادة الدخول (Line Chart)
    // ──────────────────────────────────────────────────────────
    
    async createTrendChart(elementId, data, options = {}) {
        if (!this._isReady()) return null;
        
        const element = this._getElement(elementId);
        if (!element) return null;
        
        if (!data || !data.series || data.series.length === 0) {
            console.warn('[ReadmissionChart] No trend data provided');
            return null;
        }
        
        const theme = this.getThemeConfig();
        
        const truncatedCategories = (data.categories || []).map(c => this._truncateText(c, 15));
        
        const defaultOptions = {
            chart: {
                type: 'line',
                height: options.height || 350,
                toolbar: { show: true },
                background: theme.background,
                animations: {
                    enabled: true,
                    easing: 'easeinout',
                    speed: 800
                },
                locales: [this.getLocaleConfig()],
                defaultLocale: this.isRTL ? 'ar' : 'en'
            },
            stroke: {
                curve: 'smooth',
                width: 3
            },
            colors: [this.colors.danger, this.colors.warning, this.colors.primary],
            grid: {
                borderColor: theme.gridColor,
                strokeDashArray: 5,
                position: 'back'
            },
            xaxis: {
                categories: truncatedCategories,
                labels: {
                    style: { colors: theme.textColor, fontSize: '12px' },
                    rotate: -45,
                    rotateAlways: false,
                    trim: true,
                    maxHeight: 80
                },
                title: {
                    text: options.xTitle || (this.isRTL ? 'التاريخ' : 'Date'),
                    style: { color: theme.textColor }
                }
            },
            yaxis: {
                labels: {
                    style: { colors: theme.textColor },
                    formatter: (val) => this._formatPercentage(val)
                },
                title: {
                    text: options.yTitle || (this.isRTL ? 'نسبة إعادة الدخول (%)' : 'Readmission Rate (%)'),
                    style: { color: theme.textColor }
                },
                min: 0,
                max: 100
            },
            title: {
                text: options.title || (this.isRTL ? 'اتجاهات إعادة الدخول' : 'Readmission Trends'),
                align: 'center',
                style: {
                    fontSize: '16px',
                    fontWeight: 'bold',
                    color: theme.textColor
                }
            },
            tooltip: {
                theme: theme.tooltipTheme,
                shared: true,
                intersect: false,
                y: {
                    formatter: (val) => this._formatPercentage(val)
                }
            },
            legend: {
                labels: { colors: theme.textColor },
                position: 'top'
            },
            markers: {
                size: 4,
                hover: { size: 6 }
            }
        };
        
        const finalOptions = { ...defaultOptions, ...options };
        finalOptions.series = data.series.map(s => ({
            ...s,
            data: s.data.map(d => d * 100)
        }));
        
        try {
            if (this.charts[elementId]) {
                this.destroyChart(elementId);
            }
            
            this.charts[elementId] = new ApexCharts(element, finalOptions);
            await this.charts[elementId].render();
            return this.charts[elementId];
        } catch (error) {
            console.error('[ReadmissionChart] Failed to create trend chart:', error);
            return null;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 5. رسم مقارنة بين المرضى (Grouped Bar)
    // ──────────────────────────────────────────────────────────
    
    async createComparisonChart(elementId, patients, options = {}) {
        if (!this._isReady()) return null;
        
        const element = this._getElement(elementId);
        if (!element) return null;
        
        if (!patients || patients.length === 0) {
            console.warn('[ReadmissionChart] No patients provided');
            return null;
        }
        
        const theme = this.getThemeConfig();
        
        const categories = patients.map(p => this._truncateText(p.name || p.id, 20));
        const seriesData = patients.map(p => (p.probability || 0) * 100);
        
        const defaultOptions = {
            chart: {
                type: 'bar',
                height: options.height || 400,
                toolbar: { show: true },
                background: theme.background,
                locales: [this.getLocaleConfig()],
                defaultLocale: this.isRTL ? 'ar' : 'en'
            },
            plotOptions: {
                bar: {
                    horizontal: false,
                    columnWidth: '55%',
                    borderRadius: 8,
                    dataLabels: { position: 'top' }
                }
            },
            colors: [this.colors.danger],
            grid: {
                borderColor: theme.gridColor,
                strokeDashArray: 5
            },
            xaxis: {
                categories: categories,
                labels: {
                    style: { colors: theme.textColor, fontSize: '12px' },
                    rotate: -45,
                    rotateAlways: false,
                    trim: true,
                    maxHeight: 100
                },
                title: {
                    text: options.xTitle || (this.isRTL ? 'المريض' : 'Patient'),
                    style: { color: theme.textColor }
                }
            },
            yaxis: {
                labels: {
                    style: { colors: theme.textColor },
                    formatter: (val) => this._formatPercentage(val)
                },
                title: {
                    text: options.yTitle || (this.isRTL ? 'خطر إعادة الدخول (%)' : 'Readmission Risk (%)'),
                    style: { color: theme.textColor }
                },
                min: 0,
                max: 100
            },
            title: {
                text: options.title || (this.isRTL ? 'مقارنة خطر إعادة الدخول بين المرضى' : 'Patient Readmission Risk Comparison'),
                align: 'center',
                style: {
                    fontSize: '16px',
                    fontWeight: 'bold',
                    color: theme.textColor
                }
            },
            tooltip: {
                theme: theme.tooltipTheme,
                y: {
                    formatter: (val) => this._formatPercentage(val)
                }
            },
            dataLabels: {
                enabled: true,
                offsetY: -20,
                formatter: (val) => this._formatPercentage(val),
                style: {
                    fontSize: '11px',
                    colors: [theme.textColor]
                }
            }
        };
        
        const finalOptions = { ...defaultOptions, ...options };
        finalOptions.series = [{
            name: this.isRTL ? 'خطر إعادة الدخول' : 'Readmission Risk',
            data: seriesData
        }];
        
        try {
            if (this.charts[elementId]) {
                this.destroyChart(elementId);
            }
            
            this.charts[elementId] = new ApexCharts(element, finalOptions);
            await this.charts[elementId].render();
            return this.charts[elementId];
        } catch (error) {
            console.error('[ReadmissionChart] Failed to create comparison chart:', error);
            return null;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 6. تحديث المؤشرات
    // ──────────────────────────────────────────────────────────
    
    updateRiskGauge(elementId, probability, confidence = 0.85) {
        const chart = this.charts[elementId];
        if (!chart) return false;
        
        const validProb = Math.max(0, Math.min(100, (probability || 0) * 100));
        
        let color = this.colors.secondary;
        if (validProb > 70) color = this.colors.critical;
        else if (validProb > 40) color = this.colors.warning;
        
        try {
            chart.updateSeries([validProb]);
            chart.updateOptions({
                colors: [color],
                plotOptions: {
                    radialBar: {
                        dataLabels: {
                            value: { color: color }
                        }
                    }
                },
                subtitle: {
                    text: this.isRTL ? `الثقة: ${this._formatPercentage(confidence * 100)}` : `Confidence: ${this._formatPercentage(confidence * 100)}`
                }
            });
            return true;
        } catch (error) {
            console.error('[ReadmissionChart] Failed to update risk gauge:', error);
            return false;
        }
    }
    
    updateRiskFactorsChart(elementId, riskFactors, options = {}) {
        const chart = this.charts[elementId];
        if (!chart) return false;
        
        if (!riskFactors || riskFactors.length === 0) return false;
        
        const sortedFactors = [...riskFactors].sort((a, b) => Math.abs(b.shapValue) - Math.abs(a.shapValue));
        const topFactors = sortedFactors.slice(0, options.topK || 8);
        
        const categories = topFactors.map(f => this._truncateText(f.nameAr || f.name, 25));
        const seriesData = topFactors.map(f => Math.abs(f.shapValue) * 100);
        
        try {
            chart.updateSeries([{ data: seriesData }]);
            chart.updateOptions({
                xaxis: { categories: categories }
            });
            return true;
        } catch (error) {
            console.error('[ReadmissionChart] Failed to update risk factors:', error);
            return false;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 7. تصدير الرسم كصورة
    // ──────────────────────────────────────────────────────────
    
    async exportChart(elementId, format = 'png') {
        const chart = this.charts[elementId];
        if (!chart) {
            console.warn(`[ReadmissionChart] Chart ${elementId} not found`);
            return null;
        }
        
        try {
            const dataUrl = await chart.dataURI();
            const link = document.createElement('a');
            link.download = `readmission_chart_${elementId}_${Date.now()}.${format}`;
            link.href = dataUrl;
            link.click();
            return dataUrl;
        } catch (error) {
            console.error('[ReadmissionChart] Export failed:', error);
            return null;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 8. تصدير جميع الرسوم كملف PDF
    // ──────────────────────────────────────────────────────────
    
    async exportAsPDF(elementIds, filename = 'readmission_report.pdf', options = {}) {
        if (!html2canvasLoaded || !jspdfLoaded) {
            console.warn('[ReadmissionChart] PDF export libraries not loaded');
            await Promise.all([loadHtml2Canvas(), loadJSPDF()]);
        }
        
        if (typeof html2canvas === 'undefined' || typeof jspdf === 'undefined') {
            console.error('[ReadmissionChart] PDF export libraries unavailable');
            return false;
        }
        
        const { jsPDF } = jspdf;
        const pdf = new jsPDF({
            orientation: 'portrait',
            unit: 'mm',
            format: 'a4'
        });
        
        let yOffset = 20;
        const pageWidth = 210;
        const pageHeight = 297;
        const margin = 20;
        const contentWidth = pageWidth - (margin * 2);
        
        for (let i = 0; i < elementIds.length; i++) {
            const elementId = elementIds[i];
            const element = document.getElementById(elementId);
            
            if (!element) {
                console.warn(`[ReadmissionChart] Element ${elementId} not found for PDF export`);
                continue;
            }
            
            try {
                const canvas = await html2canvas(element, {
                    scale: 2,
                    backgroundColor: this.darkMode ? '#1e1e1e' : '#ffffff',
                    logging: false
                });
                
                const imgData = canvas.toDataURL('image/png');
                const imgWidth = contentWidth;
                const imgHeight = (canvas.height * imgWidth) / canvas.width;
                
                if (yOffset + imgHeight > pageHeight - margin) {
                    pdf.addPage();
                    yOffset = margin;
                }
                
                pdf.addImage(imgData, 'PNG', margin, yOffset, imgWidth, imgHeight);
                yOffset += imgHeight + 20;
                
            } catch (error) {
                console.error(`[ReadmissionChart] Failed to render chart ${elementId}:`, error);
            }
        }
        
        pdf.save(filename);
        return true;
    }
    
    // ──────────────────────────────────────────────────────────
    // 9. تحديث جميع الرسوم البيانية (Dark Mode)
    // ──────────────────────────────────────────────────────────
    
    updateAllCharts() {
        const theme = this.getThemeConfig();
        
        Object.keys(this.charts).forEach(elementId => {
            const chart = this.charts[elementId];
            if (chart && chart.updateOptions) {
                try {
                    chart.updateOptions({
                        chart: { background: theme.background },
                        grid: { borderColor: theme.gridColor },
                        tooltip: { theme: theme.tooltipTheme },
                        title: { style: { color: theme.textColor } },
                        xaxis: { labels: { style: { colors: theme.textColor } } },
                        yaxis: { labels: { style: { colors: theme.textColor } } },
                        legend: { labels: { colors: theme.textColor } }
                    });
                } catch (error) {
                    console.error(`[ReadmissionChart] Failed to update chart ${elementId}:`, error);
                }
            }
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 10. حذف الرسم البياني
    // ──────────────────────────────────────────────────────────
    
    destroyChart(elementId) {
        const chart = this.charts[elementId];
        if (chart) {
            try {
                chart.destroy();
            } catch (error) {
                console.warn(`[ReadmissionChart] Error destroying chart ${elementId}:`, error);
            }
            delete this.charts[elementId];
            return true;
        }
        return false;
    }
    
    // ──────────────────────────────────────────────────────────
    // 11. الحصول على حالة الرسم البياني
    // ──────────────────────────────────────────────────────────
    
    getChartStatus(elementId) {
        return {
            exists: !!this.charts[elementId],
            initialized: this.initialized,
            apexChartsLoaded: typeof ApexCharts !== 'undefined'
        };
    }
    
    // ──────────────────────────────────────────────────────────
    // 12. إضافة أنماط CSS
    // ──────────────────────────────────────────────────────────
    
    injectStyles() {
        if (document.getElementById('riva-readmission-styles')) return;
        
        const isDark = this.darkMode;
        
        const styles = `
            <style id="riva-readmission-styles">
                .readmission-container {
                    background: ${isDark ? '#2d2e2e' : '#ffffff'};
                    border-radius: 16px;
                    padding: 20px;
                    margin: 16px 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                }
                
                .readmission-title {
                    font-size: 18px;
                    font-weight: 600;
                    margin-bottom: 16px;
                    text-align: center;
                    color: ${isDark ? '#e8eaed' : '#202124'};
                }
                
                .risk-level-badge {
                    display: inline-block;
                    padding: 4px 12px;
                    border-radius: 20px;
                    font-size: 12px;
                    font-weight: 500;
                }
                
                .risk-low {
                    background: rgba(52, 168, 83, 0.1);
                    color: #34a853;
                }
                
                .risk-medium {
                    background: rgba(251, 188, 4, 0.1);
                    color: #fbbc04;
                }
                
                .risk-high {
                    background: rgba(234, 67, 53, 0.1);
                    color: #ea4335;
                }
                
                .risk-critical {
                    background: rgba(197, 34, 31, 0.1);
                    color: #c5221f;
                }
                
                .readmission-stats-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 16px;
                    margin-bottom: 24px;
                }
                
                .readmission-stat-card {
                    background: ${isDark ? '#3c4043' : '#f8f9fa'};
                    border-radius: 12px;
                    padding: 16px;
                    text-align: center;
                }
                
                .readmission-stat-value {
                    font-size: 28px;
                    font-weight: bold;
                    color: var(--primary, #1a73e8);
                }
                
                .readmission-stat-label {
                    font-size: 12px;
                    color: ${isDark ? '#9aa0a6' : '#5f6368'};
                    margin-top: 8px;
                }
                
                .pdf-export-btn {
                    margin-top: 16px;
                    padding: 8px 16px;
                    background: var(--primary, #1a73e8);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 14px;
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    transition: all 0.2s ease;
                }
                
                .pdf-export-btn:hover {
                    background: var(--primary-dark, #0d47a1);
                    transform: translateY(-2px);
                }
                
                @media (max-width: 768px) {
                    .readmission-container {
                        padding: 12px;
                    }
                    
                    .readmission-stats-grid {
                        grid-template-columns: repeat(2, 1fr);
                    }
                    
                    .readmission-stat-value {
                        font-size: 22px;
                    }
                }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }
}

// ──────────────────────────────────────────────────────────
// 13. تصدير نسخة واحدة (Singleton)
// ──────────────────────────────────────────────────────────

const readmissionChart = new ReadmissionChart();

window.readmissionChart = readmissionChart;

export default readmissionChart;
