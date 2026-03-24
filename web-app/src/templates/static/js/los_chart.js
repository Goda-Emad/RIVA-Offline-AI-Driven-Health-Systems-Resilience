/**
 * los_chart.js
 * ============
 * RIVA Health Platform - Length of Stay (LOS) Chart Module
 * وحدة الرسوم البيانية لمدة الإقامة في المستشفى
 * 
 * المسؤوليات:
 * - عرض مدة الإقامة المتوقعة (Days)
 * - رسم مخططات توزيع المدة حسب القسم
 * - عرض اتجاهات الإقامة عبر الزمن
 * - تكامل مع صفحة 14_los_dashboard.html
 * 
 * المسار: web-app/src/static/js/los_chart.js
 * 
 * التحسينات:
 * - دعم الوضع الليلي (Dark Mode)
 * - تحديث ديناميكي للبيانات
 * - عرض هامش الخطأ (Confidence Interval)
 * - تفاعل مع المستخدم (Hover, Click)
 * - التحقق من وجود ApexCharts قبل الاستخدام
 * - التحقق من وجود عناصر DOM قبل الرسم
 * - معالجة الحالات الحدودية (Edge Cases)
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
        
        // إضافة إلى قائمة الانتظار
        const checkInterval = setInterval(() => {
            if (typeof ApexCharts !== 'undefined') {
                clearInterval(checkInterval);
                apexChartsLoaded = true;
                // تنفيذ جميع العمليات المعلقة
                apexChartsQueue.forEach(cb => cb());
                apexChartsQueue = [];
                resolve();
            }
        }, 100);
        
        // timeout بعد 10 ثواني
        setTimeout(() => {
            clearInterval(checkInterval);
            if (!apexChartsLoaded) {
                console.error('[LOSChart] ApexCharts failed to load after 10 seconds');
                resolve();
            }
        }, 10000);
    });
}

// ──────────────────────────────────────────────────────────
// 1. كلاس LOS Chart
// ──────────────────────────────────────────────────────────

class LOSChart {
    constructor() {
        this.charts = {};
        this.initialized = false;
        this.colors = {
            primary: '#1a73e8',
            secondary: '#34a853',
            warning: '#fbbc04',
            danger: '#ea4335',
            info: '#4285f4',
            dark: '#202124',
            gray: '#5f6368',
            light: '#f8f9fa'
        };
        
        this.darkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
        
        // مراقبة تغيير الوضع الليلي
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            this.darkMode = e.matches;
            this.updateAllCharts();
        });
        
        this.init();
    }
    
    async init() {
        console.log('[LOSChart] Initializing...');
        await waitForApexCharts();
        this.initialized = true;
        this.injectStyles();
        console.log('[LOSChart] Initialized');
    }
    
    // التحقق من وجود العنصر في DOM
    _getElement(elementId) {
        const element = document.getElementById(elementId);
        if (!element) {
            console.warn(`[LOSChart] Element with id "${elementId}" not found in DOM`);
            return null;
        }
        return element;
    }
    
    // التحقق من جاهزية ApexCharts
    _isReady() {
        if (!this.initialized) {
            console.warn('[LOSChart] Not initialized yet. Please wait for initialization.');
            return false;
        }
        if (typeof ApexCharts === 'undefined') {
            console.error('[LOSChart] ApexCharts library not loaded');
            return false;
        }
        return true;
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
    
    // ──────────────────────────────────────────────────────────
    // 2. رسم مخطط مدة الإقامة (Gauge Chart)
    // ──────────────────────────────────────────────────────────
    
    async createLOSGauge(elementId, days, minDays = 1, maxDays = 30, options = {}) {
        if (!this._isReady()) return null;
        
        const element = this._getElement(elementId);
        if (!element) return null;
        
        // التحقق من صحة المدخلات
        const validDays = Math.max(minDays, Math.min(maxDays, days || 0));
        const percentage = Math.min(100, (validDays / maxDays) * 100);
        
        const theme = this.getThemeConfig();
        
        // تحديد اللون حسب المدة
        let color = this.colors.success;
        if (validDays > 14) color = this.colors.danger;
        else if (validDays > 7) color = this.colors.warning;
        
        const defaultOptions = {
            chart: {
                type: 'radialBar',
                height: options.height || 350,
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
                            fontSize: '32px',
                            fontWeight: 'bold',
                            color: color,
                            formatter: function(val) {
                                return Math.round(val) + ' يوم';
                            },
                            offsetY: 10
                        }
                    }
                }
            },
            colors: [color],
            stroke: { lineCap: 'round' },
            labels: [options.label || 'مدة الإقامة المتوقعة'],
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
                    formatter: function(val) {
                        return Math.round(val) + ' يوم';
                    }
                }
            }
        };
        
        const finalOptions = { ...defaultOptions, ...options };
        finalOptions.series = [percentage];
        
        try {
            // إزالة الرسم القديم إذا كان موجوداً
            if (this.charts[elementId]) {
                this.destroyChart(elementId);
            }
            
            this.charts[elementId] = new ApexCharts(element, finalOptions);
            await this.charts[elementId].render();
            return this.charts[elementId];
        } catch (error) {
            console.error(`[LOSChart] Failed to create gauge chart for ${elementId}:`, error);
            return null;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 3. رسم مخطط توزيع الإقامة (Bar Chart)
    // ──────────────────────────────────────────────────────────
    
    async createLOSDistributionChart(elementId, data, options = {}) {
        if (!this._isReady()) return null;
        
        const element = this._getElement(elementId);
        if (!element) return null;
        
        // التحقق من صحة البيانات
        if (!data || !data.series || data.series.length === 0) {
            console.warn(`[LOSChart] No data provided for distribution chart ${elementId}`);
            return null;
        }
        
        const theme = this.getThemeConfig();
        
        const defaultOptions = {
            chart: {
                type: 'bar',
                height: options.height || 350,
                toolbar: { show: true },
                background: theme.background,
                animations: {
                    enabled: true,
                    easing: 'easeinout',
                    speed: 800
                }
            },
            plotOptions: {
                bar: {
                    horizontal: false,
                    columnWidth: '60%',
                    borderRadius: 8,
                    dataLabels: { position: 'top' }
                }
            },
            colors: [this.colors.primary],
            grid: {
                borderColor: theme.gridColor,
                strokeDashArray: 5,
                position: 'back'
            },
            xaxis: {
                categories: data.categories || [],
                labels: {
                    style: { colors: theme.textColor, fontSize: '12px' },
                    rotate: -45,
                    rotateAlways: false
                },
                title: {
                    text: options.xTitle || 'مدة الإقامة (أيام)',
                    style: { color: theme.textColor }
                }
            },
            yaxis: {
                labels: {
                    style: { colors: theme.textColor },
                    formatter: function(val) {
                        return val.toFixed(0);
                    }
                },
                title: {
                    text: options.yTitle || 'عدد المرضى',
                    style: { color: theme.textColor }
                }
            },
            title: {
                text: options.title || 'توزيع مدة الإقامة',
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
                    formatter: function(val) {
                        return val + ' مريض';
                    }
                }
            },
            dataLabels: {
                enabled: true,
                offsetY: -20,
                style: {
                    fontSize: '12px',
                    colors: [theme.textColor]
                },
                formatter: function(val) {
                    return val;
                }
            }
        };
        
        const finalOptions = { ...defaultOptions, ...options };
        finalOptions.series = data.series;
        
        try {
            if (this.charts[elementId]) {
                this.destroyChart(elementId);
            }
            
            this.charts[elementId] = new ApexCharts(element, finalOptions);
            await this.charts[elementId].render();
            return this.charts[elementId];
        } catch (error) {
            console.error(`[LOSChart] Failed to create distribution chart for ${elementId}:`, error);
            return null;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 4. رسم مخطط اتجاهات الإقامة (Line Chart)
    // ──────────────────────────────────────────────────────────
    
    async createLOSTrendChart(elementId, data, options = {}) {
        if (!this._isReady()) return null;
        
        const element = this._getElement(elementId);
        if (!element) return null;
        
        if (!data || !data.series || data.series.length === 0) {
            console.warn(`[LOSChart] No data provided for trend chart ${elementId}`);
            return null;
        }
        
        const theme = this.getThemeConfig();
        
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
                }
            },
            stroke: {
                curve: 'smooth',
                width: 3
            },
            colors: [this.colors.primary, this.colors.warning],
            grid: {
                borderColor: theme.gridColor,
                strokeDashArray: 5,
                position: 'back'
            },
            xaxis: {
                categories: data.categories || [],
                labels: {
                    style: { colors: theme.textColor, fontSize: '12px' },
                    rotate: -45
                },
                title: {
                    text: options.xTitle || 'التاريخ',
                    style: { color: theme.textColor }
                }
            },
            yaxis: {
                labels: {
                    style: { colors: theme.textColor },
                    formatter: function(val) {
                        return val.toFixed(1) + ' يوم';
                    }
                },
                title: {
                    text: options.yTitle || 'متوسط مدة الإقامة (أيام)',
                    style: { color: theme.textColor }
                }
            },
            title: {
                text: options.title || 'اتجاهات مدة الإقامة',
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
                    formatter: function(val) {
                        return val.toFixed(1) + ' يوم';
                    }
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
        finalOptions.series = data.series;
        
        try {
            if (this.charts[elementId]) {
                this.destroyChart(elementId);
            }
            
            this.charts[elementId] = new ApexCharts(element, finalOptions);
            await this.charts[elementId].render();
            return this.charts[elementId];
        } catch (error) {
            console.error(`[LOSChart] Failed to create trend chart for ${elementId}:`, error);
            return null;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 5. رسم مخطط المقارنة بين الأقسام (Horizontal Bar)
    // ──────────────────────────────────────────────────────────
    
    async createLOSDepartmentChart(elementId, data, options = {}) {
        if (!this._isReady()) return null;
        
        const element = this._getElement(elementId);
        if (!element) return null;
        
        if (!data || !data.series || data.series.length === 0) {
            console.warn(`[LOSChart] No data provided for department chart ${elementId}`);
            return null;
        }
        
        const theme = this.getThemeConfig();
        
        const defaultOptions = {
            chart: {
                type: 'bar',
                height: options.height || 400,
                toolbar: { show: true },
                background: theme.background
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
                categories: data.categories || [],
                labels: {
                    style: { colors: theme.textColor },
                    formatter: function(val) {
                        return val.toFixed(1) + ' يوم';
                    }
                },
                title: {
                    text: options.xTitle || 'متوسط مدة الإقامة (أيام)',
                    style: { color: theme.textColor }
                }
            },
            yaxis: {
                labels: {
                    style: { colors: theme.textColor, fontSize: '13px' }
                },
                title: {
                    text: options.yTitle || 'القسم',
                    style: { color: theme.textColor }
                }
            },
            title: {
                text: options.title || 'متوسط مدة الإقامة حسب القسم',
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
                    formatter: function(val) {
                        return val.toFixed(1) + ' يوم';
                    }
                }
            },
            dataLabels: {
                enabled: true,
                formatter: function(val) {
                    return val.toFixed(1) + ' يوم';
                },
                style: {
                    fontSize: '12px',
                    colors: [theme.textColor]
                }
            }
        };
        
        const finalOptions = { ...defaultOptions, ...options };
        finalOptions.series = data.series;
        
        try {
            if (this.charts[elementId]) {
                this.destroyChart(elementId);
            }
            
            this.charts[elementId] = new ApexCharts(element, finalOptions);
            await this.charts[elementId].render();
            return this.charts[elementId];
        } catch (error) {
            console.error(`[LOSChart] Failed to create department chart for ${elementId}:`, error);
            return null;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 6. رسم مخطط مع هامش الخطأ (Confidence Interval)
    // ──────────────────────────────────────────────────────────
    
    async createLOSConfidenceChart(elementId, data, options = {}) {
        if (!this._isReady()) return null;
        
        const element = this._getElement(elementId);
        if (!element) return null;
        
        if (!data || !data.series || data.series.length === 0) {
            console.warn(`[LOSChart] No data provided for confidence chart ${elementId}`);
            return null;
        }
        
        const theme = this.getThemeConfig();
        
        const defaultOptions = {
            chart: {
                type: 'line',
                height: options.height || 350,
                toolbar: { show: true },
                background: theme.background
            },
            stroke: {
                curve: 'smooth',
                width: 2
            },
            fill: {
                type: 'gradient',
                gradient: {
                    shadeIntensity: 0.3,
                    opacityFrom: 0.3,
                    opacityTo: 0.1,
                    stops: [0, 100]
                }
            },
            colors: [this.colors.primary, this.colors.gray],
            grid: {
                borderColor: theme.gridColor,
                strokeDashArray: 5
            },
            xaxis: {
                categories: data.categories || [],
                labels: {
                    style: { colors: theme.textColor }
                }
            },
            yaxis: {
                labels: {
                    style: { colors: theme.textColor },
                    formatter: function(val) {
                        return val.toFixed(1) + ' يوم';
                    }
                }
            },
            title: {
                text: options.title || 'مدة الإقامة مع هامش الخطأ (95% CI)',
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
                    formatter: function(val) {
                        return val.toFixed(1) + ' يوم';
                    }
                }
            },
            legend: {
                labels: { colors: theme.textColor }
            }
        };
        
        const finalOptions = { ...defaultOptions, ...options };
        finalOptions.series = data.series;
        
        try {
            if (this.charts[elementId]) {
                this.destroyChart(elementId);
            }
            
            this.charts[elementId] = new ApexCharts(element, finalOptions);
            await this.charts[elementId].render();
            return this.charts[elementId];
        } catch (error) {
            console.error(`[LOSChart] Failed to create confidence chart for ${elementId}:`, error);
            return null;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 7. تحديث البيانات (مع التحقق من الوجود)
    // ──────────────────────────────────────────────────────────
    
    updateLOSGauge(elementId, days, maxDays = 30) {
        const chart = this.charts[elementId];
        if (!chart) {
            console.warn(`[LOSChart] Chart ${elementId} not found for update`);
            return false;
        }
        
        const validDays = Math.max(1, Math.min(maxDays, days || 0));
        const percentage = Math.min(100, (validDays / maxDays) * 100);
        
        let color = this.colors.success;
        if (validDays > 14) color = this.colors.danger;
        else if (validDays > 7) color = this.colors.warning;
        
        try {
            chart.updateSeries([percentage]);
            chart.updateOptions({
                colors: [color],
                plotOptions: {
                    radialBar: {
                        dataLabels: {
                            value: { color: color }
                        }
                    }
                }
            });
            return true;
        } catch (error) {
            console.error(`[LOSChart] Failed to update gauge chart ${elementId}:`, error);
            return false;
        }
    }
    
    updateBarChart(elementId, series, categories = null) {
        const chart = this.charts[elementId];
        if (!chart) {
            console.warn(`[LOSChart] Chart ${elementId} not found for update`);
            return false;
        }
        
        try {
            chart.updateSeries(series);
            if (categories) {
                chart.updateOptions({ xaxis: { categories: categories } });
            }
            return true;
        } catch (error) {
            console.error(`[LOSChart] Failed to update bar chart ${elementId}:`, error);
            return false;
        }
    }
    
    updateLineChart(elementId, series, categories = null) {
        const chart = this.charts[elementId];
        if (!chart) {
            console.warn(`[LOSChart] Chart ${elementId} not found for update`);
            return false;
        }
        
        try {
            chart.updateSeries(series);
            if (categories) {
                chart.updateOptions({ xaxis: { categories: categories } });
            }
            return true;
        } catch (error) {
            console.error(`[LOSChart] Failed to update line chart ${elementId}:`, error);
            return false;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 8. تحديث جميع الرسوم البيانية عند تغيير السمة
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
                    console.error(`[LOSChart] Failed to update chart ${elementId}:`, error);
                }
            }
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 9. تصدير الرسم كصورة
    // ──────────────────────────────────────────────────────────
    
    async exportChart(elementId, format = 'png') {
        const chart = this.charts[elementId];
        if (!chart) {
            console.warn(`[LOSChart] Chart ${elementId} not found for export`);
            return null;
        }
        
        try {
            const dataUrl = await chart.dataURI();
            const link = document.createElement('a');
            link.download = `los_chart_${elementId}_${Date.now()}.${format}`;
            link.href = dataUrl;
            link.click();
            return dataUrl;
        } catch (error) {
            console.error('[LOSChart] Export failed:', error);
            return null;
        }
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
                console.warn(`[LOSChart] Error destroying chart ${elementId}:`, error);
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
        if (document.getElementById('riva-los-chart-styles')) return;
        
        const isDark = this.darkMode;
        
        const styles = `
            <style id="riva-los-chart-styles">
                .los-chart-container {
                    background: ${isDark ? '#2d2e2e' : '#ffffff'};
                    border-radius: 16px;
                    padding: 20px;
                    margin: 16px 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                }
                
                .los-chart-title {
                    font-size: 18px;
                    font-weight: 600;
                    margin-bottom: 16px;
                    text-align: center;
                    color: ${isDark ? '#e8eaed' : '#202124'};
                }
                
                .los-chart-subtitle {
                    font-size: 14px;
                    color: ${isDark ? '#9aa0a6' : '#5f6368'};
                    text-align: center;
                    margin-bottom: 20px;
                }
                
                .los-stats-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 16px;
                    margin-bottom: 24px;
                }
                
                .los-stat-card {
                    background: ${isDark ? '#3c4043' : '#f8f9fa'};
                    border-radius: 12px;
                    padding: 16px;
                    text-align: center;
                }
                
                .los-stat-value {
                    font-size: 28px;
                    font-weight: bold;
                    color: var(--primary, #1a73e8);
                }
                
                .los-stat-label {
                    font-size: 12px;
                    color: ${isDark ? '#9aa0a6' : '#5f6368'};
                    margin-top: 8px;
                }
                
                .los-confidence-badge {
                    display: inline-block;
                    padding: 4px 12px;
                    border-radius: 20px;
                    font-size: 12px;
                    font-weight: 500;
                    background: rgba(52, 168, 83, 0.1);
                    color: var(--success, #34a853);
                }
                
                .los-confidence-badge.low {
                    background: rgba(234, 67, 53, 0.1);
                    color: var(--danger, #ea4335);
                }
                
                .los-confidence-badge.medium {
                    background: rgba(251, 188, 4, 0.1);
                    color: var(--warning, #fbbc04);
                }
                
                @media (max-width: 768px) {
                    .los-chart-container {
                        padding: 12px;
                    }
                    
                    .los-stats-grid {
                        grid-template-columns: repeat(2, 1fr);
                    }
                    
                    .los-stat-value {
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

// إنشاء نسخة واحدة
const losChart = new LOSChart();

// تخزين في window للاستخدام العادي
window.losChart = losChart;
window.rivaLOSChart = losChart;

// ES Module export
export default losChart;
export { losChart };
