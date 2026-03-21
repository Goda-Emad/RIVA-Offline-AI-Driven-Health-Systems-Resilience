/**
 * charts.js
 * =========
 * RIVA Health Platform - Charts Core Module
 * الوحدة الأساسية للرسوم البيانية
 * 
 * المسؤوليات:
 * - تهيئة مكتبة الرسوم البيانية (ApexCharts)
 * - توفير دوال مساعدة لإنشاء أنواع مختلفة من الرسوم
 * - إدارة السمات (Themes) والتفاعل
 * - تكامل مع صفحات التنبؤات والداشبورد
 * 
 * المسار: web-app/src/static/js/charts.js
 */

class RIVACharts {
    constructor() {
        this.charts = {};
        this.theme = this.detectTheme();
        this.defaultColors = {
            primary: '#1a73e8',
            secondary: '#34a853',
            danger: '#ea4335',
            warning: '#fbbc04',
            info: '#4285f4',
            success: '#34a853',
            dark: '#202124',
            gray: '#5f6368',
            light: '#f8f9fa'
        };
        
        this.init();
    }

    // ──────────────────────────────────────────────────────────
    // 1. التهيئة
    // ──────────────────────────────────────────────────────────

    init() {
        console.log('[RIVACharts] Initialized');
        this.loadApexCharts();
    }

    loadApexCharts() {
        if (typeof ApexCharts !== 'undefined') {
            console.log('[RIVACharts] ApexCharts already loaded');
            return;
        }
        
        // تحميل ApexCharts ديناميكياً
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/apexcharts';
        script.onload = () => {
            console.log('[RIVACharts] ApexCharts loaded');
            window.dispatchEvent(new Event('apexcharts-loaded'));
        };
        document.head.appendChild(script);
    }

    detectTheme() {
        const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        return isDark ? 'dark' : 'light';
    }

    // ──────────────────────────────────────────────────────────
    // 2. إعدادات السمات (Themes)
    // ──────────────────────────────────────────────────────────

    getThemeConfig() {
        if (this.theme === 'dark') {
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
    // 3. رسم بياني خطي (Line Chart)
    // ──────────────────────────────────────────────────────────

    createLineChart(elementId, data, options = {}) {
        const theme = this.getThemeConfig();
        
        const defaultOptions = {
            chart: {
                type: 'line',
                height: 350,
                toolbar: {
                    show: true,
                    tools: {
                        download: true,
                        selection: true,
                        zoom: true,
                        zoomin: true,
                        zoomout: true,
                        pan: true,
                        reset: true
                    }
                },
                animations: {
                    enabled: true,
                    easing: 'easeinout',
                    speed: 800
                },
                background: theme.background
            },
            stroke: {
                curve: 'smooth',
                width: 3
            },
            colors: [this.defaultColors.primary, this.defaultColors.secondary, this.defaultColors.warning],
            grid: {
                borderColor: theme.gridColor,
                strokeDashArray: 5,
                position: 'back'
            },
            xaxis: {
                categories: data.categories || [],
                labels: {
                    style: {
                        colors: theme.textColor,
                        fontSize: '12px'
                    }
                },
                title: {
                    text: options.xTitle || '',
                    style: {
                        color: theme.textColor
                    }
                }
            },
            yaxis: {
                labels: {
                    style: {
                        colors: theme.textColor
                    }
                },
                title: {
                    text: options.yTitle || '',
                    style: {
                        color: theme.textColor
                    }
                }
            },
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
                shared: true,
                intersect: false
            },
            legend: {
                labels: {
                    colors: theme.textColor
                },
                position: 'top'
            }
        };
        
        const finalOptions = { ...defaultOptions, ...options };
        finalOptions.series = data.series || [];
        
        this.charts[elementId] = new ApexCharts(
            document.getElementById(elementId),
            finalOptions
        );
        
        this.charts[elementId].render();
        return this.charts[elementId];
    }

    // ──────────────────────────────────────────────────────────
    // 4. رسم بياني دائري (Pie Chart)
    // ──────────────────────────────────────────────────────────

    createPieChart(elementId, data, options = {}) {
        const theme = this.getThemeConfig();
        
        const defaultOptions = {
            chart: {
                type: 'pie',
                height: 350,
                background: theme.background
            },
            labels: data.labels || [],
            colors: [this.defaultColors.primary, this.defaultColors.secondary, 
                     this.defaultColors.warning, this.defaultColors.danger, 
                     this.defaultColors.info],
            title: {
                text: options.title || '',
                align: 'center',
                style: {
                    fontSize: '16px',
                    fontWeight: 'bold',
                    color: theme.textColor
                }
            },
            legend: {
                labels: {
                    colors: theme.textColor
                },
                position: 'bottom'
            },
            tooltip: {
                theme: theme.tooltipTheme,
                y: {
                    formatter: function(val) {
                        return val + '%';
                    }
                }
            },
            dataLabels: {
                enabled: true,
                style: {
                    fontSize: '12px'
                }
            }
        };
        
        const finalOptions = { ...defaultOptions, ...options };
        finalOptions.series = data.series || [];
        
        this.charts[elementId] = new ApexCharts(
            document.getElementById(elementId),
            finalOptions
        );
        
        this.charts[elementId].render();
        return this.charts[elementId];
    }

    // ──────────────────────────────────────────────────────────
    // 5. رسم بياني شريطي (Bar Chart)
    // ──────────────────────────────────────────────────────────

    createBarChart(elementId, data, options = {}) {
        const theme = this.getThemeConfig();
        
        const defaultOptions = {
            chart: {
                type: 'bar',
                height: 350,
                toolbar: {
                    show: true
                },
                background: theme.background
            },
            plotOptions: {
                bar: {
                    horizontal: false,
                    columnWidth: '55%',
                    borderRadius: 4,
                    dataLabels: {
                        position: 'top'
                    }
                }
            },
            colors: [this.defaultColors.primary],
            grid: {
                borderColor: theme.gridColor,
                strokeDashArray: 5
            },
            xaxis: {
                categories: data.categories || [],
                labels: {
                    style: {
                        colors: theme.textColor,
                        fontSize: '12px'
                    },
                    rotate: -45,
                    rotateAlways: false
                },
                title: {
                    text: options.xTitle || '',
                    style: {
                        color: theme.textColor
                    }
                }
            },
            yaxis: {
                labels: {
                    style: {
                        colors: theme.textColor
                    }
                },
                title: {
                    text: options.yTitle || '',
                    style: {
                        color: theme.textColor
                    }
                }
            },
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
                theme: theme.tooltipTheme
            },
            legend: {
                labels: {
                    colors: theme.textColor
                }
            }
        };
        
        const finalOptions = { ...defaultOptions, ...options };
        finalOptions.series = data.series || [];
        
        this.charts[elementId] = new ApexCharts(
            document.getElementById(elementId),
            finalOptions
        );
        
        this.charts[elementId].render();
        return this.charts[elementId];
    }

    // ──────────────────────────────────────────────────────────
    // 6. رسم بياني للمخاطر (Risk Gauge)
    // ──────────────────────────────────────────────────────────

    createRiskGauge(elementId, value, options = {}) {
        const theme = this.getThemeConfig();
        
        // تحديد لون المؤشر حسب القيمة
        let color = this.defaultColors.success;
        if (value > 70) color = this.defaultColors.danger;
        else if (value > 40) color = this.defaultColors.warning;
        
        const defaultOptions = {
            chart: {
                type: 'radialBar',
                height: 350,
                background: theme.background
            },
            plotOptions: {
                radialBar: {
                    startAngle: -135,
                    endAngle: 135,
                    hollow: {
                        margin: 0,
                        size: '70%',
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
                            fontSize: '16px',
                            fontWeight: 'bold',
                            color: theme.textColor,
                            offsetY: -10
                        },
                        value: {
                            show: true,
                            fontSize: '36px',
                            fontWeight: 'bold',
                            color: color,
                            formatter: function(val) {
                                return val + '%';
                            },
                            offsetY: 10
                        }
                    }
                }
            },
            colors: [color],
            stroke: {
                lineCap: 'round'
            },
            labels: [options.label || 'Risk Level'],
            title: {
                text: options.title || '',
                align: 'center',
                style: {
                    fontSize: '16px',
                    fontWeight: 'bold',
                    color: theme.textColor
                }
            }
        };
        
        const finalOptions = { ...defaultOptions, ...options };
        finalOptions.series = [value];
        
        this.charts[elementId] = new ApexCharts(
            document.getElementById(elementId),
            finalOptions
        );
        
        this.charts[elementId].render();
        return this.charts[elementId];
    }

    // ──────────────────────────────────────────────────────────
    // 7. رسم بياني مساحي (Area Chart)
    // ──────────────────────────────────────────────────────────

    createAreaChart(elementId, data, options = {}) {
        const theme = this.getThemeConfig();
        
        const defaultOptions = {
            chart: {
                type: 'area',
                height: 350,
                toolbar: {
                    show: true
                },
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
                    shadeIntensity: 1,
                    opacityFrom: 0.7,
                    opacityTo: 0.3,
                    stops: [0, 90, 100]
                }
            },
            colors: [this.defaultColors.primary],
            grid: {
                borderColor: theme.gridColor,
                strokeDashArray: 5
            },
            xaxis: {
                categories: data.categories || [],
                labels: {
                    style: {
                        colors: theme.textColor
                    }
                }
            },
            yaxis: {
                labels: {
                    style: {
                        colors: theme.textColor
                    }
                }
            },
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
                theme: theme.tooltipTheme
            },
            legend: {
                labels: {
                    colors: theme.textColor
                }
            }
        };
        
        const finalOptions = { ...defaultOptions, ...options };
        finalOptions.series = data.series || [];
        
        this.charts[elementId] = new ApexCharts(
            document.getElementById(elementId),
            finalOptions
        );
        
        this.charts[elementId].render();
        return this.charts[elementId];
    }

    // ──────────────────────────────────────────────────────────
    // 8. تحديث الرسم البياني (عام)
    // ──────────────────────────────────────────────────────────

    updateChart(elementId, newData, options = {}) {
        const chart = this.charts[elementId];
        if (!chart) {
            console.warn(`[RIVACharts] Chart ${elementId} not found`);
            return false;
        }
        
        try {
            if (newData.series) {
                chart.updateSeries(newData.series);
            }
            if (newData.categories && chart.updateOptions) {
                chart.updateOptions({
                    xaxis: { categories: newData.categories }
                });
            }
            console.log(`[RIVACharts] Chart ${elementId} updated`);
            return true;
        } catch (error) {
            console.error(`[RIVACharts] Failed to update chart ${elementId}:`, error);
            return false;
        }
    }

    // ──────────────────────────────────────────────────────────
    // 9. تحديث الرسم البياني الدائري (Pie Chart)
    // ──────────────────────────────────────────────────────────

    updatePieChart(elementId, series, labels = null) {
        const chart = this.charts[elementId];
        if (!chart) {
            console.warn(`[RIVACharts] Pie chart ${elementId} not found`);
            return false;
        }
        
        try {
            // تحديث القيم
            chart.updateSeries(series);
            
            // تحديث التصنيفات إذا وُجدت
            if (labels && chart.updateOptions) {
                chart.updateOptions({
                    labels: labels
                });
            }
            
            console.log(`[RIVACharts] Pie chart ${elementId} updated successfully`);
            return true;
            
        } catch (error) {
            console.error(`[RIVACharts] Failed to update pie chart ${elementId}:`, error);
            return false;
        }
    }

    // ──────────────────────────────────────────────────────────
    // 10. تحديث الرسم البياني الشريطي (Bar Chart)
    // ──────────────────────────────────────────────────────────

    updateBarChart(elementId, series, categories = null) {
        const chart = this.charts[elementId];
        if (!chart) {
            console.warn(`[RIVACharts] Bar chart ${elementId} not found`);
            return false;
        }
        
        try {
            chart.updateSeries(series);
            
            if (categories && chart.updateOptions) {
                chart.updateOptions({
                    xaxis: { categories: categories }
                });
            }
            
            console.log(`[RIVACharts] Bar chart ${elementId} updated`);
            return true;
        } catch (error) {
            console.error(`[RIVACharts] Failed to update bar chart ${elementId}:`, error);
            return false;
        }
    }

    // ──────────────────────────────────────────────────────────
    // 11. تحديث الرسم البياني الخطي (Line Chart)
    // ──────────────────────────────────────────────────────────

    updateLineChart(elementId, series, categories = null) {
        const chart = this.charts[elementId];
        if (!chart) {
            console.warn(`[RIVACharts] Line chart ${elementId} not found`);
            return false;
        }
        
        try {
            chart.updateSeries(series);
            
            if (categories && chart.updateOptions) {
                chart.updateOptions({
                    xaxis: { categories: categories }
                });
            }
            
            console.log(`[RIVACharts] Line chart ${elementId} updated`);
            return true;
        } catch (error) {
            console.error(`[RIVACharts] Failed to update line chart ${elementId}:`, error);
            return false;
        }
    }

    // ──────────────────────────────────────────────────────────
    // 12. تحديث مؤشر المخاطر (Risk Gauge)
    // ──────────────────────────────────────────────────────────

    updateRiskGauge(elementId, value, label = null) {
        const chart = this.charts[elementId];
        if (!chart) {
            console.warn(`[RIVACharts] Risk gauge ${elementId} not found`);
            return false;
        }
        
        try {
            // تحديث القيمة
            chart.updateSeries([value]);
            
            // تحديث اللون حسب القيمة
            let color = this.defaultColors.success;
            if (value > 70) color = this.defaultColors.danger;
            else if (value > 40) color = this.defaultColors.warning;
            
            chart.updateOptions({
                colors: [color],
                plotOptions: {
                    radialBar: {
                        dataLabels: {
                            value: {
                                color: color
                            }
                        }
                    }
                }
            });
            
            // تحديث التصنيف إذا وُجد
            if (label && chart.updateOptions) {
                chart.updateOptions({
                    labels: [label]
                });
            }
            
            console.log(`[RIVACharts] Risk gauge ${elementId} updated to ${value}%`);
            return true;
        } catch (error) {
            console.error(`[RIVACharts] Failed to update risk gauge ${elementId}:`, error);
            return false;
        }
    }

    // ──────────────────────────────────────────────────────────
    // 13. تصدير الرسم البياني كصورة
    // ──────────────────────────────────────────────────────────

    async exportChart(elementId, format = 'png') {
        const chart = this.charts[elementId];
        if (!chart) {
            console.warn(`[RIVACharts] Chart ${elementId} not found`);
            return null;
        }
        
        try {
            const dataUrl = await chart.dataURI();
            const link = document.createElement('a');
            link.download = `chart_${elementId}_${Date.now()}.${format}`;
            link.href = dataUrl;
            link.click();
            console.log(`[RIVACharts] Chart ${elementId} exported as ${format}`);
            return dataUrl;
        } catch (error) {
            console.error(`[RIVACharts] Failed to export chart ${elementId}:`, error);
            return null;
        }
    }

    // ──────────────────────────────────────────────────────────
    // 14. حذف الرسم البياني
    // ──────────────────────────────────────────────────────────

    destroyChart(elementId) {
        const chart = this.charts[elementId];
        if (chart) {
            chart.destroy();
            delete this.charts[elementId];
            console.log(`[RIVACharts] Chart ${elementId} destroyed`);
            return true;
        }
        return false;
    }

    // ──────────────────────────────────────────────────────────
    // 15. حذف جميع الرسوم البيانية
    // ──────────────────────────────────────────────────────────

    destroyAllCharts() {
        Object.keys(this.charts).forEach(elementId => {
            this.destroyChart(elementId);
        });
        console.log('[RIVACharts] All charts destroyed');
    }

    // ──────────────────────────────────────────────────────────
    // 16. تغيير السمة (Theme)
    // ──────────────────────────────────────────────────────────

    setTheme(theme) {
        if (theme !== 'light' && theme !== 'dark') {
            console.warn(`[RIVACharts] Invalid theme: ${theme}`);
            return;
        }
        
        this.theme = theme;
        
        // إعادة رسم جميع الرسوم البيانية
        Object.keys(this.charts).forEach(elementId => {
            const chart = this.charts[elementId];
            if (chart && chart.updateOptions) {
                const themeConfig = this.getThemeConfig();
                chart.updateOptions({
                    chart: { background: themeConfig.background },
                    grid: { borderColor: themeConfig.gridColor },
                    tooltip: { theme: themeConfig.tooltipTheme },
                    title: { style: { color: themeConfig.textColor } },
                    xaxis: { labels: { style: { colors: themeConfig.textColor } } },
                    yaxis: { labels: { style: { colors: themeConfig.textColor } } },
                    legend: { labels: { colors: themeConfig.textColor } }
                });
            }
        });
        
        console.log(`[RIVACharts] Theme changed to ${theme}`);
    }

    // ──────────────────────────────────────────────────────────
    // 17. تغيير حجم الرسم البياني
    // ──────────────────────────────────────────────────────────

    resizeChart(elementId, height) {
        const chart = this.charts[elementId];
        if (chart) {
            chart.updateOptions({
                chart: { height: height }
            });
            console.log(`[RIVACharts] Chart ${elementId} resized to ${height}px`);
        }
    }

    // ──────────────────────────────────────────────────────────
    // 18. الحصول على بيانات الرسم البياني
    // ──────────────────────────────────────────────────────────

    getChartData(elementId) {
        const chart = this.charts[elementId];
        if (chart && chart.w) {
            return {
                series: chart.w.config.series,
                categories: chart.w.config.xaxis.categories,
                options: chart.w.config,
                type: chart.w.config.chart.type
            };
        }
        return null;
    }

    // ──────────────────────────────────────────────────────────
    // 19. إظهار/إخفاء الرسم البياني
    // ──────────────────────────────────────────────────────────

    toggleChart(elementId, show) {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.display = show ? 'block' : 'none';
            console.log(`[RIVACharts] Chart ${elementId} ${show ? 'shown' : 'hidden'}`);
        }
    }

    // ──────────────────────────────────────────────────────────
    // 20. إضافة بيانات جديدة (Append Data)
    // ──────────────────────────────────────────────────────────

    appendData(elementId, newData) {
        const chart = this.charts[elementId];
        if (!chart) {
            console.warn(`[RIVACharts] Chart ${elementId} not found`);
            return false;
        }
        
        try {
            chart.appendData(newData);
            console.log(`[RIVACharts] Data appended to chart ${elementId}`);
            return true;
        } catch (error) {
            console.error(`[RIVACharts] Failed to append data:`, error);
            return false;
        }
    }
}

// ──────────────────────────────────────────────────────────
// 21. تصدير نسخة واحدة (Singleton)
// ──────────────────────────────────────────────────────────

const rivaCharts = new RIVACharts();

// إتاحته للاستخدام العالمي
window.rivaCharts = rivaCharts;

export default rivaCharts;
