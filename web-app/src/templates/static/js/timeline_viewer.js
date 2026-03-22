/**
 * timeline_viewer.js
 * ==================
 * RIVA Health Platform - Timeline Viewer Module
 * وحدة عرض الخط الزمني للحمل والتاريخ الطبي
 * 
 * المسؤوليات:
 * - عرض الخط الزمني لمتابعة الحمل (أسابيع الحمل)
 * - عرض الأحداث الطبية الهامة (فحوصات، زيارات، تطعيمات)
 * - عرض التاريخ الطبي للمريض بشكل زمني
 * - دعم RTL للغة العربية
 * 
 * المسار: web-app/src/static/js/timeline_viewer.js
 * 
 * التحسينات:
 * - دعم RTL للغة العربية
 * - عرض تفاعلي مع تمرير الفأرة
 * - إضافة أيقونات حسب نوع الحدث
 * - دمج مع API التاريخ الطبي
 * 
 * الإصدار: 4.2.1
 */

// ──────────────────────────────────────────────────────────
// 0. التحقق من تحميل المكتبات المساعدة
// ──────────────────────────────────────────────────────────

let chartLibLoaded = false;

function loadChartLibrary() {
    return new Promise((resolve) => {
        if (typeof ApexCharts !== 'undefined') {
            chartLibLoaded = true;
            resolve();
            return;
        }
        
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/apexcharts';
        script.onload = () => {
            chartLibLoaded = true;
            resolve();
        };
        script.onerror = () => {
            console.warn('[TimelineViewer] ApexCharts failed to load, using fallback');
            resolve();
        };
        document.head.appendChild(script);
    });
}

// ──────────────────────────────────────────────────────────
// 1. كلاس Timeline Viewer
// ──────────────────────────────────────────────────────────

class TimelineViewer {
    constructor() {
        this.charts = {};
        this.initialized = false;
        this.isRTL = document.documentElement.dir === 'rtl' || document.body.dir === 'rtl';
        this.currentPatientId = null;
        this.pregnancyData = null;
        this.medicalHistory = null;
        
        this.eventIcons = {
            'checkup': '🩺',
            'lab': '🔬',
            'ultrasound': '📷',
            'vaccine': '💉',
            'medication': '💊',
            'emergency': '🚨',
            'birth': '👶',
            'visit': '🏥',
            'prescription': '📝',
            'surgery': '⚕️',
            'default': '📌'
        };
        
        this.eventColors = {
            'checkup': '#1a73e8',
            'lab': '#34a853',
            'ultrasound': '#fbbc04',
            'vaccine': '#ea4335',
            'medication': '#9c27b0',
            'emergency': '#d32f2f',
            'birth': '#ff9800',
            'visit': '#00bcd4',
            'prescription': '#3f51b5',
            'surgery': '#795548',
            'default': '#5f6368'
        };
        
        this.darkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
        
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            this.darkMode = e.matches;
            this.updateAllCharts();
        });
        
        this.init();
    }
    
    async init() {
        console.log('[TimelineViewer] Initializing...');
        await loadChartLibrary();
        this.initialized = true;
        this.injectStyles();
        console.log('[TimelineViewer] Initialized', { isRTL: this.isRTL });
    }
    
    _getElement(elementId) {
        const element = document.getElementById(elementId);
        if (!element) {
            console.warn(`[TimelineViewer] Element "${elementId}" not found`);
            return null;
        }
        return element;
    }
    
    _formatDate(dateStr) {
        if (!dateStr) return '';
        try {
            const date = new Date(dateStr);
            if (this.isRTL) {
                return `${date.getFullYear()}/${date.getMonth() + 1}/${date.getDate()}`;
            }
            return `${date.getDate()}/${date.getMonth() + 1}/${date.getFullYear()}`;
        } catch {
            return dateStr;
        }
    }
    
    _getWeekNumber(gestationalAge) {
        if (!gestationalAge) return 0;
        if (typeof gestationalAge === 'number') return gestationalAge;
        if (typeof gestationalAge === 'string') {
            const match = gestationalAge.match(/(\d+)/);
            return match ? parseInt(match[1]) : 0;
        }
        return 0;
    }
    
    _getEventIcon(type) {
        return this.eventIcons[type] || this.eventIcons.default;
    }
    
    _getEventColor(type) {
        return this.eventColors[type] || this.eventColors.default;
    }
    
    // ──────────────────────────────────────────────────────────
    // 2. عرض خط زمني للحمل (Pregnancy Timeline)
    // ──────────────────────────────────────────────────────────
    
    renderPregnancyTimeline(containerId, pregnancyData, options = {}) {
        const container = this._getElement(containerId);
        if (!container) return;
        
        this.pregnancyData = pregnancyData;
        
        const currentWeek = this._getWeekNumber(pregnancyData.gestational_age_weeks);
        const dueDate = pregnancyData.due_date ? this._formatDate(pregnancyData.due_date) : 
                        (this.isRTL ? 'غير محدد' : 'Not set');
        const riskLevel = pregnancyData.risk_level || 'medium';
        
        const riskLevelClass = {
            low: 'risk-low',
            medium: 'risk-medium',
            high: 'risk-high',
            critical: 'risk-critical'
        }[riskLevel] || 'risk-medium';
        
        const riskLevelText = {
            low: this.isRTL ? 'منخفض' : 'Low',
            medium: this.isRTL ? 'متوسط' : 'Medium',
            high: this.isRTL ? 'مرتفع' : 'High',
            critical: this.isRTL ? 'حرج' : 'Critical'
        }[riskLevel] || (this.isRTL ? 'متوسط' : 'Medium');
        
        // حساب التقدم
        const weeksProgress = (currentWeek / 40) * 100;
        
        // الأحداث الهامة في الحمل
        const milestones = this.generatePregnancyMilestones(currentWeek, pregnancyData.events || []);
        
        const html = `
            <div class="timeline-container pregnancy-timeline">
                <div class="timeline-header">
                    <h3>${this.isRTL ? '📅 متابعة الحمل' : '📅 Pregnancy Timeline'}</h3>
                    <div class="timeline-stats">
                        <div class="stat-item">
                            <span class="stat-label">${this.isRTL ? 'الأسبوع الحالي' : 'Current Week'}</span>
                            <span class="stat-value">${currentWeek}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">${this.isRTL ? 'تاريخ الولادة المتوقع' : 'Due Date'}</span>
                            <span class="stat-value">${dueDate}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">${this.isRTL ? 'مستوى الخطورة' : 'Risk Level'}</span>
                            <span class="stat-value ${riskLevelClass}">${riskLevelText}</span>
                        </div>
                    </div>
                </div>
                
                <div class="pregnancy-progress">
                    <div class="progress-track">
                        <div class="progress-fill" style="width: ${weeksProgress}%"></div>
                        <div class="progress-milestones">
                            <span class="milestone" style="left: 12.5%">📅 ${this.isRTL ? 'الثلث الأول' : '1st Trimester'}</span>
                            <span class="milestone" style="left: 37.5%">📅 ${this.isRTL ? 'الثلث الثاني' : '2nd Trimester'}</span>
                            <span class="milestone" style="left: 75%">📅 ${this.isRTL ? 'الثلث الثالث' : '3rd Trimester'}</span>
                        </div>
                    </div>
                </div>
                
                <div class="timeline-events">
                    <h4>${this.isRTL ? '📋 الأحداث الهامة' : '📋 Important Events'}</h4>
                    <div class="events-list">
                        ${milestones.map(event => `
                            <div class="event-item" data-week="${event.week}">
                                <div class="event-icon">${event.icon}</div>
                                <div class="event-content">
                                    <div class="event-week">${this.isRTL ? `الأسبوع ${event.week}` : `Week ${event.week}`}</div>
                                    <div class="event-title">${this.escapeHtml(event.title)}</div>
                                    ${event.description ? `<div class="event-description">${this.escapeHtml(event.description)}</div>` : ''}
                                </div>
                                ${event.completed ? '<span class="event-completed">✓</span>' : ''}
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
        
        container.innerHTML = html;
        
        // إضافة تفاعل للتمرير إلى الأسبوع
        this.setupEventClickHandlers(container);
    }
    
    generatePregnancyMilestones(currentWeek, customEvents = []) {
        const milestones = [
            { week: 8, icon: '❤️', title: this.isRTL ? 'بداية نبض الجنين' : 'Fetal heartbeat detected', completed: currentWeek >= 8 },
            { week: 12, icon: '🔬', title: this.isRTL ? 'فحص الثلث الأول (NT Scan)' : 'First trimester screening', completed: currentWeek >= 12 },
            { week: 16, icon: '👶', title: this.isRTL ? 'بداية الشعور بالحركة' : 'Quickening (first movements)', completed: currentWeek >= 16 },
            { week: 20, icon: '📷', title: this.isRTL ? 'السونار المفصل (Anomaly Scan)' : 'Detailed anatomy scan', completed: currentWeek >= 20 },
            { week: 24, icon: '🩺', title: this.isRTL ? 'اختبار تحمل الجلوكوز' : 'Glucose tolerance test', completed: currentWeek >= 24 },
            { week: 28, icon: '💉', title: this.isRTL ? 'تطعيم السعال الديكي' : 'Tdap vaccine', completed: currentWeek >= 28 },
            { week: 32, icon: '📊', title: this.isRTL ? 'فحص نمو الجنين' : 'Fetal growth scan', completed: currentWeek >= 32 },
            { week: 36, icon: '👩‍⚕️', title: this.isRTL ? 'فحص وضع الجنين' : 'Fetal position check', completed: currentWeek >= 36 },
            { week: 40, icon: '🎉', title: this.isRTL ? 'موعد الولادة المتوقع' : 'Estimated due date', completed: currentWeek >= 40 }
        ];
        
        // إضافة الأحداث المخصصة
        for (const event of customEvents) {
            milestones.push({
                week: event.week,
                icon: this._getEventIcon(event.type),
                title: event.title,
                description: event.description,
                completed: event.completed || currentWeek >= event.week
            });
        }
        
        return milestones.sort((a, b) => a.week - b.week);
    }
    
    setupEventClickHandlers(container) {
        const events = container.querySelectorAll('.event-item');
        events.forEach(event => {
            event.addEventListener('click', () => {
                const week = event.dataset.week;
                if (week) {
                    window.dispatchEvent(new CustomEvent('riva-timeline-event-click', {
                        detail: { week: parseInt(week), type: 'pregnancy' }
                    }));
                }
            });
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 3. عرض خط زمني للتاريخ الطبي (Medical History Timeline)
    // ──────────────────────────────────────────────────────────
    
    renderMedicalTimeline(containerId, historyData, options = {}) {
        const container = this._getElement(containerId);
        if (!container) return;
        
        this.medicalHistory = historyData;
        
        // تجميع الأحداث من مصادر مختلفة
        const events = [];
        
        // إضافة التشخيصات
        if (historyData.diagnoses) {
            for (const dx of historyData.diagnoses) {
                events.push({
                    date: dx.diagnosed_date,
                    type: 'checkup',
                    title: dx.condition,
                    description: dx.notes,
                    status: dx.status,
                    source: 'diagnosis'
                });
            }
        }
        
        // إضافة الزيارات
        if (historyData.visits) {
            for (const visit of historyData.visits) {
                events.push({
                    date: visit.visit_date,
                    type: 'visit',
                    title: visit.chief_complaint,
                    description: visit.assessment,
                    doctor: visit.doctor_name,
                    source: 'visit'
                });
            }
        }
        
        // إضافة التطعيمات
        if (historyData.immunizations) {
            for (const imm of historyData.immunizations) {
                events.push({
                    date: imm.administered_date,
                    type: 'vaccine',
                    title: imm.vaccine_name,
                    description: `${this.isRTL ? 'الجرعة' : 'Dose'} ${imm.dose_number}`,
                    source: 'immunization'
                });
            }
        }
        
        // إضافة الأدوية
        if (historyData.medications) {
            for (const med of historyData.medications) {
                events.push({
                    date: med.start_date,
                    type: 'medication',
                    title: med.name,
                    description: `${med.dosage} - ${med.frequency}`,
                    status: med.status,
                    source: 'medication'
                });
            }
        }
        
        // إضافة التحاليل
        if (historyData.lab_results) {
            for (const lab of historyData.lab_results) {
                events.push({
                    date: lab.performed_date,
                    type: 'lab',
                    title: lab.test_name,
                    description: `${lab.value} ${lab.unit || ''} - ${lab.status}`,
                    source: 'lab'
                });
            }
        }
        
        // ترتيب الأحداث حسب التاريخ (الأحدث أولاً)
        events.sort((a, b) => new Date(b.date) - new Date(a.date));
        
        // تجميع الأحداث حسب السنة
        const eventsByYear = {};
        for (const event of events) {
            const year = new Date(event.date).getFullYear();
            if (!eventsByYear[year]) eventsByYear[year] = [];
            eventsByYear[year].push(event);
        }
        
        const html = `
            <div class="timeline-container medical-timeline">
                <div class="timeline-header">
                    <h3>${this.isRTL ? '📋 التاريخ الطبي' : '📋 Medical History Timeline'}</h3>
                    <div class="timeline-stats">
                        <div class="stat-item">
                            <span class="stat-label">${this.isRTL ? 'إجمالي الأحداث' : 'Total Events'}</span>
                            <span class="stat-value">${events.length}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">${this.isRTL ? 'الفترة' : 'Period'}</span>
                            <span class="stat-value">${Object.keys(eventsByYear).length} ${this.isRTL ? 'سنة' : 'years'}</span>
                        </div>
                    </div>
                </div>
                
                <div class="timeline-years">
                    ${Object.keys(eventsByYear).sort((a, b) => b - a).map(year => `
                        <div class="timeline-year">
                            <div class="year-header">
                                <span class="year-icon">📅</span>
                                <span class="year-label">${year}</span>
                                <span class="year-count">${eventsByYear[year].length} ${this.isRTL ? 'حدث' : 'events'}</span>
                            </div>
                            <div class="year-events">
                                ${eventsByYear[year].map(event => `
                                    <div class="timeline-event" data-type="${event.type}" data-date="${event.date}">
                                        <div class="event-date">${this._formatDate(event.date)}</div>
                                        <div class="event-card" style="border-right-color: ${this._getEventColor(event.type)}">
                                            <div class="event-icon">${this._getEventIcon(event.type)}</div>
                                            <div class="event-details">
                                                <div class="event-title">${this.escapeHtml(event.title)}</div>
                                                ${event.description ? `<div class="event-description">${this.escapeHtml(event.description)}</div>` : ''}
                                                ${event.doctor ? `<div class="event-doctor">👨‍⚕️ ${this.escapeHtml(event.doctor)}</div>` : ''}
                                                ${event.status ? `<div class="event-status status-${event.status}">${this.escapeHtml(event.status)}</div>` : ''}
                                            </div>
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        
        container.innerHTML = html;
        
        this.setupEventClickHandlersMedical(container);
    }
    
    setupEventClickHandlersMedical(container) {
        const events = container.querySelectorAll('.timeline-event');
        events.forEach(event => {
            event.addEventListener('click', () => {
                const type = event.dataset.type;
                const date = event.dataset.date;
                window.dispatchEvent(new CustomEvent('riva-timeline-event-click', {
                    detail: { type, date, source: 'medical' }
                }));
            });
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 4. عرض خط زمني متكامل (Combined Timeline)
    // ──────────────────────────────────────────────────────────
    
    renderCombinedTimeline(containerId, pregnancyData, medicalHistory, options = {}) {
        const container = this._getElement(containerId);
        if (!container) return;
        
        // إنشاء تبويبات للتنقل بين الخطين الزمنيين
        const html = `
            <div class="combined-timeline">
                <div class="timeline-tabs">
                    <button class="tab-btn active" data-tab="pregnancy">${this.isRTL ? '🤰 متابعة الحمل' : '🤰 Pregnancy'}</button>
                    <button class="tab-btn" data-tab="medical">${this.isRTL ? '📋 التاريخ الطبي' : '📋 Medical History'}</button>
                </div>
                <div class="tab-content active" id="pregnancy-tab">
                    <div id="pregnancy-timeline-container"></div>
                </div>
                <div class="tab-content" id="medical-tab">
                    <div id="medical-timeline-container"></div>
                </div>
            </div>
        `;
        
        container.innerHTML = html;
        
        // عرض الخط الزمني للحمل
        const pregnancyContainer = document.getElementById('pregnancy-timeline-container');
        if (pregnancyContainer) {
            this.renderPregnancyTimeline('pregnancy-timeline-container', pregnancyData, options);
        }
        
        // عرض الخط الزمني للتاريخ الطبي
        const medicalContainer = document.getElementById('medical-timeline-container');
        if (medicalContainer) {
            this.renderMedicalTimeline('medical-timeline-container', medicalHistory, options);
        }
        
        // إضافة تفاعل التبويبات
        const tabs = container.querySelectorAll('.tab-btn');
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const targetTab = tab.dataset.tab;
                
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                
                const contents = container.querySelectorAll('.tab-content');
                contents.forEach(content => content.classList.remove('active'));
                
                if (targetTab === 'pregnancy') {
                    document.getElementById('pregnancy-tab').classList.add('active');
                } else {
                    document.getElementById('medical-tab').classList.add('active');
                }
            });
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 5. تحديث البيانات
    // ──────────────────────────────────────────────────────────
    
    updatePregnancyTimeline(containerId, pregnancyData) {
        this.renderPregnancyTimeline(containerId, pregnancyData);
    }
    
    updateMedicalTimeline(containerId, historyData) {
        this.renderMedicalTimeline(containerId, historyData);
    }
    
    // ──────────────────────────────────────────────────────────
    // 6. تصدير الخط الزمني كصورة
    // ──────────────────────────────────────────────────────────
    
    async exportTimelineAsImage(containerId, filename = 'timeline.png') {
        const element = this._getElement(containerId);
        if (!element) return false;
        
        if (typeof html2canvas === 'undefined') {
            console.warn('[TimelineViewer] html2canvas not loaded');
            return false;
        }
        
        try {
            const canvas = await html2canvas(element, {
                scale: 2,
                backgroundColor: this.darkMode ? '#1e1e1e' : '#ffffff'
            });
            const link = document.createElement('a');
            link.download = filename;
            link.href = canvas.toDataURL('image/png');
            link.click();
            return true;
        } catch (error) {
            console.error('[TimelineViewer] Export failed:', error);
            return false;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 7. تحديث جميع الرسوم البيانية (Dark Mode)
    // ──────────────────────────────────────────────────────────
    
    updateAllCharts() {
        // إعادة عرض الخطوط الزمنية الحالية
        if (this.pregnancyData) {
            const container = document.querySelector('.pregnancy-timeline');
            if (container && container.id) {
                this.renderPregnancyTimeline(container.id, this.pregnancyData);
            }
        }
        
        if (this.medicalHistory) {
            const container = document.querySelector('.medical-timeline');
            if (container && container.id) {
                this.renderMedicalTimeline(container.id, this.medicalHistory);
            }
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 8. دوال مساعدة
    // ──────────────────────────────────────────────────────────
    
    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    // ──────────────────────────────────────────────────────────
    // 9. إضافة أنماط CSS
    // ──────────────────────────────────────────────────────────
    
    injectStyles() {
        if (document.getElementById('riva-timeline-styles')) return;
        
        const isDark = this.darkMode;
        
        const styles = `
            <style id="riva-timeline-styles">
                .timeline-container {
                    background: ${isDark ? '#2d2e2e' : '#ffffff'};
                    border-radius: 16px;
                    padding: 20px;
                    margin: 16px 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                }
                
                .timeline-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    flex-wrap: wrap;
                    gap: 16px;
                    margin-bottom: 24px;
                    padding-bottom: 16px;
                    border-bottom: 1px solid ${isDark ? '#5f6368' : '#e8eaed'};
                }
                
                .timeline-header h3 {
                    margin: 0;
                    font-size: 18px;
                }
                
                .timeline-stats {
                    display: flex;
                    gap: 24px;
                    flex-wrap: wrap;
                }
                
                .stat-item {
                    text-align: center;
                }
                
                .stat-label {
                    display: block;
                    font-size: 11px;
                    color: ${isDark ? '#9aa0a6' : '#5f6368'};
                    margin-bottom: 4px;
                }
                
                .stat-value {
                    font-size: 18px;
                    font-weight: bold;
                }
                
                .risk-low {
                    color: #34a853;
                }
                
                .risk-medium {
                    color: #fbbc04;
                }
                
                .risk-high {
                    color: #ea4335;
                }
                
                .risk-critical {
                    color: #d32f2f;
                }
                
                .pregnancy-progress {
                    margin: 24px 0;
                }
                
                .progress-track {
                    position: relative;
                    height: 8px;
                    background: ${isDark ? '#3c4043' : '#e8eaed'};
                    border-radius: 4px;
                    overflow: visible;
                }
                
                .progress-fill {
                    height: 100%;
                    background: linear-gradient(90deg, #34a853, #1a73e8);
                    border-radius: 4px;
                    transition: width 0.5s ease;
                }
                
                .progress-milestones {
                    position: relative;
                    margin-top: 12px;
                }
                
                .milestone {
                    position: absolute;
                    transform: translateX(-50%);
                    font-size: 11px;
                    color: ${isDark ? '#9aa0a6' : '#5f6368'};
                    white-space: nowrap;
                }
                
                .timeline-events h4 {
                    margin-bottom: 16px;
                    font-size: 14px;
                    color: ${isDark ? '#9aa0a6' : '#5f6368'};
                }
                
                .events-list {
                    display: flex;
                    flex-direction: column;
                    gap: 12px;
                }
                
                .event-item {
                    display: flex;
                    align-items: center;
                    gap: 16px;
                    padding: 12px;
                    background: ${isDark ? '#3c4043' : '#f8f9fa'};
                    border-radius: 12px;
                    cursor: pointer;
                    transition: all 0.2s ease;
                }
                
                .event-item:hover {
                    transform: translateX(-4px);
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }
                
                .event-icon {
                    font-size: 24px;
                    width: 48px;
                    text-align: center;
                }
                
                .event-content {
                    flex: 1;
                }
                
                .event-week {
                    font-size: 11px;
                    color: ${isDark ? '#9aa0a6' : '#5f6368'};
                    margin-bottom: 4px;
                }
                
                .event-title {
                    font-weight: 500;
                    margin-bottom: 4px;
                }
                
                .event-description {
                    font-size: 12px;
                    color: ${isDark ? '#9aa0a6' : '#5f6368'};
                }
                
                .event-completed {
                    color: #34a853;
                    font-size: 18px;
                }
                
                .timeline-years {
                    display: flex;
                    flex-direction: column;
                    gap: 24px;
                }
                
                .timeline-year {
                    border-right: 2px solid ${isDark ? '#5f6368' : '#e8eaed'};
                    padding-right: 16px;
                }
                
                .year-header {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    margin-bottom: 16px;
                }
                
                .year-icon {
                    font-size: 20px;
                }
                
                .year-label {
                    font-size: 18px;
                    font-weight: bold;
                }
                
                .year-count {
                    font-size: 12px;
                    color: ${isDark ? '#9aa0a6' : '#5f6368'};
                }
                
                .year-events {
                    display: flex;
                    flex-direction: column;
                    gap: 16px;
                }
                
                .timeline-event {
                    cursor: pointer;
                }
                
                .event-date {
                    font-size: 11px;
                    color: ${isDark ? '#9aa0a6' : '#5f6368'};
                    margin-bottom: 4px;
                    padding-right: 12px;
                }
                
                .event-card {
                    display: flex;
                    gap: 12px;
                    padding: 12px;
                    background: ${isDark ? '#3c4043' : '#f8f9fa'};
                    border-radius: 12px;
                    border-right: 3px solid;
                    transition: all 0.2s ease;
                }
                
                .event-card:hover {
                    transform: translateX(-4px);
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }
                
                .event-details {
                    flex: 1;
                }
                
                .event-title {
                    font-weight: 500;
                    margin-bottom: 4px;
                }
                
                .event-description {
                    font-size: 12px;
                    color: ${isDark ? '#9aa0a6' : '#5f6368'};
                }
                
                .event-doctor {
                    font-size: 11px;
                    color: ${isDark ? '#9aa0a6' : '#5f6368'};
                    margin-top: 6px;
                }
                
                .event-status {
                    display: inline-block;
                    padding: 2px 8px;
                    border-radius: 12px;
                    font-size: 10px;
                    margin-top: 6px;
                }
                
                .status-active, .status-chronic {
                    background: rgba(26, 115, 232, 0.1);
                    color: #1a73e8;
                }
                
                .status-resolved {
                    background: rgba(52, 168, 83, 0.1);
                    color: #34a853;
                }
                
                .combined-timeline {
                    margin: 16px 0;
                }
                
                .timeline-tabs {
                    display: flex;
                    gap: 8px;
                    margin-bottom: 20px;
                    border-bottom: 1px solid ${isDark ? '#5f6368' : '#e8eaed'};
                }
                
                .tab-btn {
                    padding: 10px 20px;
                    background: none;
                    border: none;
                    cursor: pointer;
                    font-size: 14px;
                    font-weight: 500;
                    color: ${isDark ? '#9aa0a6' : '#5f6368'};
                    transition: all 0.2s ease;
                }
                
                .tab-btn.active {
                    color: #1a73e8;
                    border-bottom: 2px solid #1a73e8;
                }
                
                .tab-content {
                    display: none;
                }
                
                .tab-content.active {
                    display: block;
                }
                
                @media (max-width: 768px) {
                    .timeline-container {
                        padding: 12px;
                    }
                    
                    .timeline-header {
                        flex-direction: column;
                        align-items: flex-start;
                    }
                    
                    .timeline-stats {
                        width: 100%;
                        justify-content: space-between;
                    }
                    
                    .event-item {
                        flex-wrap: wrap;
                    }
                    
                    .event-card {
                        flex-wrap: wrap;
                    }
                    
                    .milestone {
                        font-size: 9px;
                        white-space: normal;
                        text-align: center;
                    }
                }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }
}

// ──────────────────────────────────────────────────────────
// 10. تصدير نسخة واحدة (Singleton)
// ──────────────────────────────────────────────────────────

const timelineViewer = new TimelineViewer();

window.timelineViewer = timelineViewer;

export default timelineViewer;
