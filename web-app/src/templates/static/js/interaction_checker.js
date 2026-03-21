/**
 * interaction_checker.js
 * ======================
 * RIVA Health Platform - Drug Interaction Checker
 * فاحص تداخلات الأدوية - واجهة المستخدم
 * 
 * المسؤوليات:
 * - التواصل مع API فحص تداخلات الأدوية
 * - عرض التنبيهات (أخضر/أصفر/أحمر) حسب شدة التداخل
 * - دعم الفحص الفردي والجماعي (Bulk)
 * - تكامل مع واجهة وصف الأدوية
 * 
 * المسار: web-app/src/static/js/interaction_checker.js
 * 
 * التحسينات:
 * - حماية من XSS
 * - فصل المهام (UI Layer)
 * - دعم الوضع الليلي (Dark Mode)
 * - إشعارات صوتية للتداخلات الخطيرة
 * - Event Delegation لتجنب تسرب الأحداث
 * - معالجة Race Conditions
 * 
 * الإصدار: 4.2.1
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
// 1. كلاس عرض واجهة المستخدم (UI Layer)
// ──────────────────────────────────────────────────────────

class InteractionCheckerUI {
    constructor() {
        this.stylesInjected = false;
        this.darkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
        
        // مراقبة تغيير الوضع الليلي
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            this.darkMode = e.matches;
            this.updateStyles();
        });
    }
    
    async sanitize(text) {
        await loadDOMPurify();
        return purifyInstance ? purifyInstance.sanitize(text) : text;
    }
    
    async renderAlerts(containerId, alerts, isSafe) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '';
        
        if (!alerts || alerts.length === 0) {
            const safeMessage = await this.createSafeMessage();
            container.appendChild(safeMessage);
            return;
        }
        
        // ترتيب التنبيهات حسب الخطورة (الأعلى أولاً)
        const sortedAlerts = [...alerts].sort((a, b) => {
            const severityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
            const sevA = severityOrder[a.severity_code || a.severity] || 0;
            const sevB = severityOrder[b.severity_code || b.severity] || 0;
            return sevB - sevA;
        });
        
        for (const alert of sortedAlerts) {
            const alertElement = await this.createAlertElement(alert);
            container.appendChild(alertElement);
        }
        
        // إضافة الصوت للتداخلات الخطيرة
        if (!isSafe) {
            this.playWarningSound();
        }
    }
    
    async createSafeMessage() {
        const div = document.createElement('div');
        div.className = 'interaction-safe';
        
        const icon = document.createElement('span');
        icon.className = 'safe-icon';
        icon.textContent = '✅';
        div.appendChild(icon);
        
        const text = document.createElement('span');
        text.className = 'safe-text';
        text.textContent = 'لا توجد تداخلات دوائية خطيرة بين الأدوية المحددة';
        div.appendChild(text);
        
        return div;
    }
    
    async createAlertElement(alert) {
        const severity = alert.severity_code || alert.severity || 'medium';
        const severityClass = this.getSeverityClass(severity);
        
        const div = document.createElement('div');
        div.className = `interaction-alert ${severityClass}`;
        
        // Header
        const header = document.createElement('div');
        header.className = 'alert-header';
        
        const severityIcon = document.createElement('span');
        severityIcon.className = 'severity-icon';
        severityIcon.textContent = this.getSeverityIcon(severity);
        header.appendChild(severityIcon);
        
        const drugs = document.createElement('span');
        drugs.className = 'alert-drugs';
        const drugA = await this.sanitize(alert.drug_a);
        const drugB = await this.sanitize(alert.drug_b);
        drugs.textContent = `${drugA} ↔ ${drugB}`;
        header.appendChild(drugs);
        
        const severityBadge = document.createElement('span');
        severityBadge.className = 'severity-badge';
        severityBadge.textContent = this.getSeverityText(severity);
        header.appendChild(severityBadge);
        
        div.appendChild(header);
        
        // Description
        const description = document.createElement('div');
        description.className = 'alert-description';
        const descText = await this.sanitize(alert.description_ar || alert.description);
        description.textContent = descText;
        div.appendChild(description);
        
        // Recommendation
        if (alert.recommendation_ar || alert.recommendation) {
            const rec = document.createElement('div');
            rec.className = 'alert-recommendation';
            const recText = await this.sanitize(alert.recommendation_ar || alert.recommendation);
            rec.textContent = `💡 ${recText}`;
            div.appendChild(rec);
        }
        
        return div;
    }
    
    getSeverityClass(severity) {
        const classes = {
            critical: 'severity-critical',
            high: 'severity-high',
            medium: 'severity-medium',
            low: 'severity-low'
        };
        return classes[severity] || 'severity-medium';
    }
    
    getSeverityIcon(severity) {
        const icons = {
            critical: '🔴',
            high: '🟠',
            medium: '🟡',
            low: '🟢'
        };
        return icons[severity] || '🟡';
    }
    
    getSeverityText(severity) {
        const texts = {
            critical: 'حرج - تدخل فوري',
            high: 'شديد - خطر كبير',
            medium: 'متوسط - يحتاج مراجعة',
            low: 'خفيف - يحتاج متابعة'
        };
        return texts[severity] || 'متوسط';
    }
    
    playWarningSound() {
        try {
            // استخدام ملف صوتي محلي بدلاً من base64
            const audio = new Audio('/static/assets/audio/emergency-alert.mp3');
            audio.volume = 0.5;
            audio.play().catch(e => console.log('Audio play failed:', e));
        } catch (e) {
            console.log('Sound not supported');
        }
    }
    
    async renderMedicinesList(containerId, medicines) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '';
        
        if (!medicines || medicines.length === 0) {
            const emptyMsg = document.createElement('p');
            emptyMsg.className = 'text-muted';
            emptyMsg.textContent = 'لا توجد أدوية مضافة';
            container.appendChild(emptyMsg);
            return;
        }
        
        for (const med of medicines) {
            const chip = document.createElement('span');
            chip.className = 'medicine-chip';
            chip.setAttribute('data-medicine', med);
            
            const name = await this.sanitize(med);
            chip.textContent = name;
            
            const removeBtn = document.createElement('button');
            removeBtn.className = 'remove-medicine';
            removeBtn.textContent = '✕';
            chip.appendChild(removeBtn);
            
            container.appendChild(chip);
        }
    }
    
    getColorForTheme(lightColor, darkColor) {
        return this.darkMode ? darkColor : lightColor;
    }
    
    updateStyles() {
        const styleElement = document.getElementById('riva-interaction-styles');
        if (styleElement) {
            styleElement.remove();
            this.stylesInjected = false;
            this.injectStyles();
        }
    }
    
    injectStyles() {
        if (this.stylesInjected) return;
        if (document.getElementById('riva-interaction-styles')) return;
        
        const isDark = this.darkMode;
        
        const styles = `
            <style id="riva-interaction-styles">
                .interaction-safe {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    padding: 16px;
                    background: ${isDark ? 'rgba(52, 168, 83, 0.2)' : 'rgba(52, 168, 83, 0.1)'};
                    border-radius: 12px;
                    border-right: 4px solid var(--success, #34a853);
                }
                
                .safe-icon {
                    font-size: 24px;
                }
                
                .safe-text {
                    color: var(--success, #34a853);
                    font-weight: 500;
                }
                
                .interaction-alert {
                    margin-bottom: 12px;
                    padding: 12px 16px;
                    border-radius: 12px;
                    background: ${isDark ? '#2d2e2e' : '#ffffff'};
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    transition: all 0.2s ease;
                }
                
                .interaction-alert:hover {
                    transform: translateX(-4px);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                }
                
                .severity-critical {
                    border-right: 4px solid #c5221f;
                    background: ${isDark ? 'linear-gradient(90deg, #2d2e2e, rgba(197, 34, 31, 0.1))' : 'linear-gradient(90deg, #fff, rgba(197, 34, 31, 0.05))'};
                }
                
                .severity-high {
                    border-right: 4px solid #ea4335;
                    background: ${isDark ? 'linear-gradient(90deg, #2d2e2e, rgba(234, 67, 53, 0.1))' : 'linear-gradient(90deg, #fff, rgba(234, 67, 53, 0.05))'};
                }
                
                .severity-medium {
                    border-right: 4px solid #fbbc04;
                    background: ${isDark ? 'linear-gradient(90deg, #2d2e2e, rgba(251, 188, 4, 0.1))' : 'linear-gradient(90deg, #fff, rgba(251, 188, 4, 0.05))'};
                }
                
                .severity-low {
                    border-right: 4px solid #34a853;
                    background: ${isDark ? 'linear-gradient(90deg, #2d2e2e, rgba(52, 168, 83, 0.1))' : 'linear-gradient(90deg, #fff, rgba(52, 168, 83, 0.05))'};
                }
                
                .alert-header {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    flex-wrap: wrap;
                    margin-bottom: 8px;
                }
                
                .severity-icon {
                    font-size: 18px;
                }
                
                .alert-drugs {
                    font-weight: 600;
                    font-size: 14px;
                    flex: 1;
                    color: ${isDark ? '#e8eaed' : '#202124'};
                }
                
                .severity-badge {
                    font-size: 11px;
                    padding: 2px 8px;
                    border-radius: 20px;
                    background: ${isDark ? '#3c4043' : '#e8eaed'};
                    font-weight: 500;
                    color: ${isDark ? '#e8eaed' : '#202124'};
                }
                
                .severity-critical .severity-badge {
                    background: #c5221f;
                    color: white;
                }
                
                .severity-high .severity-badge {
                    background: #ea4335;
                    color: white;
                }
                
                .severity-medium .severity-badge {
                    background: #fbbc04;
                    color: #202124;
                }
                
                .severity-low .severity-badge {
                    background: #34a853;
                    color: white;
                }
                
                .alert-description {
                    font-size: 13px;
                    color: ${isDark ? '#9aa0a6' : '#5f6368'};
                    margin-bottom: 8px;
                    line-height: 1.5;
                }
                
                .alert-recommendation {
                    font-size: 12px;
                    padding-top: 8px;
                    border-top: 1px dashed ${isDark ? '#3c4043' : '#e8eaed'};
                    color: ${isDark ? '#e8eaed' : '#3c4043'};
                }
                
                .medicines-list {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 8px;
                    margin: 12px 0;
                }
                
                .medicine-chip {
                    display: inline-flex;
                    align-items: center;
                    gap: 6px;
                    padding: 4px 8px 4px 4px;
                    background: ${isDark ? '#3c4043' : '#f8f9fa'};
                    border-radius: 20px;
                    font-size: 13px;
                    border: 1px solid ${isDark ? '#5f6368' : '#e8eaed'};
                    color: ${isDark ? '#e8eaed' : '#202124'};
                }
                
                .remove-medicine {
                    background: none;
                    border: none;
                    cursor: pointer;
                    color: ${isDark ? '#9aa0a6' : '#5f6368'};
                    font-size: 12px;
                    padding: 0 4px;
                    border-radius: 50%;
                }
                
                .remove-medicine:hover {
                    color: var(--danger, #ea4335);
                    background: rgba(234, 67, 53, 0.2);
                }
                
                .interaction-loader {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                    padding: 20px;
                    color: ${isDark ? '#e8eaed' : '#5f6368'};
                }
                
                .interaction-loader .spinner {
                    width: 20px;
                    height: 20px;
                    border: 2px solid ${isDark ? '#5f6368' : '#e8eaed'};
                    border-top-color: var(--primary, #1a73e8);
                    border-radius: 50%;
                    animation: spin 0.8s linear infinite;
                }
                
                .interaction-error {
                    padding: 16px;
                    background: ${isDark ? 'rgba(234, 67, 53, 0.2)' : 'rgba(234, 67, 53, 0.1)'};
                    border-radius: 12px;
                    border-right: 4px solid var(--danger, #ea4335);
                    color: var(--danger, #ea4335);
                }
                
                .text-muted {
                    color: ${isDark ? '#9aa0a6' : '#5f6368'};
                    padding: 8px;
                    text-align: center;
                }
                
                @keyframes spin {
                    to { transform: rotate(360deg); }
                }
                
                @media (max-width: 768px) {
                    .alert-header {
                        flex-direction: column;
                        align-items: flex-start;
                    }
                    
                    .alert-drugs {
                        width: 100%;
                    }
                }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
        this.stylesInjected = true;
    }
    
    showLoader(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '';
        const loader = document.createElement('div');
        loader.className = 'interaction-loader';
        
        const spinner = document.createElement('div');
        spinner.className = 'spinner';
        loader.appendChild(spinner);
        
        const text = document.createElement('span');
        text.textContent = 'جاري فحص التداخلات...';
        loader.appendChild(text);
        
        container.appendChild(loader);
    }
    
    async showError(containerId, message) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '';
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'interaction-error';
        
        const icon = document.createElement('span');
        icon.textContent = '⚠️ ';
        errorDiv.appendChild(icon);
        
        const text = document.createElement('span');
        text.textContent = await this.sanitize(message);
        errorDiv.appendChild(text);
        
        container.appendChild(errorDiv);
    }
}

// ──────────────────────────────────────────────────────────
// 2. الكلاس الرئيسي (Business Logic)
// ──────────────────────────────────────────────────────────

class InteractionChecker {
    constructor() {
        this.apiClient = window.rivaClient || null;
        this.ui = new InteractionCheckerUI();
        this.currentMedicines = [];
        this.currentAlerts = [];
        this.medicineListContainerId = null;
        
        this.init();
    }
    
    init() {
        console.log('[InteractionChecker] Initialized');
        
        if (!this.apiClient) {
            window.addEventListener('riva-client-ready', () => {
                this.apiClient = window.rivaClient;
                console.log('[InteractionChecker] API Client connected');
            });
        }
        
        this.ui.injectStyles();
    }
    
    // ──────────────────────────────────────────────────────────
    // 3. فحص تداخل دواء واحد
    // ──────────────────────────────────────────────────────────
    
    async checkSingleInteraction(newDrug, currentDrugs, resultContainerId) {
        if (!this.apiClient) {
            console.error('[InteractionChecker] API Client not available');
            return;
        }
        
        this.ui.showLoader(resultContainerId);
        
        try {
            const result = await this.apiClient.checkSingleInteraction(newDrug, currentDrugs);
            
            if (result.success) {
                this.currentAlerts = result.alerts || [];
                await this.ui.renderAlerts(resultContainerId, this.currentAlerts, result.safe);
                
                return result;
            }
            
            throw new Error(result.detail || 'Check failed');
            
        } catch (error) {
            console.error('[InteractionChecker] Check failed:', error);
            await this.ui.showError(resultContainerId, error.message);
            return null;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 4. فحص تداخلات مجموعة أدوية (Bulk)
    // ──────────────────────────────────────────────────────────
    
    async checkBulkInteractions(medications, resultContainerId) {
        if (!this.apiClient) {
            console.error('[InteractionChecker] API Client not available');
            return;
        }
        
        if (!medications || medications.length < 2) {
            await this.ui.renderAlerts(resultContainerId, [], true);
            return;
        }
        
        this.ui.showLoader(resultContainerId);
        
        try {
            const result = await this.apiClient.checkBulkInteractions(medications);
            
            if (result.success) {
                this.currentAlerts = result.alerts || [];
                await this.ui.renderAlerts(resultContainerId, this.currentAlerts, result.is_safe);
                
                return result;
            }
            
            throw new Error(result.detail || 'Bulk check failed');
            
        } catch (error) {
            console.error('[InteractionChecker] Bulk check failed:', error);
            await this.ui.showError(resultContainerId, error.message);
            return null;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 5. فحص تداخلات في واجهة الوصفات
    // ──────────────────────────────────────────────────────────
    
    async checkPrescriptionInteractions(newMedications, existingMedications, resultContainerId) {
        const allMedications = [...existingMedications, ...newMedications];
        return this.checkBulkInteractions(allMedications, resultContainerId);
    }
    
    // ──────────────────────────────────────────────────────────
    // 6. فحص تداخلات لمريض معين
    // ──────────────────────────────────────────────────────────
    
    async checkPatientInteractions(patientId, resultContainerId) {
        if (!this.apiClient) {
            console.error('[InteractionChecker] API Client not available');
            return;
        }
        
        this.ui.showLoader(resultContainerId);
        
        try {
            const result = await this.apiClient.getPatientPrescriptions(patientId);
            
            if (result.success && result.prescriptions) {
                // استخراج الأدوية من الروشتات
                const medications = [];
                for (const rx of result.prescriptions) {
                    if (rx.medications) {
                        for (const med of rx.medications) {
                            medications.push(med.name);
                        }
                    }
                }
                
                const uniqueMedications = [...new Set(medications)];
                return this.checkBulkInteractions(uniqueMedications, resultContainerId);
            }
            
            await this.ui.showError(resultContainerId, result.detail || 'No prescriptions found');
            return null;
            
        } catch (error) {
            console.error('[InteractionChecker] Patient check failed:', error);
            await this.ui.showError(resultContainerId, error.message);
            return null;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 7. إضافة دواء للقائمة (مع await لمعالجة Race Condition)
    // ──────────────────────────────────────────────────────────
    
    async addMedicine(medicine, listContainerId) {
        if (!medicine || medicine.trim() === '') return;
        
        const medName = medicine.trim();
        if (!this.currentMedicines.includes(medName)) {
            this.currentMedicines.push(medName);
            this.medicineListContainerId = listContainerId;
            
            // ✅ استخدام await لضمان اكتمال الـ render قبل إعداد الأحداث
            await this.ui.renderMedicinesList(listContainerId, this.currentMedicines);
            
            // ✅ استخدام Event Delegation بدلاً من ربط الأحداث مباشرة
            this.setupEventDelegation(listContainerId);
        }
    }
    
    // ✅ Event Delegation - ربط الحدث مرة واحدة على الحاوية
    setupEventDelegation(listContainerId) {
        const container = document.getElementById(listContainerId);
        if (!container) return;
        
        // إزالة المستمع القديم إذا كان موجوداً
        if (container._removeHandler) {
            container.removeEventListener('click', container._removeHandler);
        }
        
        // إنشاء مستمع جديد
        const handler = (e) => {
            const target = e.target;
            if (target.classList.contains('remove-medicine')) {
                const chip = target.closest('.medicine-chip');
                if (chip) {
                    const medicine = chip.getAttribute('data-medicine');
                    if (medicine) {
                        this.removeMedicine(medicine, listContainerId);
                    }
                }
            }
        };
        
        container._removeHandler = handler;
        container.addEventListener('click', handler);
    }
    
    async removeMedicine(medicine, listContainerId) {
        const index = this.currentMedicines.indexOf(medicine);
        if (index !== -1) {
            this.currentMedicines.splice(index, 1);
            await this.ui.renderMedicinesList(listContainerId, this.currentMedicines);
            // Event Delegation لا يحتاج إعادة ربط
        }
    }
    
    async clearMedicines(listContainerId) {
        this.currentMedicines = [];
        await this.ui.renderMedicinesList(listContainerId, []);
    }
    
    // ──────────────────────────────────────────────────────────
    // 8. الحصول على إحصائيات
    // ──────────────────────────────────────────────────────────
    
    getStats() {
        const highRiskAlerts = this.currentAlerts.filter(a => {
            const sev = a.severity_code || a.severity;
            return sev === 'high' || sev === 'critical';
        });
        
        return {
            totalAlerts: this.currentAlerts.length,
            highRiskAlerts: highRiskAlerts.length,
            medicinesCount: this.currentMedicines.length,
            hasInteractions: this.currentAlerts.length > 0
        };
    }
    
    // ──────────────────────────────────────────────────────────
    // 9. إعادة تعيين الحالة
    // ──────────────────────────────────────────────────────────
    
    reset() {
        this.currentMedicines = [];
        this.currentAlerts = [];
        if (this.medicineListContainerId) {
            this.ui.renderMedicinesList(this.medicineListContainerId, []);
        }
    }
}

// ──────────────────────────────────────────────────────────
// 10. تصدير نسخة واحدة (Singleton)
// ──────────────────────────────────────────────────────────

const interactionChecker = new InteractionChecker();

window.interactionChecker = interactionChecker;

export default interactionChecker;
