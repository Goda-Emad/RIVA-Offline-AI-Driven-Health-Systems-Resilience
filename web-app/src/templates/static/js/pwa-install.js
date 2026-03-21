/**
 * pwa-install.js
 * ==============
 * RIVA Health Platform - PWA Installation Manager
 * مدير تثبيت التطبيق التقدمي (Progressive Web App)
 * 
 * المسؤوليات:
 * - إدارة عملية تثبيت الـ PWA
 * - عرض زر التثبيت عند توفر الإمكانية
 * - تتبع حالة التثبيت
 * - دعم التشغيل بدون إنترنت
 * - دعم أجهزة iOS (Safari)
 * 
 * المسار: web-app/src/static/js/pwa-install.js
 * 
 * التحسينات:
 * - دعم جميع المتصفحات الحديثة
 * - تجربة مستخدم سلسة
 * - تتبع الأحداث (Analytics)
 * - تخزين حالة التثبيت
 * - تعليمات خاصة لأجهزة iOS
 * 
 * الإصدار: 4.2.1
 */

// ──────────────────────────────────────────────────────────
// 1. كلاس PWA Install Manager
// ──────────────────────────────────────────────────────────

class PWAInstallManager {
    constructor() {
        this.deferredPrompt = null;
        this.isInstalled = false;
        this.isInstallable = false;
        this.installButton = null;
        this.installEventListeners = {};
        this.storageKey = 'riva_pwa_installed';
        this.lastPromptDate = null;
        this.minDaysBetweenPrompts = 7; // 7 أيام بين طلبات التثبيت
        this.isIOS = this.detectIOS();
        this.isStandalone = false;
        
        this.init();
    }
    
    // ──────────────────────────────────────────────────────────
    // 2. التهيئة
    // ──────────────────────────────────────────────────────────
    
    async init() {
        console.log('[PWAInstall] Initializing...');
        
        // التحقق من حالة التثبيت
        this.checkInstallStatus();
        
        // التحقق من وضع العرض (Standalone)
        this.checkDisplayMode();
        
        // الاستماع لحدث beforeinstallprompt (Android/Chrome)
        window.addEventListener('beforeinstallprompt', this.handleBeforeInstallPrompt.bind(this));
        
        // الاستماع لحدث appinstalled
        window.addEventListener('appinstalled', this.handleAppInstalled.bind(this));
        
        // إنشاء زر التثبيت إذا لم يكن موجوداً
        this.createInstallButton();
        
        // إذا كان iOS، أظهر التعليمات المناسبة
        if (this.isIOS && !this.isStandalone) {
            console.log('[PWAInstall] iOS device detected, showing manual guide');
            this.showIOSInstallGuide();
        }
        
        console.log('[PWAInstall] Initialized', { isIOS: this.isIOS, isStandalone: this.isStandalone });
    }
    
    // ──────────────────────────────────────────────────────────
    // 3. كشف نظام iOS
    // ──────────────────────────────────────────────────────────
    
    detectIOS() {
        const userAgent = window.navigator.userAgent.toLowerCase();
        const isIPad = userAgent.includes('ipad');
        const isIPhone = userAgent.includes('iphone');
        const isIPod = userAgent.includes('ipod');
        
        return isIPad || isIPhone || isIPod;
    }
    
    // ──────────────────────────────────────────────────────────
    // 4. التحقق من حالة التثبيت
    // ──────────────────────────────────────────────────────────
    
    checkInstallStatus() {
        // التحقق من التخزين المحلي
        const stored = localStorage.getItem(this.storageKey);
        if (stored === 'true') {
            this.isInstalled = true;
        }
        
        // التحقق من تاريخ آخر طلب
        const lastPrompt = localStorage.getItem('riva_pwa_last_prompt');
        if (lastPrompt) {
            this.lastPromptDate = new Date(parseInt(lastPrompt));
        }
    }
    
    checkDisplayMode() {
        // التحقق من وضع العرض (Standalone = تم التثبيت)
        const isStandalone = window.matchMedia('(display-mode: standalone)').matches;
        const isFullscreen = window.matchMedia('(display-mode: fullscreen)').matches;
        const isMinimalUi = window.matchMedia('(display-mode: minimal-ui)').matches;
        
        this.isStandalone = isStandalone || isFullscreen || isMinimalUi;
        
        if (this.isStandalone) {
            this.isInstalled = true;
            localStorage.setItem(this.storageKey, 'true');
            console.log('[PWAInstall] App is installed and running in standalone mode');
        }
        
        // مراقبة تغيير وضع العرض
        window.matchMedia('(display-mode: standalone)').addEventListener('change', (e) => {
            if (e.matches) {
                this.isInstalled = true;
                this.isStandalone = true;
                localStorage.setItem(this.storageKey, 'true');
                this.hideInstallButton();
                this.emit('installed');
            }
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 5. معالجة حدث beforeinstallprompt (Android/Chrome)
    // ──────────────────────────────────────────────────────────
    
    handleBeforeInstallPrompt(event) {
        console.log('[PWAInstall] beforeinstallprompt event received');
        
        // منع ظهور الـ prompt التلقائي
        event.preventDefault();
        
        // حفظ الحدث لاستخدامه لاحقاً
        this.deferredPrompt = event;
        this.isInstallable = true;
        
        // التحقق من عدم عرض الطلب مؤخراً
        const canPrompt = this.canShowPrompt();
        
        if (canPrompt && !this.isInstalled) {
            // إظهار زر التثبيت
            this.showInstallButton();
            
            // تسجيل تاريخ الطلب
            this.lastPromptDate = new Date();
            localStorage.setItem('riva_pwa_last_prompt', this.lastPromptDate.getTime().toString());
        }
        
        this.emit('installable', { canInstall: true });
    }
    
    // ──────────────────────────────────────────────────────────
    // 6. التحقق من إمكانية عرض الطلب
    // ──────────────────────────────────────────────────────────
    
    canShowPrompt() {
        if (!this.lastPromptDate) return true;
        
        const daysSinceLastPrompt = (Date.now() - this.lastPromptDate.getTime()) / (1000 * 60 * 60 * 24);
        return daysSinceLastPrompt >= this.minDaysBetweenPrompts;
    }
    
    // ──────────────────────────────────────────────────────────
    // 7. إنشاء زر التثبيت
    // ──────────────────────────────────────────────────────────
    
    createInstallButton() {
        // البحث عن زر موجود
        this.installButton = document.getElementById('install-pwa-btn');
        
        if (!this.installButton) {
            // إنشاء زر جديد
            this.installButton = document.createElement('button');
            this.installButton.id = 'install-pwa-btn';
            this.installButton.className = 'pwa-install-btn';
            this.installButton.style.display = 'none';
            this.installButton.innerHTML = `
                <span class="install-icon">📱</span>
                <span class="install-text">${this.isIOS ? 'تثبيت التطبيق (iOS)' : 'تثبيت التطبيق'}</span>
            `;
            
            // إضافة الزر للصفحة
            const container = document.querySelector('.pwa-install-container');
            if (container) {
                container.appendChild(this.installButton);
            } else {
                document.body.appendChild(this.installButton);
            }
        }
        
        // إضافة مستمع الحدث
        this.installButton.addEventListener('click', this.handleInstallClick.bind(this));
        
        // إضافة الأنماط
        this.injectStyles();
    }
    
    // ──────────────────────────────────────────────────────────
    // 8. إظهار/إخفاء زر التثبيت
    // ──────────────────────────────────────────────────────────
    
    showInstallButton() {
        if (this.installButton && !this.isInstalled && this.isInstallable && !this.isIOS) {
            this.installButton.style.display = 'flex';
            console.log('[PWAInstall] Install button shown');
        }
    }
    
    hideInstallButton() {
        if (this.installButton) {
            this.installButton.style.display = 'none';
            console.log('[PWAInstall] Install button hidden');
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 9. معالجة حدث النقر على زر التثبيت (Android/Chrome)
    // ──────────────────────────────────────────────────────────
    
    async handleInstallClick(event) {
        event.preventDefault();
        
        if (!this.deferredPrompt) {
            console.log('[PWAInstall] No deferred prompt available');
            // عرض دليل التثبيت اليدوي (لأنه لا يوجد دعم برمجي)
            this.showManualInstallGuide();
            return;
        }
        
        try {
            // إظهار الـ prompt
            this.deferredPrompt.prompt();
            
            // انتظار اختيار المستخدم
            const choiceResult = await this.deferredPrompt.userChoice;
            
            if (choiceResult.outcome === 'accepted') {
                console.log('[PWAInstall] User accepted the install prompt');
                this.isInstalled = true;
                localStorage.setItem(this.storageKey, 'true');
                this.hideInstallButton();
                this.emit('installed', { accepted: true });
                
                this.showNotification('تم تثبيت التطبيق', 'يمكنك الآن استخدام RIVA من شاشة هاتفك الرئيسية', 'success');
            } else {
                console.log('[PWAInstall] User dismissed the install prompt');
                this.emit('dismissed', { accepted: false });
            }
            
            // تنظيف الـ deferredPrompt
            this.deferredPrompt = null;
            this.isInstallable = false;
            
        } catch (error) {
            console.error('[PWAInstall] Error during installation:', error);
            this.emit('error', { error: error.message });
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 10. عرض دليل التثبيت اليدوي (لأجهزة iOS وغير المدعومة)
    // ──────────────────────────────────────────────────────────
    
    showManualInstallGuide() {
        // إنشاء مودال للتعليمات
        const modal = document.createElement('div');
        modal.className = 'pwa-install-modal';
        
        // تعليمات حسب نوع الجهاز
        const instructions = this.isIOS ? this.getIOSInstructions() : this.getAndroidInstructions();
        
        modal.innerHTML = `
            <div class="modal-overlay"></div>
            <div class="modal-content">
                <div class="modal-header">
                    <h3>📱 تثبيت تطبيق RIVA</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <p>${instructions.description}</p>
                    <div class="install-steps">
                        ${instructions.steps.map((step, index) => `
                            <div class="step">
                                <span class="step-number">${index + 1}</span>
                                <span class="step-text">${step}</span>
                            </div>
                        `).join('')}
                    </div>
                    ${instructions.images ? `
                        <div class="install-images">
                            ${instructions.images.map(img => `
                                <img src="${img}" alt="خطوة التثبيت" class="install-image">
                            `).join('')}
                        </div>
                    ` : ''}
                    <div class="install-note">
                        <span class="note-icon">💡</span>
                        <span class="note-text">${instructions.note}</span>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="modal-btn modal-btn-primary" id="modal-close-btn">فهمت ✅</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // إضافة أنماط المودال
        this.injectModalStyles();
        
        // معالجة أحداث المودال (إغلاق فقط، لا يوجد زر تثبيت)
        const closeModal = () => modal.remove();
        
        modal.querySelector('.modal-close').addEventListener('click', closeModal);
        modal.querySelector('#modal-close-btn').addEventListener('click', closeModal);
        modal.querySelector('.modal-overlay').addEventListener('click', closeModal);
        
        // إظهار المودال مع أنيميشن
        setTimeout(() => {
            modal.classList.add('show');
        }, 10);
    }
    
    // ──────────────────────────────────────────────────────────
    // 11. تعليمات لأجهزة iOS (iPhone/iPad)
    // ──────────────────────────────────────────────────────────
    
    getIOSInstructions() {
        return {
            description: 'لتثبيت تطبيق RIVA على جهاز iPhone أو iPad:',
            steps: [
                'اضغط على زر المشاركة 📤 في أسفل الشاشة',
                'مرر لأسفل واختر "إضافة إلى الشاشة الرئيسية" ➕',
                'اضغط على "إضافة" في الزاوية العلوية اليمنى'
            ],
            images: [], // يمكن إضافة صور تعليمية هنا
            note: 'بعد التثبيت، ستجد أيقونة RIVA على شاشة هاتفك الرئيسية'
        };
    }
    
    // ──────────────────────────────────────────────────────────
    // 12. تعليمات لأجهزة Android
    // ──────────────────────────────────────────────────────────
    
    getAndroidInstructions() {
        return {
            description: 'لتثبيت تطبيق RIVA على جهاز Android:',
            steps: [
                'اضغط على زر القائمة (⋮) في أعلى المتصفح',
                'اختر "تثبيت التطبيق" أو "Add to Home Screen"',
                'اضغط "تثبيت" للتأكيد'
            ],
            images: [],
            note: 'بعد التثبيت، ستجد أيقونة RIVA على شاشة هاتفك الرئيسية'
        };
    }
    
    // ──────────────────────────────────────────────────────────
    // 13. عرض دليل التثبيت لأجهزة iOS (تلقائي)
    // ──────────────────────────────────────────────────────────
    
    showIOSInstallGuide() {
        // التحقق من عدم عرض الطلب مؤخراً
        const canPrompt = this.canShowPrompt();
        
        if (canPrompt && !this.isInstalled) {
            setTimeout(() => {
                this.showManualInstallGuide();
                this.lastPromptDate = new Date();
                localStorage.setItem('riva_pwa_last_prompt', this.lastPromptDate.getTime().toString());
            }, 2000); // تأخير 2 ثانية
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 14. معالجة حدث appinstalled
    // ──────────────────────────────────────────────────────────
    
    handleAppInstalled(event) {
        console.log('[PWAInstall] App was installed');
        this.isInstalled = true;
        this.isStandalone = true;
        localStorage.setItem(this.storageKey, 'true');
        this.hideInstallButton();
        this.emit('installed', { fromEvent: true });
        
        this.showNotification('شكراً لتثبيت RIVA!', 'يمكنك الآن استخدام التطبيق بسرعة وسهولة', 'success');
        
        // تسجيل حدث في Google Analytics
        if (typeof gtag !== 'undefined') {
            gtag('event', 'pwa_installed', {
                event_category: 'pwa',
                event_label: 'app_installed'
            });
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 15. عرض إشعار
    // ──────────────────────────────────────────────────────────
    
    showNotification(title, message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `pwa-toast pwa-toast-${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <strong>${this.escapeHtml(title)}</strong>
                <p>${this.escapeHtml(message)}</p>
            </div>
        `;
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.classList.add('fade-out');
            setTimeout(() => toast.remove(), 300);
        }, 5000);
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    // ──────────────────────────────────────────────────────────
    // 16. إضافة أنماط CSS
    // ──────────────────────────────────────────────────────────
    
    injectStyles() {
        if (document.getElementById('riva-pwa-styles')) return;
        
        const styles = `
            <style id="riva-pwa-styles">
                .pwa-install-btn {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    background: var(--primary, #1a73e8);
                    color: white;
                    border: none;
                    border-radius: 50px;
                    padding: 12px 20px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    cursor: pointer;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    z-index: 9999;
                    font-family: inherit;
                    font-size: 14px;
                    font-weight: 500;
                    transition: all 0.3s ease;
                    animation: slideUp 0.3s ease;
                }
                
                .pwa-install-btn:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 6px 16px rgba(0,0,0,0.2);
                    background: var(--primary-dark, #0d47a1);
                }
                
                .install-icon {
                    font-size: 18px;
                }
                
                @keyframes slideUp {
                    from {
                        transform: translateY(100%);
                        opacity: 0;
                    }
                    to {
                        transform: translateY(0);
                        opacity: 1;
                    }
                }
                
                @media (max-width: 768px) {
                    .pwa-install-btn {
                        bottom: 70px;
                        right: 16px;
                        padding: 10px 16px;
                        font-size: 12px;
                    }
                }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }
    
    injectModalStyles() {
        if (document.getElementById('riva-pwa-modal-styles')) return;
        
        const styles = `
            <style id="riva-pwa-modal-styles">
                .pwa-install-modal {
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    z-index: 10000;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    visibility: hidden;
                    opacity: 0;
                    transition: all 0.3s ease;
                }
                
                .pwa-install-modal.show {
                    visibility: visible;
                    opacity: 1;
                }
                
                .modal-overlay {
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(0,0,0,0.5);
                }
                
                .modal-content {
                    position: relative;
                    background: var(--white, #ffffff);
                    border-radius: 16px;
                    width: 90%;
                    max-width: 400px;
                    max-height: 90vh;
                    overflow-y: auto;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.2);
                    animation: modalSlideIn 0.3s ease;
                }
                
                @keyframes modalSlideIn {
                    from {
                        transform: translateY(-50px);
                        opacity: 0;
                    }
                    to {
                        transform: translateY(0);
                        opacity: 1;
                    }
                }
                
                .modal-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 16px 20px;
                    border-bottom: 1px solid var(--gray-lighter, #e8eaed);
                }
                
                .modal-header h3 {
                    margin: 0;
                    font-size: 18px;
                }
                
                .modal-close {
                    background: none;
                    border: none;
                    font-size: 24px;
                    cursor: pointer;
                    color: var(--gray, #5f6368);
                }
                
                .modal-body {
                    padding: 20px;
                }
                
                .install-steps {
                    margin: 16px 0;
                }
                
                .step {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    margin-bottom: 12px;
                    padding: 8px;
                    background: var(--light, #f8f9fa);
                    border-radius: 8px;
                }
                
                .step-number {
                    width: 28px;
                    height: 28px;
                    background: var(--primary, #1a73e8);
                    color: white;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    flex-shrink: 0;
                }
                
                .step-text {
                    flex: 1;
                }
                
                .install-images {
                    margin: 16px 0;
                    text-align: center;
                }
                
                .install-image {
                    max-width: 100%;
                    border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }
                
                .install-note {
                    background: rgba(251, 188, 4, 0.1);
                    padding: 12px;
                    border-radius: 8px;
                    display: flex;
                    gap: 8px;
                    margin-top: 16px;
                }
                
                .modal-footer {
                    display: flex;
                    gap: 12px;
                    padding: 16px 20px;
                    border-top: 1px solid var(--gray-lighter, #e8eaed);
                }
                
                .modal-btn {
                    flex: 1;
                    padding: 10px;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    font-family: inherit;
                    font-size: 14px;
                    font-weight: 500;
                }
                
                .modal-btn-primary {
                    background: var(--primary, #1a73e8);
                    color: white;
                }
                
                .pwa-toast {
                    position: fixed;
                    bottom: 20px;
                    left: 20px;
                    right: 20px;
                    max-width: 350px;
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    z-index: 10001;
                    animation: slideInLeft 0.3s ease;
                }
                
                .pwa-toast-success {
                    border-right: 4px solid var(--success, #34a853);
                }
                
                .pwa-toast-info {
                    border-right: 4px solid var(--info, #1a73e8);
                }
                
                .pwa-toast-warning {
                    border-right: 4px solid var(--warning, #fbbc04);
                }
                
                .toast-content {
                    padding: 12px 16px;
                }
                
                .toast-content strong {
                    display: block;
                    margin-bottom: 4px;
                }
                
                .toast-content p {
                    margin: 0;
                    font-size: 12px;
                    color: #5f6368;
                }
                
                .pwa-toast.fade-out {
                    animation: fadeOut 0.3s ease forwards;
                }
                
                @keyframes slideInLeft {
                    from {
                        transform: translateX(-100%);
                        opacity: 0;
                    }
                    to {
                        transform: translateX(0);
                        opacity: 1;
                    }
                }
                
                @keyframes fadeOut {
                    to {
                        opacity: 0;
                        transform: translateX(-100%);
                    }
                }
                
                @media (max-width: 768px) {
                    .modal-content {
                        width: 95%;
                        margin: 16px;
                    }
                    
                    .pwa-toast {
                        left: 16px;
                        right: 16px;
                        max-width: none;
                    }
                }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }
    
    // ──────────────────────────────────────────────────────────
    // 17. نظام الأحداث
    // ──────────────────────────────────────────────────────────
    
    on(event, callback) {
        if (!this.installEventListeners[event]) {
            this.installEventListeners[event] = [];
        }
        this.installEventListeners[event].push(callback);
    }
    
    off(event, callback) {
        if (!this.installEventListeners[event]) return;
        this.installEventListeners[event] = this.installEventListeners[event].filter(cb => cb !== callback);
    }
    
    emit(event, data) {
        if (!this.installEventListeners[event]) return;
        this.installEventListeners[event].forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error(`[PWAInstall] Error in event listener for ${event}:`, error);
            }
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 18. الحصول على الحالة
    // ──────────────────────────────────────────────────────────
    
    getStatus() {
        return {
            isInstalled: this.isInstalled,
            isInstallable: this.isInstallable,
            isIOS: this.isIOS,
            isStandalone: this.isStandalone,
            deferredPromptExists: this.deferredPrompt !== null,
            lastPromptDate: this.lastPromptDate,
            daysSinceLastPrompt: this.lastPromptDate ? 
                (Date.now() - this.lastPromptDate.getTime()) / (1000 * 60 * 60 * 24) : null
        };
    }
    
    // ──────────────────────────────────────────────────────────
    // 19. إعادة تعيين حالة التثبيت (للاختبار)
    // ──────────────────────────────────────────────────────────
    
    resetInstallStatus() {
        this.isInstalled = false;
        this.isStandalone = false;
        localStorage.removeItem(this.storageKey);
        localStorage.removeItem('riva_pwa_last_prompt');
        this.lastPromptDate = null;
        console.log('[PWAInstall] Install status reset');
    }
}

// ──────────────────────────────────────────────────────────
// 20. تصدير نسخة واحدة (Singleton)
// ──────────────────────────────────────────────────────────

const pwaInstall = new PWAInstallManager();

window.pwaInstall = pwaInstall;

export default pwaInstall;
