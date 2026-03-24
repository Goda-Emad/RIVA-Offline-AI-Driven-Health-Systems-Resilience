/**
 * main.js
 * =======
 * RIVA Health Platform - Main Entry Point
 * نقطة الدخول الرئيسية للتطبيق
 * 
 * المسؤوليات:
 * - تهيئة جميع وحدات التطبيق
 * - إدارة توجيه المستخدمين حسب الدور
 * - التحقق من الجلسة والمصادقة
 * - تطبيق الإعدادات (RTL، الثيم، اللغة)
 * 
 * الإصدار: 4.2.1
 */

// ──────────────────────────────────────────────────────────
// 1. تهيئة التطبيق
// ──────────────────────────────────────────────────────────

class RIVAMain {
    constructor() {
        this.apiClient = null;
        this.auth = null;
        this.offlineManager = null;
        this.pwaInstall = null;
        this.isInitialized = false;
        this.currentPage = window.location.pathname.split('/').pop() || '01_home.html';
        
        this.init();
    }
    
    async init() {
        console.log('[RIVA] Initializing application...');
        
        // انتظار تحميل DOM
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setupModules());
        } else {
            await this.setupModules();
        }
    }
    
    async setupModules() {
        console.log('[RIVA] Setting up modules...');
        
        // الحصول على الوحدات من window
        this.apiClient = window.rivaAPI;
        this.auth = window.rivaAuth;
        this.offlineManager = window.rivaOfflineManager;
        this.pwaInstall = window.rivaPWAInstall;
        
        // انتظار جاهزية الوحدات
        await this.waitForModules();
        
        // تطبيق الإعدادات
        this.applySettings();
        
        // التحقق من المصادقة
        await this.checkAuth();
        
        // تطبيق اللغة والاتجاه
        this.applyLanguage();
        
        // تهيئة الـ Service Worker
        this.initServiceWorker();
        
        this.isInitialized = true;
        console.log('[RIVA] Application initialized successfully');
        
        // إطلاق حدث الجاهزية
        window.dispatchEvent(new CustomEvent('riva-ready', {
            detail: {
                apiClient: this.apiClient,
                auth: this.auth,
                offlineManager: this.offlineManager
            }
        }));
    }
    
    async waitForModules() {
        // انتظار وجود الوحدات (بحد أقصى 5 ثوانٍ)
        let attempts = 0;
        const maxAttempts = 50; // 5 ثوانٍ
        
        while (attempts < maxAttempts) {
            if (this.apiClient && this.auth && this.offlineManager) {
                console.log('[RIVA] All modules loaded');
                return;
            }
            
            // تحديث المراجع
            this.apiClient = this.apiClient || window.rivaAPI;
            this.auth = this.auth || window.rivaAuth;
            this.offlineManager = this.offlineManager || window.rivaOfflineManager;
            this.pwaInstall = this.pwaInstall || window.rivaPWAInstall;
            
            await this.sleep(100);
            attempts++;
        }
        
        console.warn('[RIVA] Some modules not loaded after timeout');
    }
    
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    // ──────────────────────────────────────────────────────────
    // 2. التحقق من المصادقة
    // ──────────────────────────────────────────────────────────
    
    async checkAuth() {
        // الصفحات العامة (لا تحتاج مصادقة)
        const publicPages = ['login.html', '08_offline.html', 'offline-fallback.html'];
        
        if (publicPages.includes(this.currentPage)) {
            console.log('[RIVA] Public page, skipping auth check');
            return true;
        }
        
        // التحقق من وجود جلسة
        const isAuthenticated = this.auth?.isAuthenticated?.() || false;
        
        if (!isAuthenticated) {
            console.log('[RIVA] User not authenticated, redirecting to login');
            window.location.href = '/login.html';
            return false;
        }
        
        // التحقق من صلاحيات الصفحة
        const user = this.auth.getCurrentUser();
        const role = user?.role;
        
        console.log(`[RIVA] User authenticated: ${user?.username} (${role})`);
        
        return true;
    }
    
    // ──────────────────────────────────────────────────────────
    // 3. تطبيق الإعدادات
    // ──────────────────────────────────────────────────────────
    
    applySettings() {
        // تطبيق الثيم (فاتح/داكن)
        const theme = localStorage.getItem('riva_theme') || 'light';
        this.applyTheme(theme);
        
        // تطبيق اللغة
        const language = localStorage.getItem('riva_language') || 'ar';
        this.applyLanguage(language);
        
        // تطبيق RTL
        const isRTL = language === 'ar';
        this.applyRTL(isRTL);
    }
    
    applyTheme(theme) {
        if (theme === 'dark') {
            document.body.classList.add('dark-theme');
        } else {
            document.body.classList.remove('dark-theme');
        }
    }
    
    applyLanguage(language = 'ar') {
        document.documentElement.lang = language;
        document.documentElement.setAttribute('lang', language);
        
        // تغيير اتجاه الصفحة
        const isRTL = language === 'ar';
        this.applyRTL(isRTL);
        
        // إطلاق حدث تغيير اللغة
        window.dispatchEvent(new CustomEvent('riva-language-change', {
            detail: { language, isRTL }
        }));
    }
    
    applyRTL(isRTL) {
        if (isRTL) {
            document.documentElement.dir = 'rtl';
            document.body.classList.add('rtl');
            document.body.classList.remove('ltr');
        } else {
            document.documentElement.dir = 'ltr';
            document.body.classList.add('ltr');
            document.body.classList.remove('rtl');
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 4. Service Worker
    // ──────────────────────────────────────────────────────────
    
    initServiceWorker() {
        if ('serviceWorker' in navigator && this.offlineManager) {
            navigator.serviceWorker.register('/service-worker.js')
                .then(registration => {
                    console.log('[RIVA] Service Worker registered:', registration);
                })
                .catch(error => {
                    console.error('[RIVA] Service Worker registration failed:', error);
                });
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 5. أدوات مساعدة
    // ──────────────────────────────────────────────────────────
    
    getCurrentUser() {
        return this.auth?.getCurrentUser?.() || null;
    }
    
    getUserRole() {
        return this.auth?.getCurrentRole?.() || null;
    }
    
    isAuthenticated() {
        return this.auth?.isAuthenticated?.() || false;
    }
    
    // ──────────────────────────────────────────────────────────
    // 6. تحديث واجهة المستخدم حسب الدور
    // ──────────────────────────────────────────────────────────
    
    updateUIByRole() {
        const user = this.getCurrentUser();
        const role = user?.role;
        
        if (!role) return;
        
        // إضافة كلاس للـ body حسب الدور
        document.body.classList.add(`role-${role}`);
        
        // إظهار/إخفاء عناصر حسب الدور
        const elementsByRole = document.querySelectorAll('[data-role]');
        elementsByRole.forEach(el => {
            const allowedRoles = el.getAttribute('data-role').split(',');
            if (allowedRoles.includes(role)) {
                el.style.display = '';
            } else {
                el.style.display = 'none';
            }
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 7. تسجيل الخروج
    // ──────────────────────────────────────────────────────────
    
    async logout(redirectTo = '/login.html') {
        if (this.auth) {
            await this.auth.logout(redirectTo);
        } else {
            window.location.href = redirectTo;
        }
    }
}

// ──────────────────────────────────────────────────────────
// 8. إنشاء نسخة واحدة وتخزينها في window
// ──────────────────────────────────────────────────────────

const rivaMain = new RIVAMain();

// تخزين في window للاستخدام العادي
window.rivaMain = rivaMain;
window.RIVA = rivaMain;

// ES Module export
export default rivaMain;
export { rivaMain };
