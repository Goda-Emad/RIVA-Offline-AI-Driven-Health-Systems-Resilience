/**
 * offline-detection.js
 * ====================
 * RIVA Health Platform - Offline Detection & Management
 * كشف وضع عدم الاتصال وإدارة النظام بدون إنترنت
 * 
 * المسؤوليات:
 * - مراقبة حالة الاتصال بالإنترنت (Online/Offline)
 * - عرض إشعارات للمستخدم عند قطع النت
 * - إدارة وضع Offline Mode
 * - تخزين البيانات مؤقتاً في IndexedDB
 * - مزامنة البيانات تلقائياً عند عودة النت
 * - دعم FormData وملفات للـ Offline
 * - مزامنة متوازية مع تحكم في التدفق
 * 
 * المسار: web-app/src/static/js/offline-detection.js
 * 
 * التحسينات:
 * - دعم PWA (Progressive Web App)
 * - إشعارات مرئية ومسموعة
 * - تخزين آمن للبيانات (دعم FormData)
 * - مزامنة ذكية مع الأولويات
 * - تحكم في التدفق (Rate Limiting)
 * 
 * الإصدار: 4.2.1
 */

// ──────────────────────────────────────────────────────────
// 0. تحميل أنماط CSS من ملف خارجي (لتجنب مشاكل CSP)
// ──────────────────────────────────────────────────────────

function loadOfflineStyles() {
    if (document.getElementById('riva-offline-styles-link')) return;
    
    const link = document.createElement('link');
    link.id = 'riva-offline-styles-link';
    link.rel = 'stylesheet';
    link.href = '/static/css/offline.css';
    document.head.appendChild(link);
}

// ──────────────────────────────────────────────────────────
// 1. كلاس إدارة الوضع غير المتصل
// ──────────────────────────────────────────────────────────

class OfflineManager {
    constructor() {
        this.isOnline = navigator.onLine;
        this.isOfflineMode = false;
        this.pendingRequests = [];
        this.eventListeners = {};
        this.notificationShown = false;
        this.dbName = 'RIVA_OfflineDB';
        this.dbVersion = 2; // ترقية الإصدار لدعم FormData
        this.db = null;
        this.syncInProgress = false;
        this.maxConcurrentSync = 3; // الحد الأقصى للمزامنة المتوازية
        this.syncQueue = [];
        
        this.init();
    }
    
    // ──────────────────────────────────────────────────────────
    // 2. التهيئة
    // ──────────────────────────────────────────────────────────
    
    async init() {
        console.log('[OfflineManager] Initializing...');
        
        // تحميل الأنماط من ملف خارجي
        loadOfflineStyles();
        
        // فتح قاعدة البيانات
        await this.openDatabase();
        
        // استرجاع الطلبات المعلقة من التخزين المحلي
        await this.loadPendingRequests();
        
        // إضافة مستمعي الأحداث
        window.addEventListener('online', this.handleOnline.bind(this));
        window.addEventListener('offline', this.handleOffline.bind(this));
        
        // التحقق من حالة الـ Service Worker
        if ('serviceWorker' in navigator) {
            this.registerServiceWorker();
        }
        
        console.log(`[OfflineManager] Initialized. Status: ${this.isOnline ? 'Online' : 'Offline'}`);
        
        // عرض حالة الاتصال الأولية
        this.updateUI();
    }
    
    // ──────────────────────────────────────────────────────────
    // 3. إدارة IndexedDB (مع دعم FormData)
    // ──────────────────────────────────────────────────────────
    
    openDatabase() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName, this.dbVersion);
            
            request.onerror = () => {
                console.error('[OfflineManager] Failed to open database:', request.error);
                reject(request.error);
            };
            
            request.onsuccess = () => {
                this.db = request.result;
                console.log('[OfflineManager] Database opened');
                resolve(this.db);
            };
            
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                const oldVersion = event.oldVersion;
                
                // جدول الطلبات المعلقة
                if (!db.objectStoreNames.contains('pending_requests')) {
                    const store = db.createObjectStore('pending_requests', { 
                        keyPath: 'id', 
                        autoIncrement: true 
                    });
                    store.createIndex('by_timestamp', 'timestamp');
                    store.createIndex('by_priority', 'priority');
                }
                
                // جدول البيانات المخزنة مؤقتاً
                if (!db.objectStoreNames.contains('cached_data')) {
                    const cacheStore = db.createObjectStore('cached_data', { 
                        keyPath: 'key' 
                    });
                    cacheStore.createIndex('by_expiry', 'expiry');
                }
                
                // ترقية الإصدار لدعم FormData
                if (oldVersion < 2) {
                    console.log('[OfflineManager] Upgrading database to support FormData');
                }
                
                console.log('[OfflineManager] Database upgraded');
            };
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 4. تحويل FormData للتخزين في IndexedDB
    // ──────────────────────────────────────────────────────────
    
    async serializeBody(body) {
        if (!body) return null;
        
        // إذا كان نص (JSON string)
        if (typeof body === 'string') {
            return { type: 'json', data: body };
        }
        
        // إذا كان FormData
        if (body instanceof FormData) {
            const formDataObj = {};
            for (const [key, value] of body.entries()) {
                if (value instanceof File) {
                    // تحويل الملف إلى Base64
                    formDataObj[key] = {
                        type: 'file',
                        name: value.name,
                        size: value.size,
                        type: value.type,
                        data: await this.fileToBase64(value)
                    };
                } else {
                    formDataObj[key] = { type: 'text', data: value };
                }
            }
            return { type: 'formdata', data: formDataObj };
        }
        
        // إذا كان كائن (Object)
        if (typeof body === 'object') {
            return { type: 'object', data: JSON.stringify(body) };
        }
        
        return { type: 'raw', data: body };
    }
    
    async deserializeBody(serialized) {
        if (!serialized) return null;
        
        if (serialized.type === 'json') {
            return serialized.data;
        }
        
        if (serialized.type === 'formdata') {
            const formData = new FormData();
            for (const [key, value] of Object.entries(serialized.data)) {
                if (value.type === 'file') {
                    const blob = await this.base64ToBlob(value.data, value.type);
                    const file = new File([blob], value.name, { type: value.type });
                    formData.append(key, file);
                } else {
                    formData.append(key, value.data);
                }
            }
            return formData;
        }
        
        if (serialized.type === 'object') {
            return serialized.data;
        }
        
        return serialized.data;
    }
    
    fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }
    
    base64ToBlob(base64, mimeType) {
        const parts = base64.split(',');
        const byteString = atob(parts[1]);
        const ab = new ArrayBuffer(byteString.length);
        const ia = new Uint8Array(ab);
        for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
        }
        return new Blob([ab], { type: mimeType });
    }
    
    // ──────────────────────────────────────────────────────────
    // 5. حفظ الطلبات في IndexedDB (مع دعم FormData)
    // ──────────────────────────────────────────────────────────
    
    async savePendingRequest(request) {
        if (!this.db) return;
        
        // تحويل body للتخزين
        const serializedBody = await this.serializeBody(request.body);
        
        const pendingRequest = {
            id: Date.now() + Math.random(),
            url: request.url,
            method: request.method || 'POST',
            headers: request.headers || {},
            body: serializedBody,
            timestamp: Date.now(),
            priority: request.priority || 1,
            retryCount: 0,
            maxRetries: request.maxRetries || 3
        };
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction(['pending_requests'], 'readwrite');
            const store = transaction.objectStore('pending_requests');
            
            const addRequest = store.add(pendingRequest);
            
            addRequest.onsuccess = () => {
                this.pendingRequests.push(pendingRequest);
                console.log(`[OfflineManager] Saved pending request: ${request.url}`);
                resolve(addRequest.result);
            };
            
            addRequest.onerror = () => {
                console.error('[OfflineManager] Failed to save pending request:', addRequest.error);
                reject(addRequest.error);
            };
        });
    }
    
    async loadPendingRequests() {
        if (!this.db) return;
        
        return new Promise((resolve) => {
            const transaction = this.db.transaction(['pending_requests'], 'readonly');
            const store = transaction.objectStore('pending_requests');
            const requests = [];
            
            const cursor = store.openCursor();
            
            cursor.onsuccess = async (event) => {
                const cursor = event.target.result;
                if (cursor) {
                    requests.push(cursor.value);
                    cursor.continue();
                } else {
                    this.pendingRequests = requests;
                    console.log(`[OfflineManager] Loaded ${requests.length} pending requests`);
                    resolve(requests);
                }
            };
            
            cursor.onerror = () => {
                console.error('[OfflineManager] Failed to load pending requests');
                resolve([]);
            };
        });
    }
    
    async removePendingRequest(id) {
        if (!this.db) return;
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction(['pending_requests'], 'readwrite');
            const store = transaction.objectStore('pending_requests');
            
            const deleteRequest = store.delete(id);
            
            deleteRequest.onsuccess = () => {
                this.pendingRequests = this.pendingRequests.filter(r => r.id !== id);
                resolve(true);
            };
            
            deleteRequest.onerror = () => {
                reject(deleteRequest.error);
            };
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 6. تخزين البيانات مؤقتاً (Cache)
    // ──────────────────────────────────────────────────────────
    
    async cacheData(key, data, ttl = 3600000) {
        if (!this.db) return false;
        
        const expiry = Date.now() + ttl;
        
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction(['cached_data'], 'readwrite');
            const store = transaction.objectStore('cached_data');
            
            const putRequest = store.put({ key, data, expiry });
            
            putRequest.onsuccess = () => {
                console.log(`[OfflineManager] Cached data: ${key}`);
                resolve(true);
            };
            
            putRequest.onerror = () => {
                console.error('[OfflineManager] Failed to cache data:', putRequest.error);
                reject(false);
            };
        });
    }
    
    async getCachedData(key) {
        if (!this.db) return null;
        
        return new Promise((resolve) => {
            const transaction = this.db.transaction(['cached_data'], 'readonly');
            const store = transaction.objectStore('cached_data');
            
            const getRequest = store.get(key);
            
            getRequest.onsuccess = () => {
                const cached = getRequest.result;
                if (cached && cached.expiry > Date.now()) {
                    resolve(cached.data);
                } else {
                    if (cached) {
                        this.db.transaction(['cached_data'], 'readwrite')
                            .objectStore('cached_data')
                            .delete(key);
                    }
                    resolve(null);
                }
            };
            
            getRequest.onerror = () => {
                resolve(null);
            };
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 7. إدارة الطلبات في وضع Offline
    // ──────────────────────────────────────────────────────────
    
    async queueRequest(url, options = {}) {
        if (this.isOnline) {
            return this.sendRequest(url, options);
        } else {
            console.log(`[OfflineManager] Queuing request for later: ${url}`);
            
            const pendingRequest = {
                url,
                method: options.method || 'POST',
                headers: options.headers || {},
                body: options.body,
                priority: options.priority || 1,
                maxRetries: options.maxRetries || 3
            };
            
            await this.savePendingRequest(pendingRequest);
            
            this.showNotification('الطلب محفوظ', 'سيتم إرسال الطلب تلقائياً عند عودة الاتصال');
            
            return {
                success: false,
                offline: true,
                queued: true,
                message: 'الطلب تم حفظه وسيتم إرساله عند عودة الاتصال'
            };
        }
    }
    
    async sendRequest(url, options) {
        try {
            // إعادة بناء الـ body إذا كان مخزناً
            if (options.body && options.body.type) {
                options.body = await this.deserializeBody(options.body);
            }
            
            const response = await fetch(url, options);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('[OfflineManager] Request failed:', error);
            throw error;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 8. مزامنة الطلبات المعلقة (مع تحكم في التدفق)
    // ──────────────────────────────────────────────────────────
    
    async syncPendingRequests() {
        if (!this.isOnline) {
            console.log('[OfflineManager] Cannot sync while offline');
            return { synced: 0, failed: 0 };
        }
        
        if (this.syncInProgress) {
            console.log('[OfflineManager] Sync already in progress');
            return { synced: 0, failed: 0, pending: true };
        }
        
        if (this.pendingRequests.length === 0) {
            console.log('[OfflineManager] No pending requests to sync');
            return { synced: 0, failed: 0 };
        }
        
        console.log(`[OfflineManager] Syncing ${this.pendingRequests.length} pending requests...`);
        
        this.syncInProgress = true;
        
        // ترتيب حسب الأولوية (الأعلى أولاً)
        const sortedRequests = [...this.pendingRequests].sort((a, b) => b.priority - a.priority);
        
        let synced = 0;
        let failed = 0;
        
        // مزامنة متوازية مع تحكم في التدفق
        const chunks = [];
        for (let i = 0; i < sortedRequests.length; i += this.maxConcurrentSync) {
            chunks.push(sortedRequests.slice(i, i + this.maxConcurrentSync));
        }
        
        for (const chunk of chunks) {
            const results = await Promise.allSettled(
                chunk.map(request => this.syncSingleRequest(request))
            );
            
            for (const result of results) {
                if (result.status === 'fulfilled' && result.value.success) {
                    synced++;
                } else {
                    failed++;
                }
            }
        }
        
        console.log(`[OfflineManager] Sync completed: ${synced} synced, ${failed} failed`);
        
        this.syncInProgress = false;
        
        // إشعار للمستخدم
        if (synced > 0) {
            this.showNotification('تمت المزامنة', `تم إرسال ${synced} طلب بنجاح`, 'success');
        }
        
        return { synced, failed };
    }
    
    async syncSingleRequest(request) {
        try {
            const options = {
                method: request.method,
                headers: request.headers,
                body: request.body
            };
            
            await this.sendRequest(request.url, options);
            await this.removePendingRequest(request.id);
            
            console.log(`[OfflineManager] Synced: ${request.url}`);
            return { success: true };
            
        } catch (error) {
            console.warn(`[OfflineManager] Failed to sync: ${request.url}`, error);
            
            request.retryCount++;
            
            if (request.retryCount >= request.maxRetries) {
                console.warn(`[OfflineManager] Request failed permanently: ${request.url}`);
                await this.removePendingRequest(request.id);
            } else {
                await this.savePendingRequest(request);
            }
            
            return { success: false, error: error.message };
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 9. معالجة الأحداث (Online/Offline)
    // ──────────────────────────────────────────────────────────
    
    handleOnline() {
        console.log('[OfflineManager] Network is back online');
        this.isOnline = true;
        this.isOfflineMode = false;
        this.notificationShown = false;
        
        this.hideOfflineBanner();
        
        // محاولة مزامنة الطلبات المعلقة (غير متزامن)
        setTimeout(() => this.syncPendingRequests(), 100);
        
        this.showNotification('تم الاتصال بالإنترنت', 'جميع الخدمات متاحة الآن', 'success');
        this.updateUI();
        this.emit('online');
    }
    
    handleOffline() {
        console.log('[OfflineManager] Network is offline');
        this.isOnline = false;
        this.isOfflineMode = true;
        
        if (!this.notificationShown) {
            this.showOfflineBanner();
            this.showNotification('لا يوجد اتصال بالإنترنت', 'سيتم حفظ البيانات وإرسالها تلقائياً عند عودة الاتصال', 'warning');
            this.notificationShown = true;
        }
        
        this.updateUI();
        this.emit('offline');
    }
    
    // ──────────────────────────────────────────────────────────
    // 10. واجهة المستخدم (UI)
    // ──────────────────────────────────────────────────────────
    
    showOfflineBanner() {
        this.hideOfflineBanner();
        
        const banner = document.createElement('div');
        banner.id = 'offline-banner';
        banner.className = 'offline-banner';
        banner.innerHTML = `
            <div class="offline-banner-content">
                <span class="offline-icon">📡</span>
                <span class="offline-text">لا يوجد اتصال بالإنترنت - النظام يعمل في وضع عدم الاتصال</span>
                <span class="offline-saved">✅ سيتم حفظ البيانات محلياً</span>
                <button class="offline-close" id="offline-close">✕</button>
            </div>
        `;
        
        document.body.insertBefore(banner, document.body.firstChild);
        
        const closeBtn = document.getElementById('offline-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.hideOfflineBanner());
        }
    }
    
    hideOfflineBanner() {
        const banner = document.getElementById('offline-banner');
        if (banner) banner.remove();
    }
    
    showNotification(message, details, type = 'info') {
        if ('Notification' in window && Notification.permission === 'granted') {
            new Notification(message, { body: details, icon: '/static/assets/logo.png' });
        }
        
        const toast = document.createElement('div');
        toast.className = `offline-toast offline-toast-${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <strong>${this.escapeHtml(message)}</strong>
                <p>${this.escapeHtml(details)}</p>
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
    
    updateUI() {
        if (this.isOnline) {
            document.body.classList.remove('offline-mode');
            document.body.classList.add('online-mode');
        } else {
            document.body.classList.remove('online-mode');
            document.body.classList.add('offline-mode');
        }
        
        const statusIndicators = document.querySelectorAll('.connection-status');
        statusIndicators.forEach(el => {
            el.textContent = this.isOnline ? '🟢 متصل' : '🔴 غير متصل';
            el.classList.toggle('online', this.isOnline);
            el.classList.toggle('offline', !this.isOnline);
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 11. Service Worker Registration (PWA)
    // ──────────────────────────────────────────────────────────
    
    async registerServiceWorker() {
        try {
            const registration = await navigator.serviceWorker.register('/sw.js');
            console.log('[OfflineManager] Service Worker registered:', registration);
            
            if ('Notification' in window && Notification.permission === 'default') {
                Notification.requestPermission();
            }
        } catch (error) {
            console.error('[OfflineManager] Service Worker registration failed:', error);
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 12. نظام الأحداث (Event System)
    // ──────────────────────────────────────────────────────────
    
    on(event, callback) {
        if (!this.eventListeners[event]) this.eventListeners[event] = [];
        this.eventListeners[event].push(callback);
    }
    
    off(event, callback) {
        if (!this.eventListeners[event]) return;
        this.eventListeners[event] = this.eventListeners[event].filter(cb => cb !== callback);
    }
    
    emit(event, data) {
        if (!this.eventListeners[event]) return;
        this.eventListeners[event].forEach(callback => {
            try { callback(data); } catch (error) {
                console.error(`[OfflineManager] Error in event listener for ${event}:`, error);
            }
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 13. الحصول على الحالة
    // ──────────────────────────────────────────────────────────
    
    getStatus() {
        return {
            isOnline: this.isOnline,
            isOfflineMode: this.isOfflineMode,
            pendingRequests: this.pendingRequests.length,
            syncInProgress: this.syncInProgress,
            maxConcurrentSync: this.maxConcurrentSync
        };
    }
    
    resetNotifications() {
        this.notificationShown = false;
    }
    
    // ──────────────────────────────────────────────────────────
    // 14. تعيين الحد الأقصى للمزامنة المتوازية
    // ──────────────────────────────────────────────────────────
    
    setMaxConcurrentSync(max) {
        if (max > 0 && max <= 10) {
            this.maxConcurrentSync = max;
            console.log(`[OfflineManager] Max concurrent sync set to ${max}`);
        }
    }
}

// ──────────────────────────────────────────────────────────
// 15. تصدير نسخة واحدة (Singleton)
// ──────────────────────────────────────────────────────────

const offlineManager = new OfflineManager();

window.offlineManager = offlineManager;

export default offlineManager;
