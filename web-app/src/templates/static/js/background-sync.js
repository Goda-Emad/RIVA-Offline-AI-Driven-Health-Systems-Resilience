/**
 * background-sync.js
 * ==================
 * RIVA Health Platform - Background Sync Manager
 * مدير المزامنة الخلفية للتطبيق
 * 
 * المسؤوليات:
 * - مزامنة البيانات عند عودة الاتصال
 * - إدارة قائمة الانتظار للطلبات المعلقة
 * - دعم Background Sync API (إذا كان متاحاً)
 * - المزامنة التلقائية مع OfflineManager
 * 
 * الإصدار: 4.2.1
 */

class BackgroundSyncManager {
    constructor() {
        this.isSupported = 'sync' in navigator.serviceWorker || 'periodicSync' in navigator.serviceWorker;
        this.isSyncing = false;
        this.syncQueue = [];
        this.eventListeners = {};
        
        this.init();
    }
    
    async init() {
        console.log('[BackgroundSync] Initializing...', { isSupported: this.isSupported });
        
        if ('serviceWorker' in navigator && 'SyncManager' in window) {
            await this.registerSync();
        }
        
        // الاستماع لأحداث العودة للاتصال
        window.addEventListener('online', this.handleOnline.bind(this));
        
        // استخدام OfflineManager إذا كان متاحاً
        if (window.rivaOfflineManager) {
            this.offlineManager = window.rivaOfflineManager;
            console.log('[BackgroundSync] Connected to OfflineManager');
        }
        
        console.log('[BackgroundSync] Initialized');
    }
    
    async registerSync() {
        try {
            const registration = await navigator.serviceWorker.ready;
            
            // تسجيل مزامنة خلفية
            if ('sync' in registration) {
                await registration.sync.register('riva-sync');
                console.log('[BackgroundSync] Background sync registered');
            }
            
            // تسجيل مزامنة دورية (إذا كانت مدعومة)
            if ('periodicSync' in registration) {
                const status = await navigator.permissions.query({
                    name: 'periodic-background-sync',
                });
                
                if (status.state === 'granted') {
                    await registration.periodicSync.register('riva-periodic-sync', {
                        minInterval: 24 * 60 * 60 * 1000 // مرة كل 24 ساعة
                    });
                    console.log('[BackgroundSync] Periodic sync registered');
                }
            }
        } catch (error) {
            console.warn('[BackgroundSync] Failed to register sync:', error);
        }
    }
    
    async handleOnline() {
        console.log('[BackgroundSync] Network online, starting sync...');
        
        if (this.isSyncing) {
            console.log('[BackgroundSync] Sync already in progress');
            return;
        }
        
        await this.startSync();
    }
    
    async startSync() {
        if (this.isSyncing) return;
        
        this.isSyncing = true;
        console.log('[BackgroundSync] Starting sync process');
        
        try {
            // استخدام OfflineManager للمزامنة إذا كان متاحاً
            if (this.offlineManager && typeof this.offlineManager.syncPendingRequests === 'function') {
                const result = await this.offlineManager.syncPendingRequests();
                console.log('[BackgroundSync] OfflineManager sync result:', result);
                this.emit('sync-complete', result);
            } else {
                // مزامنة يدوية
                await this.syncManually();
            }
        } catch (error) {
            console.error('[BackgroundSync] Sync failed:', error);
            this.emit('sync-error', error);
        } finally {
            this.isSyncing = false;
        }
    }
    
    async syncManually() {
        // استرجاع الطلبات المعلقة من localStorage
        const pendingRequests = this.getPendingRequests();
        
        if (pendingRequests.length === 0) {
            console.log('[BackgroundSync] No pending requests');
            return { synced: 0, failed: 0 };
        }
        
        let synced = 0;
        let failed = 0;
        
        for (const request of pendingRequests) {
            try {
                await this.sendRequest(request);
                this.removePendingRequest(request.id);
                synced++;
            } catch (error) {
                console.warn('[BackgroundSync] Failed to sync request:', request.url, error);
                failed++;
            }
        }
        
        console.log(`[BackgroundSync] Manual sync completed: ${synced} synced, ${failed} failed`);
        
        return { synced, failed };
    }
    
    getPendingRequests() {
        try {
            const requests = localStorage.getItem('riva_pending_requests');
            return requests ? JSON.parse(requests) : [];
        } catch {
            return [];
        }
    }
    
    savePendingRequests(requests) {
        localStorage.setItem('riva_pending_requests', JSON.stringify(requests));
    }
    
    addPendingRequest(request) {
        const requests = this.getPendingRequests();
        request.id = Date.now() + Math.random();
        request.timestamp = Date.now();
        requests.push(request);
        this.savePendingRequests(requests);
        
        console.log('[BackgroundSync] Request queued:', request.url);
    }
    
    removePendingRequest(id) {
        const requests = this.getPendingRequests();
        const filtered = requests.filter(r => r.id !== id);
        this.savePendingRequests(filtered);
    }
    
    async sendRequest(request) {
        const options = {
            method: request.method || 'POST',
            headers: request.headers || {},
            body: request.body
        };
        
        const response = await fetch(request.url, options);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        return await response.json();
    }
    
    async queueRequest(url, options = {}) {
        const request = {
            url,
            method: options.method || 'POST',
            headers: options.headers || {},
            body: options.body,
            priority: options.priority || 1,
            maxRetries: options.maxRetries || 3,
            retryCount: 0
        };
        
        this.addPendingRequest(request);
        
        // إذا كان هناك اتصال، حاول المزامنة فوراً
        if (navigator.onLine) {
            setTimeout(() => this.startSync(), 100);
        }
        
        return {
            queued: true,
            message: 'Request queued for background sync'
        };
    }
    
    getStatus() {
        const pending = this.getPendingRequests();
        
        return {
            isSupported: this.isSupported,
            isSyncing: this.isSyncing,
            pendingRequests: pending.length,
            offlineManagerAvailable: !!this.offlineManager
        };
    }
    
    // نظام الأحداث
    on(event, callback) {
        if (!this.eventListeners[event]) {
            this.eventListeners[event] = [];
        }
        this.eventListeners[event].push(callback);
    }
    
    off(event, callback) {
        if (!this.eventListeners[event]) return;
        this.eventListeners[event] = this.eventListeners[event].filter(cb => cb !== callback);
    }
    
    emit(event, data) {
        if (!this.eventListeners[event]) return;
        this.eventListeners[event].forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error(`[BackgroundSync] Error in event listener for ${event}:`, error);
            }
        });
    }
}

// إنشاء نسخة واحدة
const backgroundSync = new BackgroundSyncManager();

// تخزين في window
window.backgroundSync = backgroundSync;
window.rivaBackgroundSync = backgroundSync;

// ES Module export
export default backgroundSync;
export { backgroundSync };
