/**
 * sw.js
 * =====
 * RIVA Health Platform - Service Worker
 * عامل الخدمة للتشغيل بدون إنترنت (PWA)
 * 
 * المسؤوليات:
 * - تخزين الملفات الأساسية في Cache (HTML, CSS, JS, Assets)
 * - تمكين التشغيل بدون إنترنت (Offline Mode)
 * - مزامنة الطلبات المعلقة بعد عودة الاتصال
 * - إدارة الإشعارات (Push Notifications)
 * 
 * المسار: web-app/sw.js (جذر المشروع)
 * 
 * التحسينات:
 * - استراتيجية Cache First للملفات الثابتة
 * - Network First للـ API
 * - تحديث تلقائي للـ Cache
 * - دعم PWA للتثبيت
 * - معالجة موحدة لطلبات التنقل (Navigation)
 * 
 * الإصدار: 4.2.1
 */

const CACHE_VERSION = 'v4.2.1';
const CACHE_NAME = `riva-${CACHE_VERSION}`;
const API_CACHE = 'riva-api';

// الملفات الأساسية التي سيتم تخزينها
const STATIC_ASSETS = [
    '/',
    '/index.html',
    '/01_home.html',
    '/02_chatbot.html',
    '/03_triage.html',
    '/04_result.html',
    '/05_history.html',
    '/06_pregnancy.html',
    '/07_school.html',
    '/08_offline.html',
    '/09_doctor_dashboard.html',
    '/10_mother_dashboard.html',
    '/11_school_dashboard.html',
    '/12_ai_explanation.html',
    '/13_readmission.html',
    '/14_los_dashboard.html',
    '/15_combined_dashboard.html',
    '/16_doctor_notes.html',
    '/17_sustainability.html',
    '/login.html',
    '/register.html',
    '/offline.html'
];

// ملفات CSS
const CSS_ASSETS = [
    '/static/css/style.css',
    '/static/css/rtl.css',
    '/static/css/mobile.css',
    '/static/css/dashboard.css',
    '/static/css/prediction_dashboard.css'
];

// ملفات JS الأساسية
const JS_ASSETS = [
    '/static/js/api_client.js',
    '/static/js/auth.js',
    '/static/js/main.js',
    '/static/js/offline-detection.js',
    '/static/js/background-sync.js',
    '/static/js/pwa-install.js',
    '/static/js/chat_ui.js',
    '/static/js/voice_recorder.js',
    '/static/js/prediction_client.js',
    '/static/js/readmission_chart.js',
    '/static/js/los_chart.js',
    '/static/js/sentiment_display.js',
    '/static/js/timeline_viewer.js',
    '/static/js/qr_handler.js',
    '/static/js/interaction_checker.js',
    '/static/js/explainability.js',
    '/static/js/explanation_viewer.js',
    '/static/js/charts.js'
];

// ملفات الوسائط (Assets)
const ASSET_FILES = [
    '/static/assets/logo.png',
    '/static/assets/favicon.ico',
    '/static/assets/audio/emergency-alert.mp3',
    '/static/assets/audio/welcome.mp3',
    '/static/assets/images/chatbot-icon.svg',
    '/static/assets/images/doctor-avatar.svg',
    '/static/assets/images/patient-avatar.svg',
    '/static/assets/images/mother-avatar.svg',
    '/static/assets/images/school-avatar.svg'
];

// تجميع جميع الملفات
const ALL_ASSETS = [
    ...STATIC_ASSETS,
    ...CSS_ASSETS,
    ...JS_ASSETS,
    ...ASSET_FILES
];

// ──────────────────────────────────────────────────────────
// 1. تثبيت Service Worker
// ──────────────────────────────────────────────────────────

self.addEventListener('install', (event) => {
    console.log('[SW] Installing new version:', CACHE_VERSION);
    
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('[SW] Caching static assets...');
                return cache.addAll(ALL_ASSETS);
            })
            .then(() => {
                console.log('[SW] Installation complete');
                return self.skipWaiting();
            })
            .catch((error) => {
                console.error('[SW] Installation failed:', error);
            })
    );
});

// ──────────────────────────────────────────────────────────
// 2. تفعيل Service Worker (تنظيف الـ Cache القديم)
// ──────────────────────────────────────────────────────────

self.addEventListener('activate', (event) => {
    console.log('[SW] Activating new version:', CACHE_VERSION);
    
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cacheName) => {
                    if (cacheName !== CACHE_NAME && cacheName !== API_CACHE) {
                        console.log('[SW] Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        }).then(() => {
            console.log('[SW] Taking control of clients');
            return self.clients.claim();
        })
    );
});

// ──────────────────────────────────────────────────────────
// 3. اعتراض الطلبات (Fetch) - نقطة واحدة موحدة
// ──────────────────────────────────────────────────────────

self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);
    const request = event.request;
    
    // تجاهل طلبات التحليلات والإحصائيات
    if (url.pathname.includes('analytics') || url.pathname.includes('gtag')) {
        return event.respondWith(fetch(request));
    }
    
    // معالجة طلبات الـ API
    if (url.pathname.startsWith('/api/')) {
        return event.respondWith(handleAPIRequest(request));
    }
    
    // معالجة الملفات الثابتة (بما في ذلك صفحات HTML)
    return event.respondWith(handleStaticRequest(request));
});

// ──────────────────────────────────────────────────────────
// 4. معالجة طلبات الملفات الثابتة (مع دعم Offline Page)
// ──────────────────────────────────────────────────────────

async function handleStaticRequest(request) {
    const cache = await caches.open(CACHE_NAME);
    const cachedResponse = await cache.match(request);
    
    // ✅ معالجة طلبات التنقل (Navigation) بشكل خاص
    const isNavigation = request.mode === 'navigate';
    
    if (cachedResponse) {
        // إرجاع الملف من الـ Cache مع تحديث في الخلفية
        fetch(request).then((networkResponse) => {
            if (networkResponse && networkResponse.status === 200) {
                cache.put(request, networkResponse.clone());
            }
        }).catch(() => {});
        
        return cachedResponse;
    }
    
    // إذا لم يكن في الـ Cache، حاول من الشبكة
    try {
        const networkResponse = await fetch(request);
        if (networkResponse && networkResponse.status === 200) {
            cache.put(request, networkResponse.clone());
        }
        return networkResponse;
    } catch (error) {
        // ✅ إذا كان طلب صفحة HTML (Navigation)، عرض صفحة Offline
        if (isNavigation) {
            console.log('[SW] Serving offline page for:', request.url);
            return caches.match('/offline.html');
        }
        
        // للملفات الأخرى (صور، إلخ) إرجاع خطأ
        return new Response('Network error', {
            status: 408,
            statusText: 'Network Error',
            headers: { 'Content-Type': 'text/plain' }
        });
    }
}

// ──────────────────────────────────────────────────────────
// 5. معالجة طلبات الـ API (Network First)
// ──────────────────────────────────────────────────────────

async function handleAPIRequest(request) {
    const url = new URL(request.url);
    
    // طلبات POST لا يتم تخزينها
    if (request.method !== 'GET') {
        return fetch(request).catch((error) => {
            console.error('[SW] API request failed:', url.pathname, error);
            
            // إرجاع استجابة فشل
            return new Response(JSON.stringify({
                success: false,
                offline: true,
                error: 'No internet connection',
                queued: true
            }), {
                status: 503,
                headers: { 'Content-Type': 'application/json' }
            });
        });
    }
    
    // طلبات GET - محاولة الشبكة أولاً
    const cache = await caches.open(API_CACHE);
    
    try {
        const networkResponse = await fetch(request);
        
        // تخزين الاستجابة الناجحة في الـ Cache
        if (networkResponse && networkResponse.status === 200) {
            cache.put(request, networkResponse.clone());
        }
        
        return networkResponse;
        
    } catch (error) {
        // فشل الشبكة - محاولة الـ Cache
        const cachedResponse = await cache.match(request);
        
        if (cachedResponse) {
            console.log('[SW] Serving API from cache:', url.pathname);
            return cachedResponse;
        }
        
        // لا يوجد Cache
        return new Response(JSON.stringify({
            success: false,
            offline: true,
            error: 'No cached data available'
        }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

// ──────────────────────────────────────────────────────────
// 6. مزامنة الخلفية (Background Sync)
// ──────────────────────────────────────────────────────────

self.addEventListener('sync', (event) => {
    console.log('[SW] Background sync event:', event.tag);
    
    if (event.tag === 'sync-pending-requests') {
        event.waitUntil(syncPendingRequests());
    }
});

async function syncPendingRequests() {
    console.log('[SW] Syncing pending requests...');
    
    const db = await openDatabase();
    const pendingRequests = await getAllPendingRequests(db);
    
    let synced = 0;
    let failed = 0;
    
    for (const request of pendingRequests) {
        try {
            const response = await fetch(request.url, {
                method: request.method,
                headers: request.headers,
                body: request.body
            });
            
            if (response.ok) {
                await deletePendingRequest(db, request.id);
                synced++;
                console.log('[SW] Synced:', request.url);
            } else {
                failed++;
            }
        } catch (error) {
            failed++;
            console.error('[SW] Sync failed:', request.url, error);
        }
    }
    
    console.log(`[SW] Sync completed: ${synced} synced, ${failed} failed`);
    
    if (synced > 0) {
        await showNotification('تمت المزامنة', `تم إرسال ${synced} طلب بنجاح`);
    }
    
    return { synced, failed };
}

// ──────────────────────────────────────────────────────────
// 7. إدارة قاعدة البيانات (IndexedDB)
// ──────────────────────────────────────────────────────────

function openDatabase() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('RIVA_OfflineDB', 2);
        
        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);
        
        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            
            if (!db.objectStoreNames.contains('pending_requests')) {
                const store = db.createObjectStore('pending_requests', { 
                    keyPath: 'id', 
                    autoIncrement: true 
                });
                store.createIndex('by_timestamp', 'timestamp');
            }
        };
    });
}

function getAllPendingRequests(db) {
    return new Promise((resolve, reject) => {
        const transaction = db.transaction(['pending_requests'], 'readonly');
        const store = transaction.objectStore('pending_requests');
        const requests = [];
        
        const cursor = store.openCursor();
        
        cursor.onsuccess = (event) => {
            const cursor = event.target.result;
            if (cursor) {
                requests.push(cursor.value);
                cursor.continue();
            } else {
                resolve(requests);
            }
        };
        
        cursor.onerror = () => reject(cursor.error);
    });
}

function deletePendingRequest(db, id) {
    return new Promise((resolve, reject) => {
        const transaction = db.transaction(['pending_requests'], 'readwrite');
        const store = transaction.objectStore('pending_requests');
        
        const request = store.delete(id);
        request.onsuccess = () => resolve(true);
        request.onerror = () => reject(request.error);
    });
}

// ──────────────────────────────────────────────────────────
// 8. إدارة الإشعارات (Push Notifications)
// ──────────────────────────────────────────────────────────

self.addEventListener('push', (event) => {
    console.log('[SW] Push notification received');
    
    let data = {};
    
    if (event.data) {
        try {
            data = event.data.json();
        } catch {
            data = { title: 'RIVA', body: event.data.text() };
        }
    }
    
    const options = {
        body: data.body || 'لديك إشعار جديد من RIVA',
        icon: '/static/assets/images/logo.png',
        badge: '/static/assets/images/logo.png',
        vibrate: [200, 100, 200],
        data: {
            url: data.url || '/',
            dateOfArrival: Date.now()
        },
        actions: [
            {
                action: 'open',
                title: 'فتح التطبيق'
            },
            {
                action: 'dismiss',
                title: 'إغلاق'
            }
        ]
    };
    
    event.waitUntil(
        self.registration.showNotification(data.title || 'RIVA', options)
    );
});

self.addEventListener('notificationclick', (event) => {
    event.notification.close();
    
    const url = event.notification.data?.url || '/';
    
    event.waitUntil(
        clients.matchAll({ type: 'window', includeUncontrolled: true })
            .then((clientList) => {
                for (const client of clientList) {
                    if (client.url === url && 'focus' in client) {
                        return client.focus();
                    }
                }
                if (clients.openWindow) {
                    return clients.openWindow(url);
                }
            })
    );
});

async function showNotification(title, body) {
    if (self.registration.showNotification) {
        await self.registration.showNotification(title, {
            body: body,
            icon: '/static/assets/images/logo.png',
            badge: '/static/assets/images/logo.png',
            vibrate: [200, 100, 200]
        });
    }
}

// ──────────────────────────────────────────────────────────
// 9. تحديث الـ Cache في الخلفية
// ──────────────────────────────────────────────────────────

self.addEventListener('message', (event) => {
    if (event.data === 'refresh-cache') {
        console.log('[SW] Refreshing cache...');
        
        event.waitUntil(
            caches.open(CACHE_NAME).then((cache) => {
                return cache.addAll(ALL_ASSETS);
            })
        );
    }
});

console.log('[SW] Service Worker loaded - Version:', CACHE_VERSION);
