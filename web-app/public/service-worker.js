/**
 * service-worker.js
 * =================
 * RIVA Health Platform - Service Worker
 * عامل الخدمة للتشغيل بدون إنترنت (PWA)
 * 
 * المسار: web-app/public/service-worker.js
 * 
 * الإصدار: 4.2.1
 */

const CACHE_VERSION = 'v4.2.1';
const CACHE_NAME = `riva-${CACHE_VERSION}`;
const API_CACHE = 'riva-api';

// الملفات الأساسية للتخزين
const STATIC_ASSETS = [
  '/',
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
  '/offline-fallback.html'
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

// ملفات الوسائط الأساسية
const ASSET_FILES = [
  '/static/assets/logo.png',
  '/static/assets/favicon.ico',
  '/static/assets/audio/emergency-alert.mp3',
  '/static/assets/images/doctor-avatar.svg',
  '/static/assets/images/mother-avatar.svg',
  '/static/assets/images/school-avatar.svg',
  '/static/assets/images/patient-avatar.svg'
];

const ALL_ASSETS = [...STATIC_ASSETS, ...CSS_ASSETS, ...JS_ASSETS, ...ASSET_FILES];

// ──────────────────────────────────────────────────────────
// 1. تثبيت Service Worker
// ──────────────────────────────────────────────────────────

self.addEventListener('install', (event) => {
  console.log('[SW] Installing version:', CACHE_VERSION);
  
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
// 2. تفعيل Service Worker
// ──────────────────────────────────────────────────────────

self.addEventListener('activate', (event) => {
  console.log('[SW] Activating version:', CACHE_VERSION);
  
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
// 3. اعتراض الطلبات
// ──────────────────────────────────────────────────────────

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  
  // طلبات API
  if (url.pathname.startsWith('/api/')) {
    return event.respondWith(handleAPIRequest(event.request));
  }
  
  // طلبات الملفات الثابتة
  return event.respondWith(handleStaticRequest(event.request));
});

async function handleStaticRequest(request) {
  const cache = await caches.open(CACHE_NAME);
  const cachedResponse = await cache.match(request);
  const isNavigation = request.mode === 'navigate';
  
  if (cachedResponse) {
    // تحديث في الخلفية
    fetch(request).then((networkResponse) => {
      if (networkResponse && networkResponse.status === 200) {
        cache.put(request, networkResponse.clone());
      }
    }).catch(() => {});
    return cachedResponse;
  }
  
  try {
    const networkResponse = await fetch(request);
    if (networkResponse && networkResponse.status === 200) {
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    if (isNavigation) {
      return caches.match('/offline-fallback.html');
    }
    return new Response('Network error', { status: 408 });
  }
}

async function handleAPIRequest(request) {
  if (request.method !== 'GET') {
    return fetch(request).catch(() => {
      return new Response(JSON.stringify({
        success: false,
        offline: true,
        error: 'No internet connection'
      }), {
        status: 503,
        headers: { 'Content-Type': 'application/json' }
      });
    });
  }
  
  const cache = await caches.open(API_CACHE);
  
  try {
    const networkResponse = await fetch(request);
    if (networkResponse && networkResponse.status === 200) {
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    const cachedResponse = await cache.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
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
// 4. مزامنة الخلفية
// ──────────────────────────────────────────────────────────

self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-pending-requests') {
    event.waitUntil(syncPendingRequests());
  }
});

async function syncPendingRequests() {
  console.log('[SW] Syncing pending requests...');
  // يمكن إضافة منطق المزامنة هنا
  return true;
}

// ──────────────────────────────────────────────────────────
// 5. الإشعارات
// ──────────────────────────────────────────────────────────

self.addEventListener('push', (event) => {
  let data = {};
  if (event.data) {
    try {
      data = event.data.json();
    } catch {
      data = { title: 'RIVA', body: event.data.text() };
    }
  }
  
  event.waitUntil(
    self.registration.showNotification(data.title || 'RIVA', {
      body: data.body || 'لديك إشعار جديد',
      icon: '/static/assets/logo.png',
      badge: '/static/assets/logo.png',
      vibrate: [200, 100, 200],
      data: { url: data.url || '/' }
    })
  );
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const url = event.notification.data?.url || '/';
  event.waitUntil(
    clients.matchAll({ type: 'window' }).then((clientList) => {
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

console.log('[SW] Service Worker loaded - Version:', CACHE_VERSION);
