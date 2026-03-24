// sw.js - RIVA Service Worker
const CACHE_NAME = 'riva-v1';

self.addEventListener('install', (event) => {
    console.log('[SW] Installing...');
    event.waitUntil(self.skipWaiting());
});

self.addEventListener('activate', (event) => {
    console.log('[SW] Activating...');
    event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (event) => {
    event.respondWith(fetch(event.request).catch(() => {
        return new Response('Offline', { status: 503 });
    }));
});
