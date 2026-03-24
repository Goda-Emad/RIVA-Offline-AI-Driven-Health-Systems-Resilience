/**
 * qr_handler.js
 * ==============
 * RIVA Health Platform - QR Code Handler
 * معالج رموز QR للمرضى والوصفات الطبية
 * 
 * المسؤوليات:
 * - توليد رموز QR آمنة للروشتات والنتائج
 * - قراءة رموز QR من الكاميرا
 * - مشاركة البيانات عبر QR (مشفرة)
 * - تخزين QR مؤقتاً للاستخدام بدون إنترنت
 * - طلب إذن الكاميرا من المريض
 * 
 * المسار: web-app/src/static/js/qr_handler.js
 * 
 * التحسينات الأمنية:
 * - 🔐 تشفير البيانات الحساسة قبل تخزينها في QR
 * - 🔒 استخدام معرفات مشفرة (Tokens) بدلاً من البيانات المباشرة
 * - 📱 طلب إذن الكاميرا بشكل صريح
 * - 💾 دعم Offline مع Service Worker
 * - 🧹 تحرير موارد الكاميرا بشكل كامل
 * 
 * الإصدار: 4.2.1
 */

// ──────────────────────────────────────────────────────────
// 0. تحميل مكتبات خارجية (مع دعم Offline)
// ──────────────────────────────────────────────────────────

let QRCodeLibLoaded = false;
let QrScannerLoaded = false;
let cryptoLibLoaded = false;

// تحميل مكتبة التشفير المحلية
async function loadCryptoLib() {
    return new Promise((resolve) => {
        if (typeof CryptoJS !== 'undefined') {
            cryptoLibLoaded = true;
            resolve();
            return;
        }
        
        // استخدام مكتبة تشفير محلية مبسطة
        window.SimpleCrypto = {
            encrypt: (data, key) => {
                const str = JSON.stringify(data);
                let result = '';
                for (let i = 0; i < str.length; i++) {
                    result += String.fromCharCode(str.charCodeAt(i) ^ key.charCodeAt(i % key.length));
                }
                return btoa(result);
            },
            decrypt: (encrypted, key) => {
                const decoded = atob(encrypted);
                let result = '';
                for (let i = 0; i < decoded.length; i++) {
                    result += String.fromCharCode(decoded.charCodeAt(i) ^ key.charCodeAt(i % key.length));
                }
                return JSON.parse(result);
            }
        };
        cryptoLibLoaded = true;
        resolve();
    });
}

function loadQRCodeLib() {
    return new Promise((resolve) => {
        if (typeof QRCode !== 'undefined') {
            QRCodeLibLoaded = true;
            resolve();
            return;
        }
        
        // محاولة تحميل من المسار المحلي أولاً (للـ Offline)
        const script = document.createElement('script');
        script.src = '/static/libs/qrcode.min.js';
        script.onerror = () => {
            // Fallback إلى CDN
            const fallbackScript = document.createElement('script');
            fallbackScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js';
            fallbackScript.onload = () => {
                QRCodeLibLoaded = true;
                resolve();
            };
            document.head.appendChild(fallbackScript);
        };
        script.onload = () => {
            QRCodeLibLoaded = true;
            resolve();
        };
        document.head.appendChild(script);
    });
}

function loadQrScannerLib() {
    return new Promise((resolve) => {
        if (typeof QrScanner !== 'undefined') {
            QrScannerLoaded = true;
            resolve();
            return;
        }
        
        // محاولة تحميل من المسار المحلي أولاً
        const script = document.createElement('script');
        script.src = '/static/libs/qr-scanner.min.js';
        script.onerror = () => {
            const fallbackScript = document.createElement('script');
            fallbackScript.src = 'https://cdn.jsdelivr.net/npm/qr-scanner/qr-scanner.min.js';
            fallbackScript.onload = () => {
                QrScannerLoaded = true;
                resolve();
            };
            document.head.appendChild(fallbackScript);
        };
        script.onload = () => {
            QrScannerLoaded = true;
            resolve();
        };
        document.head.appendChild(script);
    });
}

// ──────────────────────────────────────────────────────────
// 1. كلاس QR Handler
// ──────────────────────────────────────────────────────────

class QRHandler {
    constructor() {
        this.qrInstances = new Map();
        this.qrScanner = null;
        this.isScanning = false;
        this.dbName = 'RIVA_QR_DB';
        this.dbVersion = 2;
        this.db = null;
        this.dbEnabled = true;
        this.encryptionKey = 'RIVA_SECURE_KEY_2024';
        this.tokenStorage = new Map();
        
        this.init();
    }
    
    // ──────────────────────────────────────────────────────────
    // 2. التهيئة
    // ──────────────────────────────────────────────────────────
    
    async init() {
        console.log('[QRHandler] Initializing...');
        
        await Promise.all([
            loadQRCodeLib(),
            loadQrScannerLib(),
            loadCryptoLib()
        ]);
        
        try {
            await this.openDatabase();
        } catch (error) {
            console.warn('[QRHandler] Database not available, continuing without storage:', error);
            this.dbEnabled = false;
        }
        
        this.injectStyles();
        
        console.log('[QRHandler] Initialized', { dbEnabled: this.dbEnabled });
    }
    
    // ──────────────────────────────────────────────────────────
    // 3. إدارة IndexedDB (مع معالجة الأخطاء)
    // ──────────────────────────────────────────────────────────
    
    openDatabase() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName, this.dbVersion);
            
            request.onerror = (event) => {
                console.error('[QRHandler] Failed to open database:', event.target.error);
                reject(event.target.error);
            };
            
            request.onsuccess = (event) => {
                this.db = event.target.result;
                console.log('[QRHandler] Database opened');
                resolve(this.db);
            };
            
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                
                if (!db.objectStoreNames.contains('qr_codes')) {
                    const store = db.createObjectStore('qr_codes', { 
                        keyPath: 'id' 
                    });
                    store.createIndex('by_patient', 'patientId');
                    store.createIndex('by_type', 'type');
                    store.createIndex('by_timestamp', 'timestamp');
                    store.createIndex('by_token', 'token');
                    store.createIndex('by_expiry', 'expiry');
                }
                
                console.log('[QRHandler] Database upgraded');
            };
        });
    }
    
    async saveQR(qrData) {
        if (!this.dbEnabled || !this.db) return false;
        
        const qrRecord = {
            id: qrData.id || `qr_${Date.now()}`,
            patientId: qrData.patientId,
            type: qrData.type,
            token: qrData.token,
            data: qrData.data,
            timestamp: Date.now(),
            expiry: qrData.expiry || Date.now() + 30 * 24 * 60 * 60 * 1000
        };
        
        return new Promise((resolve) => {
            try {
                const transaction = this.db.transaction(['qr_codes'], 'readwrite');
                const store = transaction.objectStore('qr_codes');
                store.put(qrRecord);
                transaction.oncomplete = () => {
                    console.log(`[QRHandler] QR saved: ${qrRecord.id}`);
                    resolve(true);
                };
                transaction.onerror = () => {
                    console.error('[QRHandler] Transaction failed');
                    resolve(false);
                };
            } catch (error) {
                console.error('[QRHandler] Failed to save QR:', error);
                resolve(false);
            }
        });
    }
    
    async getQRByToken(token) {
        if (!this.dbEnabled || !this.db) return null;
        
        return new Promise((resolve) => {
            try {
                const transaction = this.db.transaction(['qr_codes'], 'readonly');
                const store = transaction.objectStore('qr_codes');
                const index = store.index('by_token');
                const request = index.get(token);
                
                request.onsuccess = () => {
                    resolve(request.result || null);
                };
                request.onerror = () => {
                    resolve(null);
                };
            } catch (error) {
                console.error('[QRHandler] Failed to get QR:', error);
                resolve(null);
            }
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 4. توليد معرف مشفر (Token)
    // ──────────────────────────────────────────────────────────
    
    generateSecureToken(data) {
        const timestamp = Date.now();
        const random = Math.random().toString(36).substring(2, 15);
        const hash = btoa(`${timestamp}:${random}:${JSON.stringify(data)}`).substring(0, 32);
        return `RIVA_${hash}_${timestamp}`;
    }
    
    // ──────────────────────────────────────────────────────────
    // 5. تشفير البيانات الحساسة
    // ──────────────────────────────────────────────────────────
    
    encryptData(data) {
        try {
            if (typeof CryptoJS !== 'undefined') {
                return CryptoJS.AES.encrypt(JSON.stringify(data), this.encryptionKey).toString();
            }
            return SimpleCrypto.encrypt(data, this.encryptionKey);
        } catch (error) {
            console.error('[QRHandler] Encryption failed:', error);
            return null;
        }
    }
    
    decryptData(encrypted) {
        try {
            if (typeof CryptoJS !== 'undefined') {
                const bytes = CryptoJS.AES.decrypt(encrypted, this.encryptionKey);
                return JSON.parse(bytes.toString(CryptoJS.enc.Utf8));
            }
            return SimpleCrypto.decrypt(encrypted, this.encryptionKey);
        } catch (error) {
            console.error('[QRHandler] Decryption failed:', error);
            return null;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 6. توليد QR آمن (مع تشفير)
    // ──────────────────────────────────────────────────────────
    
    async generateSecureQR(elementId, data, options = {}) {
        await loadQRCodeLib();
        
        const element = document.getElementById(elementId);
        if (!element) {
            console.error(`[QRHandler] Element ${elementId} not found`);
            return null;
        }
        
        const token = this.generateSecureToken(data);
        const encryptedData = this.encryptData(data);
        
        let qrContent;
        
        if (options.useSecureLink && window.location.origin) {
            qrContent = `${window.location.origin}/api/qr/view/${token}`;
        } else {
            qrContent = JSON.stringify({
                v: 2,
                t: token,
                d: encryptedData
            });
        }
        
        if (this.dbEnabled) {
            await this.saveQR({
                id: options.id,
                patientId: options.patientId,
                type: options.type,
                token: token,
                data: encryptedData,
                expiry: options.expiry
            });
        }
        
        this.tokenStorage.set(token, {
            data: encryptedData,
            expiry: options.expiry || Date.now() + 30 * 24 * 60 * 60 * 1000
        });
        
        element.innerHTML = '';
        
        const qrOptions = {
            text: qrContent,
            width: options.width || 256,
            height: options.height || 256,
            colorDark: options.colorDark || '#000000',
            colorLight: options.colorLight || '#ffffff',
            correctLevel: QRCode.CorrectLevel.H
        };
        
        try {
            const qr = new QRCode(element, qrOptions);
            this.qrInstances.set(elementId, qr);
            
            console.log(`[QRHandler] Secure QR generated for ${elementId}`);
            return { qr, token };
            
        } catch (error) {
            console.error('[QRHandler] Failed to generate QR:', error);
            return null;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 7. توليد QR آمن للروشتة
    // ──────────────────────────────────────────────────────────
    
    async generateSecurePrescriptionQR(elementId, prescription, options = {}) {
        const sensitiveData = {
            type: 'prescription',
            id: prescription.prescription_id,
            patient: prescription.patient_id,
            doctor: prescription.doctor_id,
            date: prescription.prescribed_at,
            medications: prescription.medications.map(m => ({
                name: m.name,
                dose: m.dose,
                frequency: m.frequency,
                duration: m.duration_days
            })),
            signature: prescription.digital_signature
        };
        
        return this.generateSecureQR(elementId, sensitiveData, {
            ...options,
            type: 'prescription',
            patientId: prescription.patient_id,
            id: prescription.prescription_id,
            useSecureLink: options.useSecureLink !== false
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 8. توليد QR آمن للنتيجة
    // ──────────────────────────────────────────────────────────
    
    async generateSecureResultQR(elementId, patientId, result, options = {}) {
        const sensitiveData = {
            type: 'result',
            patient: patientId,
            date: new Date().toISOString(),
            prediction: result
        };
        
        return this.generateSecureQR(elementId, sensitiveData, {
            ...options,
            type: 'result',
            patientId: patientId,
            useSecureLink: options.useSecureLink !== false
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 9. توليد QR آمن للمريض
    // ──────────────────────────────────────────────────────────
    
    async generateSecurePatientQR(elementId, patientData, options = {}) {
        const sensitiveData = {
            type: 'patient',
            id: patientData.id,
            name: patientData.name,
            age: patientData.age,
            gender: patientData.gender,
            bloodType: patientData.bloodType,
            conditions: patientData.chronicConditions || []
        };
        
        return this.generateSecureQR(elementId, sensitiveData, {
            ...options,
            type: 'patient',
            patientId: patientData.id,
            useSecureLink: options.useSecureLink !== false
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 10. قراءة QR آمن (مع فك التشفير)
    // ──────────────────────────────────────────────────────────
    
    async readSecureQR(qrContent) {
        let parsed;
        
        try {
            parsed = JSON.parse(qrContent);
        } catch {
            if (qrContent.startsWith('http')) {
                return this.fetchSecureData(qrContent);
            }
            return { success: false, error: 'Invalid QR format' };
        }
        
        if (parsed.v !== 2) {
            return { success: false, error: 'Unsupported QR version' };
        }
        
        const { token, d: encryptedData } = parsed;
        
        let storedData = null;
        
        if (this.dbEnabled) {
            const record = await this.getQRByToken(token);
            if (record) {
                storedData = record.data;
            }
        }
        
        if (!storedData && this.tokenStorage.has(token)) {
            const cached = this.tokenStorage.get(token);
            if (cached.expiry > Date.now()) {
                storedData = cached.data;
            } else {
                this.tokenStorage.delete(token);
            }
        }
        
        const dataToDecrypt = storedData || encryptedData;
        
        if (!dataToDecrypt) {
            return { success: false, error: 'Data not found or expired' };
        }
        
        const decrypted = this.decryptData(dataToDecrypt);
        
        if (!decrypted) {
            return { success: false, error: 'Failed to decrypt data' };
        }
        
        return {
            success: true,
            data: decrypted,
            token: token
        };
    }
    
    async fetchSecureData(url) {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error('Network error');
            }
            const data = await response.json();
            return { success: true, data };
        } catch (error) {
            return { success: false, error: 'Failed to fetch data' };
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 11. طلب إذن الكاميرا (مع رسالة للمريض)
    // ──────────────────────────────────────────────────────────
    
    async requestCameraPermission(videoElementId, onScan, options = {}) {
        const videoElement = document.getElementById(videoElementId);
        if (!videoElement) {
            console.error(`[QRHandler] Video element ${videoElementId} not found`);
            return { success: false, error: 'element_not_found' };
        }
        
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            return { success: false, error: 'camera_not_supported' };
        }
        
        if (options.showPermissionDialog !== false) {
            const confirmed = await this.showPermissionDialog();
            if (!confirmed) {
                return { success: false, error: 'permission_denied' };
            }
        }
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: 'environment' } 
            });
            stream.getTracks().forEach(track => track.stop());
            
            return await this.startScanner(videoElementId, onScan, options);
            
        } catch (error) {
            console.error('[QRHandler] Camera permission denied:', error);
            
            if (error.name === 'NotAllowedError') {
                return { success: false, error: 'permission_denied' };
            }
            if (error.name === 'NotFoundError') {
                return { success: false, error: 'no_camera' };
            }
            return { success: false, error: error.message };
        }
    }
    
    showPermissionDialog() {
        return new Promise((resolve) => {
            const modal = document.createElement('div');
            modal.className = 'qr-permission-modal';
            modal.innerHTML = `
                <div class="modal-overlay"></div>
                <div class="modal-content">
                    <div class="modal-header">
                        <h3>📷 طلب الوصول إلى الكاميرا</h3>
                    </div>
                    <div class="modal-body">
                        <p>لتتمكن من مسح رمز QR، نحتاج إلى الوصول إلى كاميرا جهازك.</p>
                        <p class="privacy-note">
                            🔒 لن يتم حفظ أي صور أو فيديو. الكاميرا تستخدم فقط لمسح الرموز.
                        </p>
                    </div>
                    <div class="modal-footer">
                        <button class="modal-btn modal-btn-secondary" id="permission-deny">إلغاء</button>
                        <button class="modal-btn modal-btn-primary" id="permission-allow">موافق ✅</button>
                    </div>
                </div>
            `;
            
            document.body.appendChild(modal);
            this.injectModalStyles();
            
            setTimeout(() => modal.classList.add('show'), 10);
            
            const allowBtn = modal.querySelector('#permission-allow');
            const denyBtn = modal.querySelector('#permission-deny');
            const overlay = modal.querySelector('.modal-overlay');
            
            const close = (granted) => {
                modal.classList.remove('show');
                setTimeout(() => modal.remove(), 300);
                resolve(granted);
            };
            
            allowBtn.addEventListener('click', () => close(true));
            denyBtn.addEventListener('click', () => close(false));
            overlay.addEventListener('click', () => close(false));
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 12. بدء المسح
    // ──────────────────────────────────────────────────────────
    
    async startScanner(videoElementId, onScan, options = {}) {
        await loadQrScannerLib();
        
        const videoElement = document.getElementById(videoElementId);
        if (!videoElement) {
            return { success: false, error: 'element_not_found' };
        }
        
        if (this.qrScanner && this.isScanning) {
            await this.stopScanner();
        }
        
        try {
            this.qrScanner = new QrScanner(
                videoElement,
                async (result) => {
                    console.log('[QRHandler] QR scanned');
                    const secureResult = await this.readSecureQR(result);
                    
                    if (onScan) {
                        onScan(secureResult);
                    }
                    
                    if (options.autoClose !== false) {
                        await this.stopScanner();
                    }
                },
                {
                    highlightScanRegion: true,
                    highlightCodeOutline: true,
                    returnDetailedScanResult: true,
                    ...options
                }
            );
            
            await this.qrScanner.start();
            this.isScanning = true;
            
            return { success: true };
            
        } catch (error) {
            console.error('[QRHandler] Failed to start scanner:', error);
            return { success: false, error: error.message };
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 13. إيقاف المسح وتحرير الكاميرا
    // ──────────────────────────────────────────────────────────
    
    async stopScanner() {
        if (this.qrScanner) {
            try {
                await this.qrScanner.stop();
                
                if (this.qrScanner.$video && this.qrScanner.$video.srcObject) {
                    const tracks = this.qrScanner.$video.srcObject.getTracks();
                    tracks.forEach(track => track.stop());
                    this.qrScanner.$video.srcObject = null;
                }
                
                if (typeof this.qrScanner.destroy === 'function') {
                    this.qrScanner.destroy();
                }
                
                this.qrScanner = null;
                this.isScanning = false;
                
                console.log('[QRHandler] Scanner stopped and camera released');
            } catch (error) {
                console.error('[QRHandler] Error stopping scanner:', error);
            }
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 14. قراءة QR من ملف صورة
    // ──────────────────────────────────────────────────────────
    
    async scanFromImage(imageFile) {
        await loadQrScannerLib();
        
        return new Promise((resolve, reject) => {
            const imageUrl = URL.createObjectURL(imageFile);
            const image = new Image();
            
            image.onload = async () => {
                try {
                    const result = await QrScanner.scanImage(image);
                    URL.revokeObjectURL(imageUrl);
                    const secureResult = await this.readSecureQR(result);
                    resolve(secureResult);
                } catch (error) {
                    reject(error);
                }
            };
            
            image.onerror = () => {
                URL.revokeObjectURL(imageUrl);
                reject(new Error('Failed to load image'));
            };
            
            image.src = imageUrl;
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 15. مشاركة QR
    // ──────────────────────────────────────────────────────────
    
    async shareQR(elementId, options = {}) {
        const element = document.getElementById(elementId);
        if (!element) return false;
        
        const canvas = element.querySelector('canvas');
        if (!canvas) return false;
        
        try {
            const blob = await new Promise((resolve) => {
                canvas.toBlob(resolve, 'image/png');
            });
            
            const file = new File([blob], 'riva_qr.png', { type: 'image/png' });
            
            if (navigator.share) {
                await navigator.share({
                    title: options.title || 'RIVA QR Code',
                    text: options.text || 'مشاركة رمز QR من RIVA',
                    files: [file]
                });
            } else {
                const link = document.createElement('a');
                link.download = options.filename || 'riva_qr.png';
                link.href = URL.createObjectURL(blob);
                link.click();
                URL.revokeObjectURL(link.href);
            }
            
            return true;
        } catch (error) {
            console.error('[QRHandler] Failed to share QR:', error);
            return false;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 16. تنزيل QR
    // ──────────────────────────────────────────────────────────
    
    downloadQR(elementId, filename = 'riva_qr.png') {
        const element = document.getElementById(elementId);
        if (!element) return false;
        
        const canvas = element.querySelector('canvas');
        if (!canvas) return false;
        
        const link = document.createElement('a');
        link.download = filename;
        link.href = canvas.toDataURL('image/png');
        link.click();
        
        return true;
    }
    
    // ──────────────────────────────────────────────────────────
    // 17. طباعة QR
    // ──────────────────────────────────────────────────────────
    
    printQR(elementId) {
        const element = document.getElementById(elementId);
        if (!element) return false;
        
        const canvas = element.querySelector('canvas');
        if (!canvas) return false;
        
        const printWindow = window.open('', '_blank');
        printWindow.document.write(`
            <html>
            <head><title>RIVA QR Code</title></head>
            <body style="display:flex;justify-content:center;align-items:center;min-height:100vh">
                <img src="${canvas.toDataURL()}" alt="QR Code">
            </body>
            </html>
        `);
        printWindow.document.close();
        printWindow.print();
        
        return true;
    }
    
    // ──────────────────────────────────────────────────────────
    // 18. تنظيف QR القديمة
    // ──────────────────────────────────────────────────────────
    
    async cleanupOldQR() {
        if (!this.dbEnabled || !this.db) return 0;
        
        const now = Date.now();
        let deletedCount = 0;
        
        return new Promise((resolve) => {
            try {
                const transaction = this.db.transaction(['qr_codes'], 'readwrite');
                const store = transaction.objectStore('qr_codes');
                const index = store.index('by_expiry');
                const cursor = index.openCursor(IDBKeyRange.upperBound(now));
                
                cursor.onsuccess = (event) => {
                    const cursor = event.target.result;
                    if (cursor) {
                        cursor.delete();
                        deletedCount++;
                        cursor.continue();
                    } else {
                        console.log(`[QRHandler] Cleaned up ${deletedCount} old QR codes`);
                        resolve(deletedCount);
                    }
                };
                cursor.onerror = () => resolve(deletedCount);
            } catch (error) {
                console.error('[QRHandler] Cleanup failed:', error);
                resolve(0);
            }
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 19. الحصول على الحالة
    // ──────────────────────────────────────────────────────────
    
    getStatus() {
        return {
            librariesLoaded: {
                qrcode: QRCodeLibLoaded,
                qrScanner: QrScannerLoaded,
                crypto: cryptoLibLoaded
            },
            isScanning: this.isScanning,
            dbEnabled: this.dbEnabled,
            tokenStorageSize: this.tokenStorage.size
        };
    }
    
    // ──────────────────────────────────────────────────────────
    // 20. إضافة أنماط CSS
    // ──────────────────────────────────────────────────────────
    
    injectStyles() {
        if (document.getElementById('riva-qr-styles')) return;
        
        const styles = `
            <style id="riva-qr-styles">
                .qr-container { display: flex; flex-direction: column; align-items: center; padding: 16px; background: var(--white, #fff); border-radius: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
                .qr-actions { display: flex; gap: 12px; margin-top: 16px; flex-wrap: wrap; justify-content: center; }
                .qr-btn { padding: 8px 16px; border: none; border-radius: 8px; cursor: pointer; font-family: inherit; font-size: 14px; display: flex; align-items: center; gap: 6px; transition: all 0.2s ease; }
                .qr-btn-primary { background: var(--primary, #1a73e8); color: white; }
                .qr-scanner-container { position: relative; width: 100%; max-width: 400px; margin: 0 auto; }
                .qr-scanner-video { width: 100%; border-radius: 12px; background: var(--dark, #202124); }
                .qr-permission-modal .modal-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); z-index: 10000; }
                .qr-permission-modal .modal-content { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: white; border-radius: 16px; width: 90%; max-width: 350px; z-index: 10001; overflow: hidden; }
                .privacy-note { font-size: 12px; color: var(--gray, #5f6368); margin-top: 12px; padding: 8px; background: #e8f0fe; border-radius: 8px; }
                @media (max-width: 768px) { .qr-actions { flex-direction: column; width: 100%; } .qr-btn { justify-content: center; } }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }
    
    injectModalStyles() {
        if (document.getElementById('riva-qr-modal-styles')) return;
        
        const styles = `
            <style id="riva-qr-modal-styles">
                .qr-permission-modal { position: fixed; top: 0; left: 0; right: 0; bottom: 0; z-index: 10000; display: flex; align-items: center; justify-content: center; visibility: hidden; opacity: 0; transition: all 0.3s ease; }
                .qr-permission-modal.show { visibility: visible; opacity: 1; }
                .qr-permission-modal .modal-overlay { position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); }
                .qr-permission-modal .modal-content { position: relative; background: white; border-radius: 16px; width: 90%; max-width: 350px; max-height: 90vh; overflow-y: auto; animation: modalSlideIn 0.3s ease; }
                .qr-permission-modal .modal-header { padding: 16px 20px; border-bottom: 1px solid #e8eaed; }
                .qr-permission-modal .modal-header h3 { margin: 0; }
                .qr-permission-modal .modal-body { padding: 20px; }
                .qr-permission-modal .modal-footer { display: flex; gap: 12px; padding: 16px 20px; border-top: 1px solid #e8eaed; }
                .qr-permission-modal .modal-btn { flex: 1; padding: 10px; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: 500; }
                .qr-permission-modal .modal-btn-primary { background: #1a73e8; color: white; }
                .qr-permission-modal .modal-btn-secondary { background: #f8f9fa; color: #5f6368; }
                @keyframes modalSlideIn { from { transform: translateY(-30px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }
}

// ──────────────────────────────────────────────────────────
// 21. دوال مساعدة للاستخدام في الصفحات
// ──────────────────────────────────────────────────────────

async function startQRScanner(videoElementId, onScan, options = {}) {
    return qrHandler.requestCameraPermission(videoElementId, onScan, options);
}

function stopQRScanner() {
    return qrHandler.stopScanner();
}

async function generatePrescriptionQR(elementId, prescription, options = {}) {
    return qrHandler.generateSecurePrescriptionQR(elementId, prescription, options);
}

async function generateResultQR(elementId, patientId, result, options = {}) {
    return qrHandler.generateSecureResultQR(elementId, patientId, result, options);
}

async function generatePatientQR(elementId, patientData, options = {}) {
    return qrHandler.generateSecurePatientQR(elementId, patientData, options);
}

async function shareQR(elementId, options = {}) {
    return qrHandler.shareQR(elementId, options);
}

function downloadQR(elementId, filename = 'riva_qr.png') {
    return qrHandler.downloadQR(elementId, filename);
}

// ──────────────────────────────────────────────────────────
// 22. تصدير نسخة واحدة (Singleton)
// ──────────────────────────────────────────────────────────

// إنشاء نسخة واحدة
const qrHandler = new QRHandler();

// تخزين في window للاستخدام العادي
window.qrHandler = qrHandler;
window.rivaQRHandler = qrHandler;
window.startQRScanner = startQRScanner;
window.stopQRScanner = stopQRScanner;
window.generatePrescriptionQR = generatePrescriptionQR;
window.generateResultQR = generateResultQR;
window.generatePatientQR = generatePatientQR;
window.shareQR = shareQR;
window.downloadQR = downloadQR;

// ES Module export
export default qrHandler;
export { qrHandler };
