/**
 * prediction_client.js
 * ====================
 * RIVA Health Platform - AI Prediction Client
 * عميل التنبؤات الذكية - يوحد جميع دوال التنبؤ في مكان واحد
 * 
 * المسؤوليات:
 * - التواصل مع جميع واجهات التنبؤ (Readmission, LOS, Triage, Pregnancy, Combined)
 * - إدارة حالة التنبؤات (Loading, Error, Success)
 * - تخزين نتائج التنبؤات مؤقتاً
 * - دعم الوضع غير المتصل (Offline Mode)
 * 
 * المسار: web-app/src/static/js/prediction_client.js
 * 
 * التحسينات:
 * - واجهة موحدة لجميع التنبؤات
 * - دعم Batch Predictions (Parallel)
 * - تخزين مؤقت (Cache) مع صلاحية
 * - معالجة الأخطاء الموحدة
 * - دعم Offline Mode
 * - دعم GET و POST بشكل صحيح
 * 
 * الإصدار: 4.2.1
 */

// ──────────────────────────────────────────────────────────
// 1. كلاس Prediction Client
// ──────────────────────────────────────────────────────────

class PredictionClient {
    constructor() {
        this.apiClient = window.rivaClient || null;
        this.offlineManager = window.offlineManager || null;
        this.cache = new Map();
        this.cacheTTL = 300000; // 5 دقائق
        this.pendingPredictions = new Map();
        
        this.init();
    }
    
    // ──────────────────────────────────────────────────────────
    // 2. التهيئة
    // ──────────────────────────────────────────────────────────
    
    init() {
        console.log('[PredictionClient] Initializing...');
        
        if (!this.apiClient) {
            window.addEventListener('riva-client-ready', () => {
                this.apiClient = window.rivaClient;
                console.log('[PredictionClient] API Client connected');
            });
        }
        
        if (!this.offlineManager) {
            window.addEventListener('offline-manager-ready', () => {
                this.offlineManager = window.offlineManager;
                console.log('[PredictionClient] Offline Manager connected');
            });
        }
        
        // تنظيف الـ Cache بشكل دوري
        setInterval(() => this.cleanupCache(), 60000);
    }
    
    // ──────────────────────────────────────────────────────────
    // 3. إدارة الـ Cache
    // ──────────────────────────────────────────────────────────
    
    getCacheKey(endpoint, params, method = 'POST') {
        return `${method}:${endpoint}:${JSON.stringify(params)}`;
    }
    
    getFromCache(key) {
        const cached = this.cache.get(key);
        if (cached && cached.expiry > Date.now()) {
            return cached.data;
        }
        if (cached) {
            this.cache.delete(key);
        }
        return null;
    }
    
    setToCache(key, data, ttl = this.cacheTTL) {
        this.cache.set(key, {
            data: data,
            expiry: Date.now() + ttl
        });
    }
    
    cleanupCache() {
        const now = Date.now();
        for (const [key, value] of this.cache.entries()) {
            if (value.expiry <= now) {
                this.cache.delete(key);
            }
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 4. معالجة الطلبات (مع دعم Offline)
    // ──────────────────────────────────────────────────────────
    
    async executePrediction(endpoint, params = {}, options = {}) {
        const method = options.method || 'POST';
        const cacheKey = this.getCacheKey(endpoint, params, method);
        
        // التحقق من الـ Cache (فقط للـ GET)
        if (!options.skipCache && method === 'GET') {
            const cached = this.getFromCache(cacheKey);
            if (cached) {
                console.log(`[PredictionClient] Using cached result for ${endpoint}`);
                return cached;
            }
        }
        
        // التحقق من حالة الاتصال
        const isOnline = this.offlineManager ? this.offlineManager.isOnline : navigator.onLine;
        
        if (!isOnline && options.requireOnline) {
            throw new Error('لا يوجد اتصال بالإنترنت. هذا التنبؤ يتطلب اتصالاً بالشبكة.');
        }
        
        // إضافة إلى المعلقات لتجنب الطلبات المتكررة
        if (this.pendingPredictions.has(cacheKey)) {
            return this.pendingPredictions.get(cacheKey);
        }
        
        const promise = this.makePredictionRequest(endpoint, params, options, method);
        this.pendingPredictions.set(cacheKey, promise);
        
        try {
            const result = await promise;
            if (!options.skipCache && result.success && method === 'GET') {
                this.setToCache(cacheKey, result, options.cacheTTL);
            }
            return result;
        } finally {
            this.pendingPredictions.delete(cacheKey);
        }
    }
    
    async makePredictionRequest(endpoint, params, options, method = 'POST') {
        try {
            // إذا كان النظام غير متصل والطلب يسمح بالـ Offline
            const isOnline = this.offlineManager ? this.offlineManager.isOnline : navigator.onLine;
            
            if (!isOnline && options.allowOffline && this.offlineManager) {
                return this.queueOfflinePrediction(endpoint, params, options, method);
            }
            
            // بناء خيارات الطلب
            const requestOptions = {
                method: method,
                headers: options.headers || {}
            };
            
            // إضافة body فقط لطلبات POST و PUT و PATCH
            if (method !== 'GET' && method !== 'HEAD' && params && Object.keys(params).length > 0) {
                requestOptions.body = JSON.stringify(params);
                requestOptions.headers['Content-Type'] = 'application/json';
            }
            
            // إضافة query parameters لطلبات GET
            let url = endpoint;
            if (method === 'GET' && params && Object.keys(params).length > 0) {
                const queryParams = new URLSearchParams();
                for (const [key, value] of Object.entries(params)) {
                    if (value !== undefined && value !== null) {
                        queryParams.append(key, value);
                    }
                }
                const queryString = queryParams.toString();
                if (queryString) {
                    url = `${endpoint}?${queryString}`;
                }
            }
            
            const response = await this.apiClient.request(url, requestOptions);
            
            return {
                success: true,
                data: response,
                timestamp: Date.now(),
                source: 'online',
                method: method
            };
        } catch (error) {
            console.error(`[PredictionClient] Prediction failed for ${endpoint}:`, error);
            
            // محاولة استخدام الـ Cache كـ Fallback (لجميع الطلبات)
            const cacheKey = this.getCacheKey(endpoint, params, method);
            const cached = this.getFromCache(cacheKey);
            if (cached) {
                return {
                    ...cached,
                    source: 'cache_fallback',
                    warning: 'استخدم بيانات مخزنة مؤقتاً بسبب خطأ في الاتصال'
                };
            }
            
            throw error;
        }
    }
    
    async queueOfflinePrediction(endpoint, params, options, method = 'POST') {
        // بناء URL كامل
        let url = `${this.apiClient.baseURL}${endpoint}`;
        
        // لطلبات GET، نضيف المعاملات للـ URL
        if (method === 'GET' && params && Object.keys(params).length > 0) {
            const queryParams = new URLSearchParams();
            for (const [key, value] of Object.entries(params)) {
                if (value !== undefined && value !== null) {
                    queryParams.append(key, value);
                }
            }
            const queryString = queryParams.toString();
            if (queryString) {
                url = `${url}?${queryString}`;
            }
        }
        
        // تحضير body لطلبات POST
        let body = null;
        if (method !== 'GET' && method !== 'HEAD' && params && Object.keys(params).length > 0) {
            body = JSON.stringify(params);
        }
        
        const queueResult = await this.offlineManager.queueRequest(url, {
            method: method,
            headers: { 'Content-Type': 'application/json' },
            body: body,
            priority: options.priority || 2
        });
        
        return {
            success: false,
            queued: true,
            message: 'تم حفظ الطلب وسيتم إرساله عند عودة الاتصال',
            data: queueResult,
            source: 'offline_queue',
            method: method
        };
    }
    
    // ──────────────────────────────────────────────────────────
    // 5. ========== READMISSION PREDICTION ==========
    // ──────────────────────────────────────────────────────────
    
    async predictReadmission(patientData, options = {}) {
        const endpoint = '/api/predict/readmission';
        
        return this.executePrediction(endpoint, patientData, {
            ...options,
            method: 'POST',
            requireOnline: false,
            allowOffline: true
        });
    }
    
    async predictReadmissionFromChat(chatText, patientId = null, options = {}) {
        const endpoint = '/api/predict/readmission/from-chat';
        const params = { chat_text: chatText, patient_id: patientId };
        
        return this.executePrediction(endpoint, params, {
            ...options,
            method: 'POST',
            requireOnline: false,
            allowOffline: true
        });
    }
    
    async batchPredictReadmission(patients, options = {}) {
        const endpoint = '/api/predict/readmission/batch';
        const params = { patients };
        
        return this.executePrediction(endpoint, params, {
            ...options,
            method: 'POST',
            requireOnline: true,
            skipCache: true
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 6. ========== LOS PREDICTION ==========
    // ──────────────────────────────────────────────────────────
    
    async predictLOS(patientData, options = {}) {
        const endpoint = '/api/predict/los';
        
        return this.executePrediction(endpoint, patientData, {
            ...options,
            method: 'POST',
            requireOnline: false,
            allowOffline: true
        });
    }
    
    async predictLOSFromChat(chatText, patientId = null, options = {}) {
        const endpoint = '/api/predict/los/from-chat';
        const params = { chat_text: chatText, patient_id: patientId };
        
        return this.executePrediction(endpoint, params, {
            ...options,
            method: 'POST',
            requireOnline: false,
            allowOffline: true
        });
    }
    
    async batchPredictLOS(patients, options = {}) {
        const endpoint = '/api/predict/los/batch';
        const params = { patients };
        
        return this.executePrediction(endpoint, params, {
            ...options,
            method: 'POST',
            requireOnline: true,
            skipCache: true
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 7. ========== TRIAGE PREDICTION ==========
    // ──────────────────────────────────────────────────────────
    
    async predictTriage(patientData, options = {}) {
        const endpoint = '/triage/predict';
        
        return this.executePrediction(endpoint, patientData, {
            ...options,
            method: 'POST',
            requireOnline: false,
            allowOffline: true,
            priority: 1  // أولوية عالية للطوارئ
        });
    }
    
    async batchPredictTriage(patients, options = {}) {
        const endpoint = '/triage/predict-batch';
        const params = { patients };
        
        return this.executePrediction(endpoint, params, {
            ...options,
            method: 'POST',
            requireOnline: true,
            skipCache: true
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 8. ========== PREGNANCY PREDICTION ==========
    // ──────────────────────────────────────────────────────────
    
    async predictPregnancyRisk(patientData, options = {}) {
        const endpoint = '/api/predict/pregnancy/predict';
        
        return this.executePrediction(endpoint, patientData, {
            ...options,
            method: 'POST',
            requireOnline: false,
            allowOffline: true
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 9. ========== COMBINED PREDICTION ==========
    // ──────────────────────────────────────────────────────────
    
    async predictCombined(patientData, options = {}) {
        const endpoint = '/api/v1/predict/combined';
        
        return this.executePrediction(endpoint, patientData, {
            ...options,
            method: 'POST',
            requireOnline: false,
            allowOffline: true
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 10. ========== SENTIMENT ANALYSIS ==========
    // ──────────────────────────────────────────────────────────
    
    async analyzeSentiment(text, patientId = null, options = {}) {
        const endpoint = '/api/sentiment/analyze';
        const params = { text, patient_id: patientId };
        
        return this.executePrediction(endpoint, params, {
            ...options,
            method: 'POST',
            requireOnline: false,
            allowOffline: true,
            priority: options.emergency ? 1 : 3
        });
    }
    
    async batchAnalyzeSentiment(texts, options = {}) {
        const endpoint = '/api/sentiment/analyze-batch';
        const params = { texts };
        
        return this.executePrediction(endpoint, params, {
            ...options,
            method: 'POST',
            requireOnline: true,
            skipCache: true
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 11. ========== SCHOOL HEALTH PREDICTION ==========
    // ──────────────────────────────────────────────────────────
    
    async analyzeStudent(studentData, options = {}) {
        const endpoint = '/api/school/analyze-student';
        
        return this.executePrediction(endpoint, studentData, {
            ...options,
            method: 'POST',
            requireOnline: false,
            allowOffline: true
        });
    }
    
    async analyzeClass(classData, options = {}) {
        const endpoint = '/api/school/analyze-class';
        
        return this.executePrediction(endpoint, classData, {
            ...options,
            method: 'POST',
            requireOnline: true,
            skipCache: true
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 12. ========== DRUG INTERACTIONS ==========
    // ──────────────────────────────────────────────────────────
    
    async checkDrugInteraction(newDrug, currentDrugs = [], options = {}) {
        const endpoint = '/interactions/check';
        const params = { new_drug: newDrug, current_drugs: currentDrugs };
        
        return this.executePrediction(endpoint, params, {
            ...options,
            method: 'POST',
            requireOnline: false,
            allowOffline: true,
            priority: 1  // أولوية عالية لسلامة الدواء
        });
    }
    
    async checkBulkInteractions(medications, options = {}) {
        const endpoint = '/interactions/check-bulk';
        const params = { medications };
        
        return this.executePrediction(endpoint, params, {
            ...options,
            method: 'POST',
            requireOnline: false,
            allowOffline: true,
            priority: 1
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 13. ========== EXPLANATION ==========
    // ──────────────────────────────────────────────────────────
    
    async getExplanation(patientId, predictionType, features, options = {}) {
        const endpoint = '/api/v1/explain';
        const params = { patient_id: patientId, prediction_type: predictionType, features };
        
        return this.executePrediction(endpoint, params, {
            ...options,
            method: 'POST',
            requireOnline: false,
            allowOffline: true,
            cacheTTL: 600000  // 10 دقائق للشروحات
        });
    }
    
    async getSimpleExplanation(patientId, predictionType, features, options = {}) {
        const endpoint = '/api/v1/explain/simple';
        const params = { patient_id: patientId, prediction_type: predictionType, features };
        
        return this.executePrediction(endpoint, params, {
            ...options,
            method: 'POST',
            requireOnline: false,
            allowOffline: true
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 14. ========== MEDICAL HISTORY ==========
    // ──────────────────────────────────────────────────────────
    
    async getMedicalHistory(patientId, options = {}, filters = {}) {
        const endpoint = `/api/v1/history/${patientId}`;
        
        return this.executePrediction(endpoint, filters, {
            ...options,
            method: 'GET',
            requireOnline: false,
            allowOffline: true,
            cacheTTL: 300000
        });
    }
    
    async searchMedicalHistory(patientId, query, sections = [], options = {}) {
        const endpoint = '/api/v1/history/search';
        const params = { patient_id: patientId, query, sections };
        
        return this.executePrediction(endpoint, params, {
            ...options,
            method: 'POST',
            requireOnline: false,
            allowOffline: true
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 15. ========== PRESCRIPTIONS ==========
    // ──────────────────────────────────────────────────────────
    
    async createPrescription(prescriptionData, options = {}) {
        const endpoint = '/prescriptions/create';
        
        return this.executePrediction(endpoint, prescriptionData, {
            ...options,
            method: 'POST',
            requireOnline: false,
            allowOffline: true,
            priority: 2
        });
    }
    
    async getPatientPrescriptions(patientId, options = {}) {
        const endpoint = `/prescriptions/patient/${patientId}`;
        
        return this.executePrediction(endpoint, {}, {
            ...options,
            method: 'GET',
            requireOnline: false,
            allowOffline: true
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 16. ========== FAMILY LINKS ==========
    // ──────────────────────────────────────────────────────────
    
    async getFamilyLinks(patientId, includeRiskReport = true, options = {}) {
        const endpoint = `/api/v1/family-links/${patientId}`;
        const params = { include_risk_report: includeRiskReport };
        
        return this.executePrediction(endpoint, params, {
            ...options,
            method: 'GET',
            requireOnline: false,
            allowOffline: true,
            cacheTTL: 3600000  // ساعة للبيانات العائلية
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 17. ========== UTILITY METHODS ==========
    // ──────────────────────────────────────────────────────────
    
    clearCache() {
        this.cache.clear();
        console.log('[PredictionClient] Cache cleared');
    }
    
    getCacheStats() {
        return {
            size: this.cache.size,
            pendingCount: this.pendingPredictions.size,
            ttl: this.cacheTTL
        };
    }
    
    // ──────────────────────────────────────────────────────────
    // 18. ========== BATCH PREDICTION UTILITY (Parallel) ==========
    // ──────────────────────────────────────────────────────────
    
    async batchPredict(predictions, options = {}) {
        // ✅ تنفيذ متوازي باستخدام Promise.allSettled
        const promises = predictions.map(async (pred) => {
            try {
                // الحصول على الدالة المناسبة
                let method = this[pred.method];
                if (!method) {
                    throw new Error(`Method ${pred.method} not found`);
                }
                
                // استدعاء الدالة مع المعاملات
                const result = await method.call(this, pred.params, options);
                
                return {
                    id: pred.id,
                    success: true,
                    data: result
                };
            } catch (error) {
                return {
                    id: pred.id,
                    success: false,
                    error: error.message
                };
            }
        });
        
        // ✅ انتظار جميع الطلبات بالتوازي
        const results = await Promise.allSettled(promises);
        
        // معالجة النتائج
        const processedResults = [];
        const errors = [];
        
        for (const result of results) {
            if (result.status === 'fulfilled') {
                processedResults.push(result.value);
                if (!result.value.success) {
                    errors.push(result.value);
                }
            } else {
                // Promise rejected (نادراً ما يحدث بسبب الـ try/catch)
                errors.push({
                    id: 'unknown',
                    error: result.reason?.message || 'Unknown error'
                });
            }
        }
        
        return {
            success: errors.length === 0,
            results: processedResults,
            errors: errors,
            total: predictions.length,
            successCount: processedResults.filter(r => r.success).length,
            errorCount: errors.length,
            parallel: true
        };
    }
    
    // ──────────────────────────────────────────────────────────
    // 19. ========== BATCH WITH CONCURRENCY LIMIT ==========
    // ──────────────────────────────────────────────────────────
    
    async batchPredictWithLimit(predictions, concurrency = 3, options = {}) {
        const results = [];
        const errors = [];
        
        // تقسيم الطلبات إلى مجموعات متوازية
        for (let i = 0; i < predictions.length; i += concurrency) {
            const batch = predictions.slice(i, i + concurrency);
            
            const batchResults = await Promise.allSettled(
                batch.map(async (pred) => {
                    try {
                        let method = this[pred.method];
                        if (!method) {
                            throw new Error(`Method ${pred.method} not found`);
                        }
                        
                        const result = await method.call(this, pred.params, options);
                        
                        return {
                            id: pred.id,
                            success: true,
                            data: result
                        };
                    } catch (error) {
                        return {
                            id: pred.id,
                            success: false,
                            error: error.message
                        };
                    }
                })
            );
            
            for (const result of batchResults) {
                if (result.status === 'fulfilled') {
                    results.push(result.value);
                    if (!result.value.success) {
                        errors.push(result.value);
                    }
                }
            }
        }
        
        return {
            success: errors.length === 0,
            results,
            errors,
            total: predictions.length,
            successCount: results.filter(r => r.success).length,
            errorCount: errors.length,
            concurrency: concurrency,
            parallel: true
        };
    }
}

// ──────────────────────────────────────────────────────────
// 20. تصدير نسخة واحدة (Singleton)
// ──────────────────────────────────────────────────────────

// إنشاء نسخة واحدة
const predictionClient = new PredictionClient();

// تخزين في window للاستخدام العادي
window.predictionClient = predictionClient;
window.rivaPredictionClient = predictionClient;

// ES Module export
export default predictionClient;
export { predictionClient };
