/**
 * api_client.js
 * RIVA Health Platform - API Client
 * Client-side API calls with session management and offline support
 */

class RIVAAPIClient {
    constructor(baseURL = '/api') {
        this.baseURL = baseURL;
        this.sessionId = null;
        this.user = null;
        this.isOnline = navigator.onLine;
        
        // Load session from localStorage if exists
        this.loadSession();
        
        // Listen to online/offline events
        window.addEventListener('online', () => this.handleOnline());
        window.addEventListener('offline', () => this.handleOffline());
    }
    
    // ============================================
    // Session Management
    // ============================================
    
    loadSession() {
        try {
            const saved = localStorage.getItem('riva_session');
            if (saved) {
                const session = JSON.parse(saved);
                this.sessionId = session.sessionId;
                this.user = session.user;
                console.log('✅ Session loaded:', this.user?.user_id);
            }
        } catch (e) {
            console.error('Failed to load session:', e);
        }
    }
    
    saveSession(sessionId, user) {
        this.sessionId = sessionId;
        this.user = user;
        try {
            localStorage.setItem('riva_session', JSON.stringify({
                sessionId: sessionId,
                user: user,
                savedAt: new Date().toISOString()
            }));
        } catch (e) {
            console.error('Failed to save session:', e);
        }
    }
    
    clearSession() {
        this.sessionId = null;
        this.user = null;
        try {
            localStorage.removeItem('riva_session');
        } catch (e) {
            console.error('Failed to clear session:', e);
        }
    }
    
    // ============================================
    // Authentication
    // ============================================
    
    async login(userId, password) {
        try {
            const response = await fetch(`${this.baseURL}/auth/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ user_id: userId, password: password })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Login failed');
            }
            
            const data = await response.json();
            this.saveSession(data.session_id, data.user);
            
            return {
                success: true,
                sessionId: data.session_id,
                user: data.user
            };
        } catch (error) {
            console.error('Login error:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }
    
    async logout() {
        if (!this.sessionId) return { success: true };
        
        try {
            await fetch(`${this.baseURL}/auth/logout`, {
                method: 'POST',
                headers: this.getHeaders()
            });
        } catch (e) {
            console.warn('Logout API failed:', e);
        }
        
        this.clearSession();
        return { success: true };
    }
    
    // ============================================
    // Headers Helper
    // ============================================
    
    getHeaders() {
        const headers = {
            'Content-Type': 'application/json'
        };
        
        if (this.sessionId) {
            headers['X-Session-ID'] = this.sessionId;
        }
        
        return headers;
    }
    
    // ============================================
    // API Calls
    // ============================================
    
    async request(endpoint, options = {}) {
        if (!this.isOnline) {
            return this.handleOfflineRequest(endpoint, options);
        }
        
        const url = `${this.baseURL}${endpoint}`;
        const headers = {
            ...this.getHeaders(),
            ...options.headers
        };
        
        try {
            const response = await fetch(url, {
                ...options,
                headers: headers
            });
            
            // Handle 403 Forbidden (role required)
            if (response.status === 403) {
                const error = await response.json();
                console.warn('Access denied:', error.detail);
                throw new Error(error.detail || 'Access denied');
            }
            
            // Handle 401 Unauthorized
            if (response.status === 401) {
                this.clearSession();
                window.location.href = '/login';
                throw new Error('Session expired');
            }
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Request failed');
            }
            
            return await response.json();
        } catch (error) {
            console.error(`API Error [${endpoint}]:`, error);
            throw error;
        }
    }
    
    // ============================================
    // Triage API
    // ============================================
    
    async triage(symptoms) {
        return this.request('/triage/analyze', {
            method: 'POST',
            body: JSON.stringify({ symptoms: symptoms })
        });
    }
    
    async getTriageHistory(patientId = null) {
        const endpoint = patientId ? `/triage/history/${patientId}` : '/triage/history';
        return this.request(endpoint);
    }
    
    // ============================================
    // Chat API
    // ============================================
    
    async chat(message, context = null) {
        return this.request('/chat/message', {
            method: 'POST',
            body: JSON.stringify({ 
                message: message,
                context: context 
            })
        });
    }
    
    async getChatHistory() {
        return this.request('/chat/history');
    }
    
    // ============================================
    // Pregnancy API
    // ============================================
    
    async analyzePregnancy(data) {
        return this.request('/pregnancy/analyze', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }
    
    // ============================================
    // School Health API
    // ============================================
    
    async analyzeSchoolHealth(studentData) {
        return this.request('/school/analyze', {
            method: 'POST',
            body: JSON.stringify(studentData)
        });
    }
    
    // ============================================
    // Readmission API
    // ============================================
    
    async predictReadmission(patientId) {
        return this.request(`/readmission/predict/${patientId}`);
    }
    
    // ============================================
    // LOS API
    // ============================================
    
    async predictLOS(patientId) {
        return this.request(`/los/predict/${patientId}`);
    }
    
    // ============================================
    // Prescription API
    // ============================================
    
    async generatePrescription(data) {
        return this.request('/prescriptions/generate', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }
    
    // ============================================
    // Drug Interaction API
    // ============================================
    
    async checkInteraction(medications) {
        return this.request('/interactions/check', {
            method: 'POST',
            body: JSON.stringify({ medications: medications })
        });
    }
    
    // ============================================
    // Sentiment Analysis API
    // ============================================
    
    async analyzeSentiment(text) {
        return this.request('/sentiment/analyze', {
            method: 'POST',
            body: JSON.stringify({ text: text })
        });
    }
    
    // ============================================
    // AI Explanation API
    // ============================================
    
    async getExplanation(predictionId, predictionType) {
        return this.request(`/explain/${predictionType}/${predictionId}`);
    }
    
    // ============================================
    // Medical History API
    // ============================================
    
    async getMedicalHistory(patientId) {
        return this.request(`/history/${patientId}`);
    }
    
    // ============================================
    // Dashboard APIs
    // ============================================
    
    async getDoctorDashboard() {
        return this.request('/dashboard/doctor');
    }
    
    async getMotherDashboard(motherId) {
        return this.request(`/dashboard/mother/${motherId}`);
    }
    
    async getSchoolDashboard(schoolId) {
        return this.request(`/dashboard/school/${schoolId}`);
    }
    
    // ============================================
    // Offline Support
    // ============================================
    
    handleOnline() {
        this.isOnline = true;
        console.log('🟢 API Client: Online');
        // Sync pending requests
        this.syncPendingRequests();
    }
    
    handleOffline() {
        this.isOnline = false;
        console.log('🔴 API Client: Offline');
    }
    
    async handleOfflineRequest(endpoint, options) {
        console.warn('Offline mode: Request queued', endpoint);
        
        // Store request in IndexedDB for later sync
        await this.queueRequest(endpoint, options);
        
        throw new Error('You are offline. Request will be synced when online.');
    }
    
    async queueRequest(endpoint, options) {
        // Implement IndexedDB storage for offline requests
        const request = {
            id: Date.now(),
            endpoint: endpoint,
            options: options,
            timestamp: new Date().toISOString()
        };
        
        try {
            const db = await this.openDB();
            const tx = db.transaction(['requests'], 'readwrite');
            const store = tx.objectStore('requests');
            await store.add(request);
            console.log('Request queued:', request);
        } catch (e) {
            console.error('Failed to queue request:', e);
        }
    }
    
    async syncPendingRequests() {
        // Sync queued requests when back online
        try {
            const db = await this.openDB();
            const tx = db.transaction(['requests'], 'readonly');
            const store = tx.objectStore('requests');
            const requests = await store.getAll();
            
            for (const req of requests) {
                try {
                    await this.request(req.endpoint, req.options);
                    await this.removeQueuedRequest(req.id);
                } catch (e) {
                    console.error('Failed to sync request:', req, e);
                }
            }
        } catch (e) {
            console.error('Failed to sync pending requests:', e);
        }
    }
    
    async openDB() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open('RIVA_OfflineDB', 1);
            
            request.onerror = () => reject(request.error);
            request.onsuccess = () => resolve(request.result);
            
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains('requests')) {
                    db.createObjectStore('requests', { keyPath: 'id' });
                }
            };
        });
    }
    
    async removeQueuedRequest(id) {
        const db = await this.openDB();
        const tx = db.transaction(['requests'], 'readwrite');
        const store = tx.objectStore('requests');
        await store.delete(id);
    }
    
    // ============================================
    // Helper Methods
    // ============================================
    
    isAuthenticated() {
        return !!this.sessionId;
    }
    
    getUser() {
        return this.user;
    }
    
    getUserRole() {
        return this.user?.role || null;
    }
    
    async checkHealth() {
        try {
            const response = await fetch(`${this.baseURL}/health`);
            return response.ok;
        } catch {
            return false;
        }
    }
}

// Create singleton instance
window.rivaAPI = new RIVAAPIClient();

// ES Module export
export default RIVAAPIClient;
export { RIVAAPIClient };
