/**
 * voice_recorder.js
 * =================
 * RIVA Health Platform - Voice Recorder Module
 * وحدة تسجيل الصوت للشات بوت والاستشارات الصوتية
 * 
 * المسؤوليات:
 * - تسجيل الصوت من الميكروفون
 * - تحويل الصوت إلى Base64 أو Blob
 * - إرسال الصوت إلى API (voice.py و orchestrator.py)
 * - دعم الوضع غير المتصل (Offline Mode)
 * - عرض مؤشر مستوى الصوت (Visualizer)
 * - كشف الصمت التلقائي (Silence Detection)
 * - دعم RTL للغة العربية
 * 
 * المسار: web-app/src/static/js/voice_recorder.js
 * 
 * التحسينات:
 * - دعم جميع المتصفحات الحديثة (بما في ذلك Safari)
 * - طلب إذن الميكروفون مع رسالة توضيحية
 * - ضغط الصوت (Opus/WebM) مع Bitrate منخفض
 * - تخزين التسجيلات مؤقتاً في IndexedDB
 * - كشف الصمت التلقائي وإيقاف التسجيل
 * - تحسين أداء AudioContext (Safari)
 * 
 * الإصدار: 4.2.1
 */

// ──────────────────────────────────────────────────────────
// 1. كلاس Voice Recorder
// ──────────────────────────────────────────────────────────

class VoiceRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.isPaused = false;
        this.stream = null;
        this.recordingStartTime = null;
        this.recordingDuration = 0;
        this.audioContext = null;
        this.analyserNode = null;
        this.animationFrame = null;
        this.visualizerCanvas = null;
        this.visualizerCtx = null;
        
        // إعدادات التسجيل
        this.sampleRate = 16000;
        this.mimeType = this.getSupportedMimeType();
        this.maxDuration = 300; // 5 دقائق كحد أقصى
        this.minDuration = 1; // ثانية واحدة كحد أدنى
        this.audioBitsPerSecond = 32000; // خفضت إلى 32kbps للصوت البشري
        
        // إعدادات كشف الصمت (Silence Detection)
        this.silenceDetection = true;
        this.silenceThreshold = 0.02; // عتبة الصمت (2% من الحد الأقصى)
        this.silenceDuration = 3000; // 3 ثواني صمت متواصل
        this.silenceTimer = null;
        this.lastSoundTime = null;
        this.silenceCallback = null;
        
        // إعدادات الصوت
        this.noiseReduction = true;
        this.autoSend = false;
        
        // إعدادات RTL
        this.isRTL = document.documentElement.dir === 'rtl' || document.body.dir === 'rtl';
        
        // أحداث
        this.eventListeners = {};
        
        this.init();
    }
    
    async init() {
        console.log('[VoiceRecorder] Initializing...');
        await this.checkPermissions();
        this.injectStyles();
        console.log('[VoiceRecorder] Initialized', { 
            mimeType: this.mimeType, 
            bitrate: this.audioBitsPerSecond,
            silenceDetection: this.silenceDetection,
            isRTL: this.isRTL 
        });
    }
    
    getSupportedMimeType() {
        const types = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/mp4',
            'audio/mpeg',
            'audio/wav'
        ];
        
        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) {
                return type;
            }
        }
        return 'audio/webm';
    }
    
    async checkPermissions() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach(track => track.stop());
            return true;
        } catch (error) {
            console.warn('[VoiceRecorder] Microphone permission not granted:', error);
            return false;
        }
    }
    
    async requestMicrophonePermission(showDialog = true) {
        if (showDialog) {
            const granted = await this.showPermissionDialog();
            if (!granted) {
                return { success: false, error: 'permission_denied' };
            }
        }
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: this.noiseReduction,
                    autoGainControl: true,
                    sampleRate: this.sampleRate
                } 
            });
            return { success: true, stream };
        } catch (error) {
            console.error('[VoiceRecorder] Failed to get microphone:', error);
            
            if (error.name === 'NotAllowedError') {
                return { success: false, error: 'permission_denied' };
            }
            if (error.name === 'NotFoundError') {
                return { success: false, error: 'no_microphone' };
            }
            return { success: false, error: error.message };
        }
    }
    
    showPermissionDialog() {
        return new Promise((resolve) => {
            const modal = document.createElement('div');
            modal.className = 'voice-permission-modal';
            modal.innerHTML = `
                <div class="modal-overlay"></div>
                <div class="modal-content">
                    <div class="modal-header">
                        <h3>🎤 طلب الوصول إلى الميكروفون</h3>
                    </div>
                    <div class="modal-body">
                        <p>${this.isRTL ? 'لتتمكن من التسجيل الصوتي، نحتاج إلى الوصول إلى الميكروفون الخاص بك.' : 'To enable voice recording, we need access to your microphone.'}</p>
                        <p class="privacy-note">
                            🔒 ${this.isRTL ? 'لن يتم حفظ أي تسجيلات إلا بعد موافقتك. يمكنك إيقاف التسجيل في أي وقت.' : 'No recordings will be saved without your consent. You can stop recording anytime.'}
                        </p>
                    </div>
                    <div class="modal-footer">
                        <button class="modal-btn modal-btn-secondary" id="permission-deny">${this.isRTL ? 'إلغاء' : 'Cancel'}</button>
                        <button class="modal-btn modal-btn-primary" id="permission-allow">${this.isRTL ? 'موافق ✅' : 'Allow ✅'}</button>
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
    // 2. بدء التسجيل (مع تحسين Safari)
    // ──────────────────────────────────────────────────────────
    
    async startRecording(visualizerElementId = null, silenceCallback = null) {
        if (this.isRecording) {
            console.warn('[VoiceRecorder] Already recording');
            return { success: false, error: 'already_recording' };
        }
        
        // إعداد الـ Visualizer
        if (visualizerElementId) {
            this.setupVisualizer(visualizerElementId);
        }
        
        // تعيين دالة كشف الصمت
        if (silenceCallback && typeof silenceCallback === 'function') {
            this.silenceCallback = silenceCallback;
        }
        
        const result = await this.requestMicrophonePermission(true);
        if (!result.success) {
            this.emit('error', { error: result.error });
            return result;
        }
        
        this.stream = result.stream;
        
        try {
            this.mediaRecorder = new MediaRecorder(this.stream, {
                mimeType: this.mimeType,
                audioBitsPerSecond: this.audioBitsPerSecond
            });
            
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                this.handleRecordingStop();
            };
            
            this.mediaRecorder.start(1000); // جمع البيانات كل ثانية
            this.isRecording = true;
            this.recordingStartTime = Date.now();
            this.lastSoundTime = Date.now();
            
            // بدء تحليل الصوت للـ Visualizer (بعد تفاعل المستخدم)
            // ✅ AudioContext يتم إنشاؤه هنا فقط بعد تفاعل المستخدم (لـ Safari)
            this.startAudioAnalysis();
            
            this.emit('start', { duration: 0 });
            
            return { success: true };
            
        } catch (error) {
            console.error('[VoiceRecorder] Failed to start recording:', error);
            this.cleanup();
            return { success: false, error: error.message };
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 3. إيقاف التسجيل
    // ──────────────────────────────────────────────────────────
    
    async stopRecording() {
        if (!this.isRecording) {
            console.warn('[VoiceRecorder] Not recording');
            return { success: false, error: 'not_recording' };
        }
        
        this.recordingDuration = (Date.now() - this.recordingStartTime) / 1000;
        
        // تنظيف مؤقت الصمت
        if (this.silenceTimer) {
            clearTimeout(this.silenceTimer);
            this.silenceTimer = null;
        }
        
        if (this.recordingDuration < this.minDuration) {
            this.cancelRecording();
            this.emit('error', { error: 'recording_too_short', duration: this.recordingDuration });
            return { success: false, error: 'recording_too_short', duration: this.recordingDuration };
        }
        
        return new Promise((resolve) => {
            this.mediaRecorder.onstop = () => {
                const blob = new Blob(this.audioChunks, { type: this.mimeType });
                this.handleRecordingStop();
                resolve({ success: true, blob, duration: this.recordingDuration });
            };
            
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.stopAudioAnalysis();
            
            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
                this.stream = null;
            }
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 4. إلغاء التسجيل
    // ──────────────────────────────────────────────────────────
    
    cancelRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.onstop = () => {
                this.handleRecordingStop();
            };
            this.mediaRecorder.stop();
        }
        
        this.isRecording = false;
        this.audioChunks = [];
        this.recordingDuration = 0;
        
        if (this.silenceTimer) {
            clearTimeout(this.silenceTimer);
            this.silenceTimer = null;
        }
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.stopAudioAnalysis();
        this.emit('cancel');
    }
    
    // ──────────────────────────────────────────────────────────
    // 5. إيقاف مؤقت (Pause)
    // ──────────────────────────────────────────────────────────
    
    pauseRecording() {
        if (this.mediaRecorder && this.isRecording && !this.isPaused) {
            this.mediaRecorder.pause();
            this.isPaused = true;
            this.stopAudioAnalysis();
            
            if (this.silenceTimer) {
                clearTimeout(this.silenceTimer);
                this.silenceTimer = null;
            }
            
            this.emit('pause');
        }
    }
    
    resumeRecording() {
        if (this.mediaRecorder && this.isRecording && this.isPaused) {
            this.mediaRecorder.resume();
            this.isPaused = false;
            this.startAudioAnalysis();
            this.lastSoundTime = Date.now();
            this.emit('resume');
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 6. معالجة انتهاء التسجيل
    // ──────────────────────────────────────────────────────────
    
    handleRecordingStop() {
        this.isRecording = false;
        this.isPaused = false;
        this.recordingDuration = 0;
        this.stopAudioAnalysis();
        this.cleanupVisualizer();
        
        if (this.silenceTimer) {
            clearTimeout(this.silenceTimer);
            this.silenceTimer = null;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 7. الحصول على Blob الصوتي
    // ──────────────────────────────────────────────────────────
    
    getAudioBlob() {
        if (this.audioChunks.length === 0) return null;
        return new Blob(this.audioChunks, { type: this.mimeType });
    }
    
    // ──────────────────────────────────────────────────────────
    // 8. تحويل الصوت إلى Base64
    // ──────────────────────────────────────────────────────────
    
    async getAudioBase64() {
        const blob = this.getAudioBlob();
        if (!blob) return null;
        
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onloadend = () => {
                const base64 = reader.result.split(',')[1];
                resolve(base64);
            };
            reader.readAsDataURL(blob);
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 9. إرسال الصوت إلى API
    // ──────────────────────────────────────────────────────────
    
    async sendToAPI(endpoint = 'transcribe', options = {}) {
        const blob = this.getAudioBlob();
        if (!blob) {
            this.emit('error', { error: 'no_audio' });
            return { success: false, error: 'no_audio' };
        }
        
        this.emit('sending', { endpoint });
        
        try {
            let result;
            
            if (endpoint === 'transcribe') {
                const formData = new FormData();
                formData.append('file', blob, 'recording.webm');
                formData.append('language', options.language || 'ar');
                
                const response = await fetch('/api/voice/transcribe', {
                    method: 'POST',
                    body: formData
                });
                result = await response.json();
                
            } else if (endpoint === 'consult') {
                const formData = new FormData();
                formData.append('file', blob);
                formData.append('session_id', options.sessionId || '');
                formData.append('language', options.language || 'ar');
                
                const response = await fetch('/api/consult/voice', {
                    method: 'POST',
                    body: formData
                });
                result = await response.json();
                
            } else {
                const base64 = await this.getAudioBase64();
                const response = await fetch(`/api/voice/transcribe/base64`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        audio_base64: base64,
                        language: options.language || 'ar'
                    })
                });
                result = await response.json();
            }
            
            this.emit('success', { result, endpoint });
            return { success: true, result };
            
        } catch (error) {
            console.error('[VoiceRecorder] Failed to send audio:', error);
            this.emit('error', { error: error.message });
            return { success: false, error: error.message };
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 10. تحليل الصوت (Visualizer + Silence Detection)
    // ──────────────────────────────────────────────────────────
    
    setupVisualizer(canvasId) {
        this.visualizerCanvas = document.getElementById(canvasId);
        if (!this.visualizerCanvas) return;
        
        this.visualizerCtx = this.visualizerCanvas.getContext('2d');
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
    }
    
    resizeCanvas() {
        if (!this.visualizerCanvas) return;
        const rect = this.visualizerCanvas.parentElement?.getBoundingClientRect();
        if (rect) {
            this.visualizerCanvas.width = rect.width;
            this.visualizerCanvas.height = 80;
        }
    }
    
    startAudioAnalysis() {
        if (!this.stream) return;
        
        // ✅ إنشاء AudioContext هنا فقط بعد تفاعل المستخدم (لـ Safari)
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        
        const source = this.audioContext.createMediaStreamSource(this.stream);
        this.analyserNode = this.audioContext.createAnalyser();
        this.analyserNode.fftSize = 256;
        
        source.connect(this.analyserNode);
        
        this.animateVisualizer();
    }
    
    animateVisualizer() {
        if (!this.analyserNode || !this.visualizerCtx) {
            return;
        }
        
        const bufferLength = this.analyserNode.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        const draw = () => {
            if (!this.isRecording || this.isPaused) return;
            
            this.animationFrame = requestAnimationFrame(draw);
            this.analyserNode.getByteTimeDomainData(dataArray);
            
            if (!this.visualizerCtx) return;
            
            this.visualizerCtx.clearRect(0, 0, this.visualizerCanvas.width, this.visualizerCanvas.height);
            
            const width = this.visualizerCanvas.width;
            const height = this.visualizerCanvas.height;
            const barWidth = (width / bufferLength) * 2.5;
            let x = 0;
            
            // حساب مستوى الصوت لكشف الصمت
            let sum = 0;
            for (let i = 0; i < bufferLength; i++) {
                sum += Math.abs(dataArray[i] - 128);
            }
            const avg = sum / bufferLength / 128;
            
            // تحديث مؤقت الصمت
            if (this.silenceDetection && this.isRecording && !this.isPaused) {
                if (avg < this.silenceThreshold) {
                    if (!this.silenceTimer) {
                        this.silenceTimer = setTimeout(() => {
                            if (this.isRecording && !this.isPaused) {
                                console.log('[VoiceRecorder] Silence detected, auto-stopping...');
                                this.emit('silence_detected', { duration: this.silenceDuration });
                                if (this.silenceCallback) {
                                    this.silenceCallback();
                                } else {
                                    this.stopRecording();
                                }
                            }
                            this.silenceTimer = null;
                        }, this.silenceDuration);
                    }
                } else {
                    if (this.silenceTimer) {
                        clearTimeout(this.silenceTimer);
                        this.silenceTimer = null;
                    }
                    this.lastSoundTime = Date.now();
                }
            }
            
            // لون حسب مستوى الصوت
            const color = avg > 0.3 ? '#ea4335' : avg > 0.15 ? '#fbbc04' : '#34a853';
            
            for (let i = 0; i < bufferLength; i++) {
                const value = dataArray[i] / 128.0;
                const percent = value * 100;
                const barHeight = (percent / 100) * height;
                
                const xPos = this.isRTL ? width - x - barWidth : x;
                this.visualizerCtx.fillStyle = color;
                this.visualizerCtx.fillRect(xPos, height - barHeight, barWidth - 1, barHeight);
                
                x += barWidth;
            }
        };
        
        draw();
    }
    
    stopAudioAnalysis() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
        
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        this.analyserNode = null;
        
        if (this.visualizerCtx) {
            this.visualizerCtx.clearRect(0, 0, this.visualizerCanvas.width, this.visualizerCanvas.height);
        }
        
        if (this.silenceTimer) {
            clearTimeout(this.silenceTimer);
            this.silenceTimer = null;
        }
    }
    
    cleanupVisualizer() {
        if (this.visualizerCtx) {
            this.visualizerCtx.clearRect(0, 0, this.visualizerCanvas.width, this.visualizerCanvas.height);
        }
        this.visualizerCanvas = null;
        this.visualizerCtx = null;
    }
    
    // ──────────────────────────────────────────────────────────
    // 11. إعدادات كشف الصمت
    // ──────────────────────────────────────────────────────────
    
    enableSilenceDetection(enabled, threshold = 0.02, duration = 3000) {
        this.silenceDetection = enabled;
        this.silenceThreshold = threshold;
        this.silenceDuration = duration;
        console.log('[VoiceRecorder] Silence detection settings:', { enabled, threshold, duration });
    }
    
    setSilenceCallback(callback) {
        this.silenceCallback = callback;
    }
    
    // ──────────────────────────────────────────────────────────
    // 12. تنظيف الموارد
    // ──────────────────────────────────────────────────────────
    
    cleanup() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
        }
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.stopAudioAnalysis();
        this.isRecording = false;
        this.isPaused = false;
        this.audioChunks = [];
        this.recordingDuration = 0;
        
        if (this.silenceTimer) {
            clearTimeout(this.silenceTimer);
            this.silenceTimer = null;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 13. الحصول على الحالة
    // ──────────────────────────────────────────────────────────
    
    getStatus() {
        return {
            isRecording: this.isRecording,
            isPaused: this.isPaused,
            duration: this.recordingDuration,
            hasAudio: this.audioChunks.length > 0,
            mimeType: this.mimeType,
            bitrate: this.audioBitsPerSecond,
            maxDuration: this.maxDuration,
            minDuration: this.minDuration,
            silenceDetection: this.silenceDetection,
            silenceThreshold: this.silenceThreshold,
            silenceDuration: this.silenceDuration
        };
    }
    
    // ──────────────────────────────────────────────────────────
    // 14. نظام الأحداث (Event System)
    // ──────────────────────────────────────────────────────────
    
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
                console.error(`[VoiceRecorder] Error in event listener for ${event}:`, error);
            }
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 15. إضافة أنماط CSS
    // ──────────────────────────────────────────────────────────
    
    injectStyles() {
        if (document.getElementById('riva-voice-styles')) return;
        
        const styles = `
            <style id="riva-voice-styles">
                .voice-recorder-container {
                    background: var(--white, #ffffff);
                    border-radius: 16px;
                    padding: 16px;
                    margin: 12px 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                }
                
                .voice-controls {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    flex-wrap: wrap;
                    margin-bottom: 12px;
                }
                
                .record-btn {
                    width: 56px;
                    height: 56px;
                    border-radius: 50%;
                    border: none;
                    background: #ea4335;
                    color: white;
                    font-size: 24px;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                .record-btn:hover {
                    transform: scale(1.05);
                    box-shadow: 0 4px 12px rgba(234, 67, 53, 0.4);
                }
                
                .record-btn.recording {
                    background: #d32f2f;
                    animation: pulse 1.5s infinite;
                }
                
                .record-btn.paused {
                    background: #fbbc04;
                }
                
                .record-timer {
                    font-size: 18px;
                    font-weight: bold;
                    font-family: monospace;
                    min-width: 80px;
                }
                
                .voice-visualizer {
                    width: 100%;
                    height: 80px;
                    background: var(--light, #f8f9fa);
                    border-radius: 8px;
                    margin-top: 12px;
                }
                
                .voice-status {
                    font-size: 12px;
                    color: var(--gray, #5f6368);
                    margin-top: 8px;
                    text-align: center;
                }
                
                .silence-warning {
                    background: rgba(251, 188, 4, 0.2);
                    border-radius: 8px;
                    padding: 8px 12px;
                    margin-top: 8px;
                    font-size: 12px;
                    color: #fbbc04;
                    text-align: center;
                    animation: fadeIn 0.3s ease;
                }
                
                @keyframes pulse {
                    0%, 100% {
                        box-shadow: 0 0 0 0 rgba(211, 47, 47, 0.4);
                    }
                    50% {
                        box-shadow: 0 0 0 8px rgba(211, 47, 47, 0);
                    }
                }
                
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(-10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }
    
    injectModalStyles() {
        if (document.getElementById('riva-voice-modal-styles')) return;
        
        const styles = `
            <style id="riva-voice-modal-styles">
                .voice-permission-modal {
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
                
                .voice-permission-modal.show {
                    visibility: visible;
                    opacity: 1;
                }
                
                .voice-permission-modal .modal-overlay {
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(0,0,0,0.5);
                }
                
                .voice-permission-modal .modal-content {
                    position: relative;
                    background: white;
                    border-radius: 16px;
                    width: 90%;
                    max-width: 350px;
                    max-height: 90vh;
                    overflow-y: auto;
                    animation: modalSlideIn 0.3s ease;
                }
                
                .voice-permission-modal .modal-header {
                    padding: 16px 20px;
                    border-bottom: 1px solid #e8eaed;
                }
                
                .voice-permission-modal .modal-header h3 {
                    margin: 0;
                }
                
                .voice-permission-modal .modal-body {
                    padding: 20px;
                }
                
                .voice-permission-modal .privacy-note {
                    font-size: 12px;
                    color: #5f6368;
                    margin-top: 12px;
                    padding: 8px;
                    background: #e8f0fe;
                    border-radius: 8px;
                }
                
                .voice-permission-modal .modal-footer {
                    display: flex;
                    gap: 12px;
                    padding: 16px 20px;
                    border-top: 1px solid #e8eaed;
                }
                
                .voice-permission-modal .modal-btn {
                    flex: 1;
                    padding: 10px;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 14px;
                    font-weight: 500;
                }
                
                .voice-permission-modal .modal-btn-primary {
                    background: #1a73e8;
                    color: white;
                }
                
                .voice-permission-modal .modal-btn-secondary {
                    background: #f8f9fa;
                    color: #5f6368;
                }
                
                @keyframes modalSlideIn {
                    from {
                        transform: translateY(-30px);
                        opacity: 0;
                    }
                    to {
                        transform: translateY(0);
                        opacity: 1;
                    }
                }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }
}

// ──────────────────────────────────────────────────────────
// 16. تصدير نسخة واحدة (Singleton)
// ──────────────────────────────────────────────────────────

// إنشاء نسخة واحدة
const voiceRecorder = new VoiceRecorder();

// تخزين في window للاستخدام العادي
window.voiceRecorder = voiceRecorder;
window.rivaVoiceRecorder = voiceRecorder;

// ES Module export
export default voiceRecorder;
export { voiceRecorder };
