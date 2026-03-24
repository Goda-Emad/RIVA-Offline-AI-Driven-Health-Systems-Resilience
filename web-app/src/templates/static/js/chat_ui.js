/**
 * chat_ui.js
 * ==========
 * RIVA Health Platform - Chat UI Module
 * واجهة المستخدم للشات بوت
 * 
 * المسؤوليات:
 * - عرض فقاعات المحادثة (Chat Bubbles)
 * - معالجة إرسال الرسائل (نص/صوت)
 * - عرض الردود كلمة بكلمة (Streaming/Typing Effect)
 * - Auto-scroll للمحادثة
 * - دعم التسجيل الصوتي
 * - دعم RTL للغة العربية
 * 
 * المسار: web-app/src/static/js/chat_ui.js
 * 
 * التحسينات:
 * - Typing Effect للردود
 * - Auto-scroll تلقائي
 * - دعم الصوت عبر voice_recorder
 * - حفظ حالة المحادثة
 * - عرض وقت الرسائل
 * 
 * الإصدار: 4.2.1
 */

// ──────────────────────────────────────────────────────────
// 1. كلاس Chat UI
// ──────────────────────────────────────────────────────────

class ChatUI {
    constructor() {
        this.apiClient = window.rivaClient || null;
        this.voiceRecorder = window.voiceRecorder || null;
        this.offlineManager = window.offlineManager || null;
        
        this.sessionId = null;
        this.isStreaming = false;
        this.currentStream = null;
        this.messageHistory = [];
        
        // عناصر DOM
        this.elements = {
            container: null,
            messagesContainer: null,
            messageInput: null,
            sendBtn: null,
            voiceBtn: null,
            micBtn: null,
            typingIndicator: null,
            offlineWarning: null
        };
        
        this.isRTL = document.documentElement.dir === 'rtl' || document.body.dir === 'rtl';
        
        this.init();
    }
    
    async init() {
        console.log('[ChatUI] Initializing...');
        
        // انتظار تحميل API Client
        if (!this.apiClient) {
            window.addEventListener('riva-client-ready', () => {
                this.apiClient = window.rivaClient;
                console.log('[ChatUI] API Client connected');
                this.loadSession();
            });
        } else {
            this.loadSession();
        }
        
        // انتظار تحميل Voice Recorder
        if (!this.voiceRecorder) {
            window.addEventListener('voice-recorder-ready', () => {
                this.voiceRecorder = window.voiceRecorder;
                console.log('[ChatUI] Voice Recorder connected');
            });
        } else {
            this.voiceRecorder = window.voiceRecorder;
        }
        
        this.injectStyles();
        this.setupElements();
        this.setupEventListeners();
        
        console.log('[ChatUI] Initialized');
    }
    
    // ──────────────────────────────────────────────────────────
    // 2. إعداد العناصر
    // ──────────────────────────────────────────────────────────
    
    setupElements() {
        // البحث عن العناصر المطلوبة
        this.elements.container = document.getElementById('chat-container');
        this.elements.messagesContainer = document.getElementById('chat-messages');
        this.elements.messageInput = document.getElementById('message-input');
        this.elements.sendBtn = document.getElementById('send-btn');
        this.elements.voiceBtn = document.getElementById('voice-btn');
        this.elements.micBtn = document.getElementById('mic-btn');
        this.elements.typingIndicator = document.getElementById('typing-indicator');
        this.elements.offlineWarning = document.getElementById('offline-warning');
        
        // إنشاء العناصر إذا لم تكن موجودة
        if (!this.elements.messagesContainer && this.elements.container) {
            this.createChatElements();
        }
    }
    
    createChatElements() {
        const container = this.elements.container;
        if (!container) return;
        
        container.innerHTML = `
            <div class="chat-header">
                <h2>🤖 ${this.isRTL ? 'ريفا - مساعدك الطبي الذكي' : 'RIVA - Your Smart Medical Assistant'}</h2>
                <p>${this.isRTL ? 'اسأل عن أي شيء طبي - بنتكلم مصري وبنفهمك كويس' : 'Ask anything medical - We speak Egyptian Arabic'}</p>
            </div>
            
            <div class="chat-messages" id="chat-messages"></div>
            
            <div class="typing-indicator" id="typing-indicator" style="display: none;">
                <span></span><span></span><span></span>
            </div>
            
            <div class="offline-warning" id="offline-warning" style="display: none;">
                <span>📡</span>
                <span>${this.isRTL ? 'لا يوجد اتصال بالإنترنت - سيتم حفظ رسالتك وإرسالها لاحقاً' : 'No internet connection - Your message will be saved and sent later'}</span>
            </div>
            
            <div class="chat-input-area">
                <div class="input-actions">
                    <button class="voice-btn" id="voice-btn" title="${this.isRTL ? 'تسجيل صوت' : 'Voice Record'}">🎤</button>
                    <button class="mic-btn" id="mic-btn" title="${this.isRTL ? 'تحدث مباشرة' : 'Speak'}">🎙️</button>
                </div>
                <textarea id="message-input" placeholder="${this.isRTL ? 'اكتب سؤالك هنا...' : 'Type your question here...'}" rows="2"></textarea>
                <button class="send-btn" id="send-btn">
                    <span>${this.isRTL ? 'إرسال' : 'Send'}</span>
                    <span class="send-icon">➤</span>
                </button>
            </div>
        `;
        
        // إعادة تعيين المراجع
        this.elements.messagesContainer = document.getElementById('chat-messages');
        this.elements.messageInput = document.getElementById('message-input');
        this.elements.sendBtn = document.getElementById('send-btn');
        this.elements.voiceBtn = document.getElementById('voice-btn');
        this.elements.micBtn = document.getElementById('mic-btn');
        this.elements.typingIndicator = document.getElementById('typing-indicator');
        this.elements.offlineWarning = document.getElementById('offline-warning');
    }
    
    // ──────────────────────────────────────────────────────────
    // 3. إدارة الجلسة
    // ──────────────────────────────────────────────────────────
    
    loadSession() {
        const savedSession = localStorage.getItem('riva_chat_session_id');
        if (savedSession) {
            this.sessionId = savedSession;
        } else {
            this.sessionId = this.generateUUID();
            localStorage.setItem('riva_chat_session_id', this.sessionId);
        }
        
        console.log('[ChatUI] Session ID:', this.sessionId);
    }
    
    generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }
    
    // ──────────────────────────────────────────────────────────
    // 4. إدارة الرسائل
    // ──────────────────────────────────────────────────────────
    
    addUserMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message';
        messageDiv.innerHTML = `
            <div class="message-avatar">👤</div>
            <div class="message-content">
                <div class="message-text">${this.escapeHtml(text)}</div>
                <div class="message-time">${this.getFormattedTime()}</div>
            </div>
        `;
        
        this.elements.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        // حفظ في التاريخ
        this.messageHistory.push({ role: 'user', content: text, timestamp: Date.now() });
        
        return messageDiv;
    }
    
    addBotMessage(text, isStreaming = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        messageDiv.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <div class="message-text">${isStreaming ? '' : this.escapeHtml(text)}</div>
                <div class="message-time">${this.getFormattedTime()}</div>
            </div>
        `;
        
        this.elements.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        // حفظ في التاريخ
        if (!isStreaming) {
            this.messageHistory.push({ role: 'assistant', content: text, timestamp: Date.now() });
        }
        
        return messageDiv;
    }
    
    updateStreamingMessage(messageDiv, text, isComplete = false) {
        const textDiv = messageDiv.querySelector('.message-text');
        if (textDiv) {
            textDiv.innerHTML = this.escapeHtml(text);
            if (isComplete) {
                // إضافة وقت الإرسال
                const timeDiv = messageDiv.querySelector('.message-time');
                if (timeDiv) {
                    timeDiv.textContent = this.getFormattedTime();
                }
                // حفظ في التاريخ
                this.messageHistory.push({ role: 'assistant', content: text, timestamp: Date.now() });
            }
        }
        this.scrollToBottom();
    }
    
    showTypingIndicator() {
        if (this.elements.typingIndicator) {
            this.elements.typingIndicator.style.display = 'flex';
            this.scrollToBottom();
        }
    }
    
    hideTypingIndicator() {
        if (this.elements.typingIndicator) {
            this.elements.typingIndicator.style.display = 'none';
        }
    }
    
    showOfflineWarning(show = true) {
        if (this.elements.offlineWarning) {
            this.elements.offlineWarning.style.display = show ? 'flex' : 'none';
        }
    }
    
    scrollToBottom() {
        if (this.elements.messagesContainer) {
            this.elements.messagesContainer.scrollTop = this.elements.messagesContainer.scrollHeight;
        }
    }
    
    getFormattedTime() {
        const now = new Date();
        return `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
    }
    
    // ──────────────────────────────────────────────────────────
    // 5. إرسال الرسائل
    // ──────────────────────────────────────────────────────────
    
    async sendMessage(text) {
        if (!text || text.trim() === '') return;
        
        // تعطيل الإدخال أثناء الإرسال
        this.setInputEnabled(false);
        
        // عرض رسالة المستخدم
        this.addUserMessage(text);
        
        // مسح حقل الإدخال
        if (this.elements.messageInput) {
            this.elements.messageInput.value = '';
            this.elements.messageInput.style.height = 'auto';
        }
        
        // التحقق من الاتصال بالإنترنت
        const isOnline = this.offlineManager ? this.offlineManager.isOnline : navigator.onLine;
        
        if (!isOnline) {
            this.showOfflineWarning(true);
            // حفظ الرسالة للمزامنة لاحقاً
            await this.saveOfflineMessage(text);
            this.setInputEnabled(true);
            return;
        }
        
        this.showOfflineWarning(false);
        
        try {
            // إظهار مؤشر الكتابة
            this.showTypingIndicator();
            
            // إرسال الرسالة والحصول على الرد
            await this.streamMessage(text);
            
        } catch (error) {
            console.error('[ChatUI] Send message failed:', error);
            this.addBotMessage(this.isRTL ? 'عذراً، حدث خطأ. يرجى المحاولة مرة أخرى.' : 'Sorry, an error occurred. Please try again.');
        } finally {
            this.hideTypingIndicator();
            this.setInputEnabled(true);
        }
    }
    
    async streamMessage(text) {
        const url = `/api/chat/message`;
        
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Session-ID': this.sessionId
                },
                body: JSON.stringify({
                    message: text,
                    session_id: this.sessionId,
                    stream: true
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            // إنشاء رسالة البوت (فارغة في البداية)
            const messageDiv = this.addBotMessage('', true);
            let fullText = '';
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const token = decoder.decode(value);
                fullText += token;
                this.updateStreamingMessage(messageDiv, fullText);
            }
            
            // تحديث الرسالة كاملة
            this.updateStreamingMessage(messageDiv, fullText, true);
            
        } catch (error) {
            console.error('[ChatUI] Stream error:', error);
            throw error;
        }
    }
    
    async sendVoiceMessage(audioBlob) {
        if (!audioBlob) return;
        
        this.setInputEnabled(false);
        this.showTypingIndicator();
        
        try {
            // تحويل الصوت إلى نص
            const formData = new FormData();
            formData.append('file', audioBlob, 'recording.webm');
            formData.append('language', 'ar');
            
            const transcribeResponse = await fetch('/api/voice/transcribe', {
                method: 'POST',
                body: formData
            });
            
            const transcribeResult = await transcribeResponse.json();
            
            if (transcribeResult.text) {
                // إرسال النص المستخرج إلى الشات
                await this.sendMessage(transcribeResult.text);
            } else {
                this.addBotMessage(this.isRTL ? 'عذراً، لم أتمكن من فهم الصوت. يرجى المحاولة مرة أخرى.' : 'Sorry, I could not understand the audio. Please try again.');
            }
            
        } catch (error) {
            console.error('[ChatUI] Voice message failed:', error);
            this.addBotMessage(this.isRTL ? 'حدث خطأ في معالجة الصوت.' : 'Error processing audio.');
        } finally {
            this.hideTypingIndicator();
            this.setInputEnabled(true);
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 6. تخزين الرسائل غير المتصلة
    // ──────────────────────────────────────────────────────────
    
    async saveOfflineMessage(text) {
        const offlineMessages = JSON.parse(localStorage.getItem('riva_offline_messages') || '[]');
        offlineMessages.push({
            text: text,
            timestamp: Date.now(),
            sessionId: this.sessionId
        });
        localStorage.setItem('riva_offline_messages', JSON.stringify(offlineMessages));
        
        // إظهار رسالة للمستخدم
        this.addBotMessage(this.isRTL ? '📡 تم حفظ رسالتك. سيتم إرسالها تلقائياً عند عودة الاتصال.' : '📡 Your message has been saved. It will be sent automatically when connection returns.');
    }
    
    async syncOfflineMessages() {
        const offlineMessages = JSON.parse(localStorage.getItem('riva_offline_messages') || '[]');
        if (offlineMessages.length === 0) return;
        
        console.log(`[ChatUI] Syncing ${offlineMessages.length} offline messages`);
        
        for (const msg of offlineMessages) {
            try {
                await this.streamMessage(msg.text);
            } catch (error) {
                console.error('[ChatUI] Failed to sync offline message:', error);
                return; // توقف إذا فشل أحد الرسائل
            }
        }
        
        // مسح الرسائل بعد المزامنة الناجحة
        localStorage.removeItem('riva_offline_messages');
        this.addBotMessage(this.isRTL ? '✅ تم إرسال جميع الرسائل المخزنة.' : '✅ All saved messages have been sent.');
    }
    
    // ──────────────────────────────────────────────────────────
    // 7. التسجيل الصوتي
    // ──────────────────────────────────────────────────────────
    
    async startVoiceRecording() {
        if (!this.voiceRecorder) {
            console.warn('[ChatUI] Voice recorder not available');
            return;
        }
        
        // تغيير لون زر التسجيل
        if (this.elements.voiceBtn) {
            this.elements.voiceBtn.classList.add('recording');
            this.elements.voiceBtn.textContent = '🔴';
        }
        
        const result = await this.voiceRecorder.startRecording('voice-visualizer', () => {
            // كشف الصمت - إيقاف تلقائي
            this.stopVoiceRecording();
        });
        
        if (!result.success) {
            console.error('[ChatUI] Failed to start recording:', result.error);
            if (this.elements.voiceBtn) {
                this.elements.voiceBtn.classList.remove('recording');
                this.elements.voiceBtn.textContent = '🎤';
            }
        }
    }
    
    async stopVoiceRecording() {
        if (!this.voiceRecorder) return;
        
        const result = await this.voiceRecorder.stopRecording();
        
        // إعادة زر التسجيل
        if (this.elements.voiceBtn) {
            this.elements.voiceBtn.classList.remove('recording');
            this.elements.voiceBtn.textContent = '🎤';
        }
        
        if (result.success && result.blob) {
            await this.sendVoiceMessage(result.blob);
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 8. إعداد الأحداث
    // ──────────────────────────────────────────────────────────
    
    setupEventListeners() {
        // إرسال الرسالة
        if (this.elements.sendBtn) {
            this.elements.sendBtn.addEventListener('click', () => {
                const text = this.elements.messageInput?.value;
                if (text && text.trim()) {
                    this.sendMessage(text);
                }
            });
        }
        
        // إدخال النص (Enter)
        if (this.elements.messageInput) {
            this.elements.messageInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    const text = this.elements.messageInput.value;
                    if (text && text.trim()) {
                        this.sendMessage(text);
                    }
                }
            });
            
            // Auto-resize textarea
            this.elements.messageInput.addEventListener('input', () => {
                this.elements.messageInput.style.height = 'auto';
                this.elements.messageInput.style.height = Math.min(this.elements.messageInput.scrollHeight, 120) + 'px';
            });
        }
        
        // زر التسجيل الصوتي
        if (this.elements.voiceBtn) {
            this.elements.voiceBtn.addEventListener('click', async () => {
                if (this.voiceRecorder && this.voiceRecorder.isRecording) {
                    await this.stopVoiceRecording();
                } else {
                    await this.startVoiceRecording();
                }
            });
        }
        
        // زر الميكروفون (تسجيل مباشر)
        if (this.elements.micBtn) {
            this.elements.micBtn.addEventListener('click', async () => {
                if (this.voiceRecorder && this.voiceRecorder.isRecording) {
                    await this.stopVoiceRecording();
                } else {
                    await this.startVoiceRecording();
                }
            });
        }
        
        // مراقبة حالة الاتصال
        window.addEventListener('online', () => {
            this.showOfflineWarning(false);
            this.syncOfflineMessages();
        });
        
        window.addEventListener('offline', () => {
            this.showOfflineWarning(true);
        });
        
        // تحميل الرسائل المخزنة عند بدء التشغيل
        setTimeout(() => {
            this.syncOfflineMessages();
        }, 2000);
    }
    
    setInputEnabled(enabled) {
        if (this.elements.messageInput) {
            this.elements.messageInput.disabled = !enabled;
        }
        if (this.elements.sendBtn) {
            this.elements.sendBtn.disabled = !enabled;
        }
        if (this.elements.voiceBtn) {
            this.elements.voiceBtn.disabled = !enabled;
        }
        if (this.elements.micBtn) {
            this.elements.micBtn.disabled = !enabled;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 9. دوال مساعدة
    // ──────────────────────────────────────────────────────────
    
    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    clearChat() {
        if (this.elements.messagesContainer) {
            this.elements.messagesContainer.innerHTML = '';
        }
        this.messageHistory = [];
        
        // إضافة رسالة ترحيب
        this.addBotMessage(this.isRTL 
            ? 'أهلاً بك! أنا ريفا، مساعدك الطبي الذكي. اسأل عن أي حاجة، أنا هنا أساعدك.'
            : 'Welcome! I am RIVA, your smart medical assistant. Ask me anything, I am here to help.');
    }
    
    // ──────────────────────────────────────────────────────────
    // 10. إضافة أنماط CSS
    // ──────────────────────────────────────────────────────────
    
    injectStyles() {
        if (document.getElementById('riva-chat-ui-styles')) return;
        
        const isRTL = this.isRTL;
        
        const styles = `
            <style id="riva-chat-ui-styles">
                .chat-container {
                    display: flex;
                    flex-direction: column;
                    height: 100vh;
                    max-height: 100vh;
                    background: var(--light, #f8f9fa);
                }
                
                .chat-header {
                    background: var(--primary, #1a73e8);
                    color: white;
                    padding: 20px;
                    text-align: center;
                    flex-shrink: 0;
                }
                
                .chat-header h2 {
                    margin: 0 0 8px;
                    font-size: 20px;
                }
                
                .chat-header p {
                    margin: 0;
                    font-size: 14px;
                    opacity: 0.9;
                }
                
                .chat-messages {
                    flex: 1;
                    overflow-y: auto;
                    padding: 20px;
                    display: flex;
                    flex-direction: column;
                    gap: 16px;
                }
                
                .message {
                    display: flex;
                    gap: 12px;
                    max-width: 85%;
                    animation: fadeIn 0.3s ease;
                }
                
                .user-message {
                    align-self: flex-end;
                    flex-direction: row-reverse;
                }
                
                .bot-message {
                    align-self: flex-start;
                }
                
                .message-avatar {
                    width: 36px;
                    height: 36px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 18px;
                    flex-shrink: 0;
                }
                
                .user-message .message-avatar {
                    background: var(--primary, #1a73e8);
                }
                
                .bot-message .message-avatar {
                    background: var(--secondary, #34a853);
                }
                
                .message-content {
                    background: white;
                    border-radius: 18px;
                    padding: 10px 16px;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                    max-width: 100%;
                }
                
                .user-message .message-content {
                    background: var(--primary, #1a73e8);
                    color: white;
                }
                
                .message-text {
                    word-wrap: break-word;
                    line-height: 1.5;
                }
                
                .message-time {
                    font-size: 10px;
                    margin-top: 4px;
                    opacity: 0.7;
                    text-align: ${isRTL ? 'right' : 'left'};
                }
                
                .typing-indicator {
                    display: flex;
                    gap: 4px;
                    padding: 12px 20px;
                    align-self: flex-start;
                }
                
                .typing-indicator span {
                    width: 8px;
                    height: 8px;
                    background: var(--gray, #5f6368);
                    border-radius: 50%;
                    animation: typing 1.4s infinite;
                }
                
                .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
                .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
                
                @keyframes typing {
                    0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
                    30% { transform: translateY(-8px); opacity: 1; }
                }
                
                .offline-warning {
                    background: rgba(251, 188, 4, 0.2);
                    color: #fbbc04;
                    padding: 8px 16px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                    font-size: 12px;
                    flex-shrink: 0;
                }
                
                .chat-input-area {
                    display: flex;
                    gap: 12px;
                    padding: 16px;
                    background: white;
                    border-top: 1px solid var(--gray-lighter, #e8eaed);
                    flex-shrink: 0;
                }
                
                .input-actions {
                    display: flex;
                    gap: 8px;
                }
                
                .voice-btn, .mic-btn {
                    width: 44px;
                    height: 44px;
                    border-radius: 50%;
                    border: none;
                    background: var(--light, #f8f9fa);
                    cursor: pointer;
                    font-size: 18px;
                    transition: all 0.2s ease;
                }
                
                .voice-btn:hover, .mic-btn:hover {
                    background: var(--gray-lighter, #e8eaed);
                }
                
                .voice-btn.recording {
                    background: #ea4335;
                    color: white;
                    animation: pulse 1.5s infinite;
                }
                
                .chat-input-area textarea {
                    flex: 1;
                    padding: 12px;
                    border: 1px solid var(--gray-lighter, #e8eaed);
                    border-radius: 24px;
                    resize: none;
                    font-family: inherit;
                    font-size: 14px;
                    line-height: 1.4;
                }
                
                .chat-input-area textarea:focus {
                    outline: none;
                    border-color: var(--primary, #1a73e8);
                }
                
                .send-btn {
                    padding: 0 20px;
                    background: var(--primary, #1a73e8);
                    color: white;
                    border: none;
                    border-radius: 24px;
                    cursor: pointer;
                    font-size: 14px;
                    font-weight: 500;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    transition: all 0.2s ease;
                }
                
                .send-btn:hover {
                    background: var(--primary-dark, #0d47a1);
                    transform: translateY(-2px);
                }
                
                .send-btn:disabled {
                    opacity: 0.6;
                    cursor: not-allowed;
                }
                
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                
                @keyframes pulse {
                    0%, 100% { box-shadow: 0 0 0 0 rgba(234, 67, 53, 0.4); }
                    50% { box-shadow: 0 0 0 8px rgba(234, 67, 53, 0); }
                }
                
                @media (max-width: 768px) {
                    .message {
                        max-width: 95%;
                    }
                    
                    .chat-input-area {
                        gap: 8px;
                        padding: 12px;
                    }
                    
                    .voice-btn, .mic-btn {
                        width: 40px;
                        height: 40px;
                    }
                    
                    .send-btn {
                        padding: 0 16px;
                    }
                }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }
}

// ──────────────────────────────────────────────────────────
// 11. تصدير نسخة واحدة (Singleton)
// ──────────────────────────────────────────────────────────

// ──────────────────────────────────────────────────────────
// 11. تصدير نسخة واحدة (Singleton)
// ──────────────────────────────────────────────────────────

let chatUI = null;

function getChatUI() {
    if (!chatUI) {
        chatUI = new ChatUI();
        // تخزين في window
        window.chatUI = chatUI;
        window.rivaChatUI = chatUI;
    }
    return chatUI;
}

// تهيئة تلقائية إذا كان عنصر الشات موجوداً
if (document.getElementById('chat-container')) {
    getChatUI();
}

// ES Module exports
export default getChatUI;
export { getChatUI, chatUI };
