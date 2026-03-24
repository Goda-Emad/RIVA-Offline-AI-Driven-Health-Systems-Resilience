/**
 * auth.js
 * =======
 * RIVA Health Platform - Authentication Module
 * وحدة المصادقة والصلاحيات
 * 
 * المسؤوليات:
 * - تسجيل الدخول (Login)
 * - تسجيل الخروج (Logout)
 * - إدارة التوكن (JWT)
 * - توجيه المستخدمين حسب الدور (Doctor, Patient, Nurse, Admin)
 * - حماية الصفحات (Route Guards)
 * 
 * المسار: web-app/src/static/js/auth.js
 * 
 * التحسينات:
 * - دعم RTL للغة العربية
 * - تخزين آمن للتوكن
 * - إعادة توجيه ذكية بعد تسجيل الدخول
 * - حماية الصفحات الحساسة
 * 
 * الإصدار: 4.2.1
 */

// ──────────────────────────────────────────────────────────
// 1. كلاس Authentication
// ──────────────────────────────────────────────────────────

class Authentication {
    constructor() {
        this.apiClient = window.rivaClient || null;
        this.tokenKey = 'riva_token';
        this.userKey = 'riva_user';
        this.redirectAfterLogin = '/';
        this.loginAttempts = 0;
        this.maxLoginAttempts = 5;
        this.lockoutTime = 15 * 60 * 1000; // 15 دقيقة
        
        this.isRTL = document.documentElement.dir === 'rtl' || document.body.dir === 'rtl';
        
        this.init();
    }
    
    async init() {
        console.log('[Auth] Initializing...');
        
        if (!this.apiClient) {
            window.addEventListener('riva-client-ready', () => {
                this.apiClient = window.rivaClient;
                console.log('[Auth] API Client connected');
                this.setupTokenFromStorage();
            });
        } else {
            this.setupTokenFromStorage();
        }
        
        this.injectStyles();
        console.log('[Auth] Initialized');
    }
    
    setupTokenFromStorage() {
        const token = localStorage.getItem(this.tokenKey);
        if (token && this.apiClient) {
            this.apiClient.setToken(token);
            console.log('[Auth] Token restored from storage');
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 2. تسجيل الدخول
    // ──────────────────────────────────────────────────────────
    
    async login(username, password, options = {}) {
        // التحقق من محاولات الدخول الفاشلة
        if (this.isLockedOut()) {
            const remainingTime = this.getLockoutRemainingTime();
            throw new Error(this.isRTL 
                ? `تم تجاوز عدد المحاولات المسموح بها. حاول مرة أخرى بعد ${Math.ceil(remainingTime / 60)} دقيقة.`
                : `Too many failed attempts. Try again in ${Math.ceil(remainingTime / 60)} minutes.`);
        }
        
        try {
            // محاولة تسجيل الدخول
            const response = await this.apiClient.request('/auth/login', {
                method: 'POST',
                body: JSON.stringify({ username, password })
            });
            
            if (response.success && response.token) {
                // حفظ التوكن
                this.apiClient.setToken(response.token);
                localStorage.setItem(this.tokenKey, response.token);
                
                // حفظ معلومات المستخدم
                const userData = {
                    id: response.user_id,
                    username: response.username,
                    name: response.name,
                    role: response.role,
                    email: response.email,
                    permissions: response.permissions || []
                };
                localStorage.setItem(this.userKey, JSON.stringify(userData));
                
                // إعادة تعيين محاولات الدخول الفاشلة
                this.resetLoginAttempts();
                
                console.log(`[Auth] Login successful: ${username} (${response.role})`);
                
                // إطلاق حدث نجاح تسجيل الدخول
                window.dispatchEvent(new CustomEvent('riva-login-success', {
                    detail: { user: userData }
                }));
                
                // التوجيه حسب الدور
                const redirectTo = options.redirect || this.getRedirectUrlByRole(response.role);
                if (redirectTo) {
                    window.location.href = redirectTo;
                }
                
                return { success: true, user: userData, redirect: redirectTo };
            }
            
            throw new Error(response.message || 'Login failed');
            
        } catch (error) {
            console.error('[Auth] Login failed:', error);
            
            // زيادة عدد المحاولات الفاشلة
            this.incrementLoginAttempts();
            
            throw error;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 3. تسجيل الخروج
    // ──────────────────────────────────────────────────────────
    
    async logout(redirectTo = '/login') {
        try {
            // محاولة إبلاغ الخادم بتسجيل الخروج (اختياري)
            if (this.apiClient && this.apiClient.token) {
                await this.apiClient.request('/auth/logout', {
                    method: 'POST'
                }).catch(() => {});
            }
        } finally {
            // مسح التوكن من التخزين
            this.apiClient?.clearToken();
            localStorage.removeItem(this.tokenKey);
            localStorage.removeItem(this.userKey);
            
            console.log('[Auth] Logged out');
            
            window.dispatchEvent(new CustomEvent('riva-logout-success'));
            
            if (redirectTo) {
                window.location.href = redirectTo;
            }
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 4. الحصول على معلومات المستخدم الحالي
    // ──────────────────────────────────────────────────────────
    
    getCurrentUser() {
        const userStr = localStorage.getItem(this.userKey);
        if (!userStr) return null;
        
        try {
            return JSON.parse(userStr);
        } catch {
            return null;
        }
    }
    
    getCurrentRole() {
        const user = this.getCurrentUser();
        return user?.role || null;
    }
    
    isAuthenticated() {
        const token = localStorage.getItem(this.tokenKey);
        const user = this.getCurrentUser();
        return !!(token && user);
    }
    
    // ──────────────────────────────────────────────────────────
    // 5. التحقق من الصلاحيات
    // ──────────────────────────────────────────────────────────
    
    hasPermission(permission) {
        const user = this.getCurrentUser();
        if (!user) return false;
        
        // Admin لديه جميع الصلاحيات
        if (user.role === 'admin' || user.role === 'supervisor') {
            return true;
        }
        
        return user.permissions?.includes(permission) || false;
    }
    
    hasRole(role) {
        const userRole = this.getCurrentRole();
        if (Array.isArray(role)) {
            return role.includes(userRole);
        }
        return userRole === role;
    }
    
    // ──────────────────────────────────────────────────────────
    // 6. التوجيه حسب الدور
    // ──────────────────────────────────────────────────────────
    
    getRedirectUrlByRole(role) {
        const routes = {
            'doctor': '/09_doctor_dashboard.html',
            'nurse': '/03_triage.html',
            'pharmacist': '/prescriptions.html',
            'admin': '/09_doctor_dashboard.html',
            'supervisor': '/15_combined_dashboard.html',
            'school_nurse': '/11_school_dashboard.html',
            'patient': '/02_chatbot.html',
            'pregnant': '/10_mother_dashboard.html',
            'default': '/'
        };
        
        return routes[role] || routes.default;
    }
    
    // ──────────────────────────────────────────────────────────
    // 7. حماية الصفحات (Route Guard)
    // ──────────────────────────────────────────────────────────
    
    guardPage(requiredRoles = null, requiredPermissions = null, redirectTo = '/login') {
        if (!this.isAuthenticated()) {
            console.warn('[Auth] Unauthenticated access, redirecting to login');
            window.location.href = redirectTo;
            return false;
        }
        
        if (requiredRoles && !this.hasRole(requiredRoles)) {
            console.warn('[Auth] Insufficient role, redirecting');
            window.location.href = redirectTo;
            return false;
        }
        
        if (requiredPermissions) {
            const perms = Array.isArray(requiredPermissions) ? requiredPermissions : [requiredPermissions];
            for (const perm of perms) {
                if (!this.hasPermission(perm)) {
                    console.warn('[Auth] Missing permission:', perm);
                    window.location.href = redirectTo;
                    return false;
                }
            }
        }
        
        return true;
    }
    
    // ──────────────────────────────────────────────────────────
    // 8. إدارة محاولات تسجيل الدخول
    // ──────────────────────────────────────────────────────────
    
    incrementLoginAttempts() {
        this.loginAttempts++;
        localStorage.setItem('riva_login_attempts', this.loginAttempts.toString());
        localStorage.setItem('riva_login_lockout_time', Date.now().toString());
    }
    
    resetLoginAttempts() {
        this.loginAttempts = 0;
        localStorage.removeItem('riva_login_attempts');
        localStorage.removeItem('riva_login_lockout_time');
    }
    
    isLockedOut() {
        const attempts = parseInt(localStorage.getItem('riva_login_attempts') || '0');
        const lockoutTime = parseInt(localStorage.getItem('riva_login_lockout_time') || '0');
        
        if (attempts >= this.maxLoginAttempts) {
            const timeElapsed = Date.now() - lockoutTime;
            if (timeElapsed < this.lockoutTime) {
                return true;
            }
            this.resetLoginAttempts();
        }
        
        return false;
    }
    
    getLockoutRemainingTime() {
        const lockoutTime = parseInt(localStorage.getItem('riva_login_lockout_time') || '0');
        const timeElapsed = Date.now() - lockoutTime;
        return Math.max(0, this.lockoutTime - timeElapsed) / 1000;
    }
    
    // ──────────────────────────────────────────────────────────
    // 9. تحديث الملف الشخصي
    // ──────────────────────────────────────────────────────────
    
    async updateProfile(userData) {
        if (!this.isAuthenticated()) {
            throw new Error(this.isRTL ? 'يجب تسجيل الدخول أولاً' : 'Please login first');
        }
        
        try {
            const response = await this.apiClient.request('/auth/profile', {
                method: 'PUT',
                body: JSON.stringify(userData)
            });
            
            if (response.success) {
                // تحديث بيانات المستخدم المخزنة
                const currentUser = this.getCurrentUser();
                const updatedUser = { ...currentUser, ...response.user };
                localStorage.setItem(this.userKey, JSON.stringify(updatedUser));
                
                window.dispatchEvent(new CustomEvent('riva-profile-updated', {
                    detail: { user: updatedUser }
                }));
                
                return { success: true, user: updatedUser };
            }
            
            throw new Error(response.message || 'Update failed');
            
        } catch (error) {
            console.error('[Auth] Profile update failed:', error);
            throw error;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 10. تغيير كلمة المرور
    // ──────────────────────────────────────────────────────────
    
    async changePassword(currentPassword, newPassword) {
        if (!this.isAuthenticated()) {
            throw new Error(this.isRTL ? 'يجب تسجيل الدخول أولاً' : 'Please login first');
        }
        
        try {
            const response = await this.apiClient.request('/auth/change-password', {
                method: 'POST',
                body: JSON.stringify({ current_password: currentPassword, new_password: newPassword })
            });
            
            if (response.success) {
                console.log('[Auth] Password changed successfully');
                return { success: true };
            }
            
            throw new Error(response.message || 'Password change failed');
            
        } catch (error) {
            console.error('[Auth] Password change failed:', error);
            throw error;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 11. إعادة تعيين كلمة المرور (طلب)
    // ──────────────────────────────────────────────────────────
    
    async requestPasswordReset(email) {
        try {
            const response = await this.apiClient.request('/auth/forgot-password', {
                method: 'POST',
                body: JSON.stringify({ email })
            });
            
            return { success: true, message: response.message };
            
        } catch (error) {
            console.error('[Auth] Password reset request failed:', error);
            throw error;
        }
    }
    
    async resetPassword(token, newPassword) {
        try {
            const response = await this.apiClient.request('/auth/reset-password', {
                method: 'POST',
                body: JSON.stringify({ token, new_password: newPassword })
            });
            
            return { success: true, message: response.message };
            
        } catch (error) {
            console.error('[Auth] Password reset failed:', error);
            throw error;
        }
    }
    
    // ──────────────────────────────────────────────────────────
    // 12. عرض واجهة تسجيل الدخول
    // ──────────────────────────────────────────────────────────
    
    renderLoginForm(containerId, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.warn(`[Auth] Container ${containerId} not found`);
            return;
        }
        
        const isRTL = this.isRTL;
        
        const html = `
            <div class="auth-login-container">
                <div class="login-card">
                    <div class="login-header">
                        <img src="/static/assets/images/logo.png" alt="RIVA" class="login-logo">
                        <h2>${isRTL ? 'تسجيل الدخول إلى ريفا' : 'Login to RIVA'}</h2>
                        <p>${isRTL ? 'منصة الصحة الذكية' : 'Smart Health Platform'}</p>
                    </div>
                    
                    <form id="login-form" class="login-form">
                        <div class="form-group">
                            <label for="username">${isRTL ? 'اسم المستخدم' : 'Username'}</label>
                            <input type="text" id="username" name="username" class="form-control" 
                                   placeholder="${isRTL ? 'أدخل اسم المستخدم' : 'Enter username'}" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="password">${isRTL ? 'كلمة المرور' : 'Password'}</label>
                            <input type="password" id="password" name="password" class="form-control" 
                                   placeholder="${isRTL ? 'أدخل كلمة المرور' : 'Enter password'}" required>
                            <button type="button" class="password-toggle" id="toggle-password">
                                👁️
                            </button>
                        </div>
                        
                        <div class="form-options">
                            <label class="checkbox-label">
                                <input type="checkbox" id="remember-me"> 
                                <span>${isRTL ? 'تذكرني' : 'Remember me'}</span>
                            </label>
                            <a href="/forgot-password" class="forgot-link">${isRTL ? 'نسيت كلمة المرور؟' : 'Forgot password?'}</a>
                        </div>
                        
                        <div id="login-error" class="login-error" style="display: none;"></div>
                        
                        <button type="submit" class="login-btn" id="login-submit">
                            <span class="btn-text">${isRTL ? 'تسجيل الدخول' : 'Login'}</span>
                            <span class="btn-spinner" style="display: none;">⏳</span>
                        </button>
                    </form>
                    
                    <div class="login-footer">
                        <p>${isRTL ? 'ليس لديك حساب؟' : "Don't have an account?"} 
                            <a href="/register">${isRTL ? 'إنشاء حساب جديد' : 'Create account'}</a>
                        </p>
                    </div>
                </div>
            </div>
        `;
        
        container.innerHTML = html;
        
        // إضافة مستمعي الأحداث
        this.setupLoginFormEvents();
    }
    
    setupLoginFormEvents() {
        const form = document.getElementById('login-form');
        const usernameInput = document.getElementById('username');
        const passwordInput = document.getElementById('password');
        const togglePassword = document.getElementById('toggle-password');
        const rememberCheckbox = document.getElementById('remember-me');
        const submitBtn = document.getElementById('login-submit');
        const errorDiv = document.getElementById('login-error');
        
        if (!form) return;
        
        // إظهار/إخفاء كلمة المرور
        if (togglePassword) {
            togglePassword.addEventListener('click', () => {
                const type = passwordInput.type === 'password' ? 'text' : 'password';
                passwordInput.type = type;
                togglePassword.textContent = type === 'password' ? '👁️' : '🙈';
            });
        }
        
        // استرجاع اسم المستخدم المخزن
        if (rememberCheckbox && localStorage.getItem('riva_remembered_user')) {
            usernameInput.value = localStorage.getItem('riva_remembered_user');
            rememberCheckbox.checked = true;
        }
        
        // معالجة submit
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const username = usernameInput.value.trim();
            const password = passwordInput.value;
            
            if (!username || !password) {
                this.showLoginError(errorDiv, this.isRTL ? 'يرجى إدخال اسم المستخدم وكلمة المرور' : 'Please enter username and password');
                return;
            }
            
            // تعطيل الزر أثناء المعالجة
            submitBtn.disabled = true;
            const btnText = submitBtn.querySelector('.btn-text');
            const btnSpinner = submitBtn.querySelector('.btn-spinner');
            if (btnText) btnText.style.display = 'none';
            if (btnSpinner) btnSpinner.style.display = 'inline';
            
            try {
                const result = await this.login(username, password);
                
                if (result.success) {
                    // حفظ اسم المستخدم إذا تم اختيار "تذكرني"
                    if (rememberCheckbox.checked) {
                        localStorage.setItem('riva_remembered_user', username);
                    } else {
                        localStorage.removeItem('riva_remembered_user');
                    }
                }
                
            } catch (error) {
                this.showLoginError(errorDiv, error.message);
                submitBtn.disabled = false;
                if (btnText) btnText.style.display = 'inline';
                if (btnSpinner) btnSpinner.style.display = 'none';
            }
        });
    }
    
    showLoginError(errorDiv, message) {
        if (!errorDiv) return;
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
        
        setTimeout(() => {
            errorDiv.style.display = 'none';
        }, 5000);
    }
    
    // ──────────────────────────────────────────────────────────
    // 13. إضافة أنماط CSS
    // ──────────────────────────────────────────────────────────
    
    injectStyles() {
        if (document.getElementById('riva-auth-styles')) return;
        
        const isRTL = this.isRTL;
        
        const styles = `
            <style id="riva-auth-styles">
                .auth-login-container {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px;
                }
                
                .login-card {
                    background: white;
                    border-radius: 24px;
                    padding: 40px;
                    width: 100%;
                    max-width: 420px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    animation: slideUp 0.5s ease;
                }
                
                .login-header {
                    text-align: center;
                    margin-bottom: 32px;
                }
                
                .login-logo {
                    width: 80px;
                    height: 80px;
                    margin-bottom: 16px;
                }
                
                .login-header h2 {
                    margin: 0 0 8px;
                    font-size: 24px;
                    color: #202124;
                }
                
                .login-header p {
                    margin: 0;
                    color: #5f6368;
                    font-size: 14px;
                }
                
                .login-form .form-group {
                    margin-bottom: 20px;
                    position: relative;
                }
                
                .login-form label {
                    display: block;
                    margin-bottom: 8px;
                    font-weight: 500;
                    color: #202124;
                }
                
                .login-form .form-control {
                    width: 100%;
                    padding: 12px 16px;
                    border: 1px solid #e8eaed;
                    border-radius: 12px;
                    font-size: 16px;
                    transition: all 0.2s ease;
                }
                
                .login-form .form-control:focus {
                    outline: none;
                    border-color: #1a73e8;
                    box-shadow: 0 0 0 3px rgba(26,115,232,0.1);
                }
                
                .password-toggle {
                    position: absolute;
                    left: ${isRTL ? '12px' : 'auto'};
                    right: ${isRTL ? 'auto' : '12px'};
                    top: 70%;
                    transform: translateY(-50%);
                    background: none;
                    border: none;
                    cursor: pointer;
                    font-size: 18px;
                    opacity: 0.6;
                }
                
                .form-options {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 24px;
                    font-size: 14px;
                }
                
                .checkbox-label {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    cursor: pointer;
                }
                
                .forgot-link {
                    color: #1a73e8;
                    text-decoration: none;
                }
                
                .forgot-link:hover {
                    text-decoration: underline;
                }
                
                .login-error {
                    background: rgba(234, 67, 53, 0.1);
                    color: #ea4335;
                    padding: 12px;
                    border-radius: 12px;
                    margin-bottom: 20px;
                    font-size: 14px;
                    text-align: center;
                }
                
                .login-btn {
                    width: 100%;
                    padding: 14px;
                    background: #1a73e8;
                    color: white;
                    border: none;
                    border-radius: 12px;
                    font-size: 16px;
                    font-weight: 500;
                    cursor: pointer;
                    transition: all 0.2s ease;
                }
                
                .login-btn:hover {
                    background: #0d47a1;
                    transform: translateY(-2px);
                }
                
                .login-btn:disabled {
                    opacity: 0.7;
                    cursor: not-allowed;
                    transform: none;
                }
                
                .login-footer {
                    text-align: center;
                    margin-top: 24px;
                    padding-top: 24px;
                    border-top: 1px solid #e8eaed;
                    font-size: 14px;
                    color: #5f6368;
                }
                
                .login-footer a {
                    color: #1a73e8;
                    text-decoration: none;
                }
                
                @keyframes slideUp {
                    from {
                        opacity: 0;
                        transform: translateY(20px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }
                
                @media (max-width: 768px) {
                    .login-card {
                        padding: 24px;
                    }
                    
                    .login-header h2 {
                        font-size: 20px;
                    }
                }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }
}

// ──────────────────────────────────────────────────────────
// 14. تصدير نسخة واحدة (Singleton)
// ──────────────────────────────────────────────────────────

// إنشاء نسخة واحدة وتخزينها في window
const auth = new Authentication();

// تخزين في window للاستخدام العادي
window.auth = auth;
window.rivaAuth = auth;

// ES Module export
export default auth;
export { aut
