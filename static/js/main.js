/**
 * MyVoiceger Flask App - Modern JavaScript Functions
 */

document.addEventListener('DOMContentLoaded', function() {
    // === ファイルアップロード機能 ===
    initializeFileUploads();
    
    // === ステップ進行状況 ===
    updateStepIndicators();
    
    // === フォームバリデーション ===
    initializeFormValidation();
    
    // === Ajax ポーリング ===
    initializeStatusPolling();
    
    // === アニメーション ===
    initializeAnimations();
    
    // === ツールチップ ===
    initializeTooltips();
});

/**
 * ファイルアップロードの初期化
 */
function initializeFileUploads() {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach(input => {
        const label = input.nextElementSibling;
        const info = label.nextElementSibling;
        
        input.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                updateFileInfo(label, info, file);
                validateFile(file, input);
            }
        });
        
        // ドラッグ&ドロップ機能
        label.addEventListener('dragover', function(e) {
            e.preventDefault();
            e.stopPropagation();
            label.classList.add('drag-over');
        });
        
        label.addEventListener('dragleave', function(e) {
            e.preventDefault();
            e.stopPropagation();
            label.classList.remove('drag-over');
        });
        
        label.addEventListener('drop', function(e) {
            e.preventDefault();
            e.stopPropagation();
            label.classList.remove('drag-over');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                input.files = files;
                const file = files[0];
                updateFileInfo(label, info, file);
                validateFile(file, input);
            }
        });
    });
}

/**
 * ファイル情報の更新
 */
function updateFileInfo(label, info, file) {
    const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
    
    info.innerHTML = `
        <p><strong>${file.name}</strong> (${sizeMB}MB)</p>
        <p class="upload-pending"><i class="fas fa-clock"></i> アップロード待</p>
    `;
    
    // ファイル拡張子による色分け
    const ext = file.name.split('.').pop().toLowerCase();
    const allowedExts = ['mp3', 'wav', 'm4a', 'flac'];
    
    if (allowedExts.includes(ext)) {
        info.style.borderColor = 'var(--success-color)';
    } else {
        info.style.borderColor = 'var(--danger-color)';
        info.innerHTML += '<p class="upload-error"><i class="fas fa-exclamation-triangle"></i> 未対応のファイル形式</p>';
    }
}

/**
 * ファイルのバリデーション
 */
function validateFile(file, input) {
    const allowedTypes = ['audio/mpeg', 'audio/wav', 'audio/mp4', 'audio/flac'];
    const maxSize = 100 * 1024 * 1024; // 100MB
    
    // ファイルサイズのチェック
    if (file.size > maxSize) {
        showNotification('ファイルサイズが大きすぎます（最大100MB）', 'error');
        input.value = '';
        return false;
    }
    
    // ファイルタイプのチェック
    if (!allowedTypes.includes(file.type)) {
        showNotification('サポートされていないファイル形式です', 'error');
        input.value = '';
        return false;
    }
    
    return true;
}

/**
 * ステップインジケーターの更新
 */
function updateStepIndicators() {
    const steps = document.querySelectorAll('.step');
    const sections = document.querySelectorAll('.step-section');
    
    // スクロール時のアクティブ更新
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const sectionId = entry.target.id;
                const stepNum = parseInt(sectionId.replace('step', ''));
                
                steps.forEach((step, index) => {
                    step.classList.toggle('active', index < stepNum);
                });
            }
        });
    }, {
        threshold: 0.5
    });
    
    sections.forEach(section => {
        observer.observe(section);
    });
}

/**
 * フォームバリデーションの初期化
 */
function initializeFormValidation() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            if (!validateForm(form)) {
                e.preventDefault();
                return false;
            }
            
            // ローディング状態にする
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.classList.add('loading');
                submitBtn.disabled = true;
            }
        });
    });
}

/**
 * フォームのバリデーション
 */
function validateForm(form) {
    const requiredFields = form.querySelectorAll('[required]');
    let isValid = true;
    
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            showFieldError(field, 'この項目は必須です');
            isValid = false;
        } else {
            clearFieldError(field);
        }
    });
    
    // ファイルアップロードの場合
    const fileInputs = form.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        if (input.hasAttribute('required') && !input.files.length) {
            showFieldError(input.nextElementSibling, 'ファイルを選択してください');
            isValid = false;
        }
    });
    
    return isValid;
}

/**
 * フィールドエラーの表示
 */
function showFieldError(field, message) {
    clearFieldError(field);
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'field-error';
    errorDiv.style.cssText = `
        color: var(--danger-color);
        font-size: var(--font-size-sm);
        margin-top: var(--spacing-xs);
        display: flex;
        align-items: center;
        gap: var(--spacing-xs);
    `;
    errorDiv.innerHTML = `<i class="fas fa-exclamation-circle"></i>${message}`;
    
    field.parentNode.appendChild(errorDiv);
    field.style.borderColor = 'var(--danger-color)';
}

/**
 * フィールドエラーのクリア
 */
function clearFieldError(field) {
    const existingError = field.parentNode.querySelector('.field-error');
    if (existingError) {
        existingError.remove();
    }
    field.style.borderColor = '';
}

/**
 * Ajax ポーリングの初期化
 */
function initializeStatusPolling() {
    // ステータス更新の定期ポーリング
    setInterval(fetchStatus, 5000);
}

/**
 * ステータスの取得
 */
function fetchStatus() {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            updateUIFromStatus(data);
        })
        .catch(error => {
            console.log('Status fetch failed:', error);
        });
}

/**
 * ステータスからUIを更新
 */
function updateUIFromStatus(data) {
    const stepCompleted = data.step_completed || {};
    
    // ステップインジケーターの更新
    const steps = document.querySelectorAll('.step');
    Object.keys(stepCompleted).forEach((step, index) => {
        if (steps[index]) {
            steps[index].classList.toggle('active', stepCompleted[step]);
        }
    });
    
    // セクションの有効化/無効化
    const sections = document.querySelectorAll('.step-section');
    sections.forEach((section, index) => {
        const isEnabled = index === 0 || stepCompleted[Object.keys(stepCompleted)[index - 1]];
        section.classList.toggle('disabled', !isEnabled);
    });
}

/**
 * アニメーションの初期化
 */
function initializeAnimations() {
    // フェードインアニメーション
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // アニメーション対象の要素を監視
    const animatedElements = document.querySelectorAll('.step-section, .upload-card, .result-card');
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
}

/**
 * ツールチップの初期化
 */
function initializeTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', showTooltip);
        element.addEventListener('mouseleave', hideTooltip);
    });
}

/**
 * ツールチップの表示
 */
function showTooltip(e) {
    const text = e.target.getAttribute('data-tooltip');
    
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.textContent = text;
    tooltip.style.cssText = `
        position: absolute;
        background: var(--gray-800);
        color: var(--white);
        padding: var(--spacing-sm) var(--spacing-md);
        border-radius: var(--radius-md);
        font-size: var(--font-size-sm);
        z-index: 1000;
        white-space: nowrap;
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.3s ease;
    `;
    
    document.body.appendChild(tooltip);
    
    const rect = e.target.getBoundingClientRect();
    tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
    tooltip.style.top = rect.top - tooltip.offsetHeight - 8 + 'px';
    
    setTimeout(() => tooltip.style.opacity = '1', 10);
    
    e.target._tooltip = tooltip;
}

/**
 * ツールチップの非表示
 */
function hideTooltip(e) {
    const tooltip = e.target._tooltip;
    if (tooltip) {
        tooltip.style.opacity = '0';
        setTimeout(() => {
            if (tooltip.parentNode) {
                tooltip.parentNode.removeChild(tooltip);
            }
        }, 300);
    }
}

/**
 * 通知の表示
 */
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        top: var(--spacing-lg);
        right: var(--spacing-lg);
        padding: var(--spacing-md) var(--spacing-lg);
        border-radius: var(--radius-lg);
        color: var(--white);
        font-weight: 500;
        z-index: 1000;
        box-shadow: var(--shadow-lg);
        transform: translateX(100%);
        transition: transform 0.3s ease;
        max-width: 400px;
    `;
    
    const bgColor = type === 'success' ? 'var(--success-color)' :
                   type === 'error' ? 'var(--danger-color)' :
                   type === 'warning' ? 'var(--warning-color)' :
                   'var(--info-color)';
    notification.style.background = bgColor;
    
    notification.innerHTML = `
        <i class="fas fa-${getIconForType(type)}"></i>
        <span style="margin-left: var(--spacing-sm)">${message}</span>
    `;
    
    document.body.appendChild(notification);
    
    // 表示アニメーション
    setTimeout(() => notification.style.transform = 'translateX(0)', 100);
    
    // 自動削除
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 5000);
}

/**
 * 通知タイプに対応するアイコンを取得
 */
function getIconForType(type) {
    const icons = {
        success: 'check-circle',
        error: 'exclamation-circle',
        warning: 'exclamation-triangle',
        info: 'info-circle'
    };
    return icons[type] || 'info-circle';
}

/**
 * プログレスバーの更新
 */
function updateProgress(percentage, message = '') {
    const progressBar = document.querySelector('.progress-bar');
    if (progressBar) {
        progressBar.style.width = percentage + '%';
        progressBar.setAttribute('aria-valuenow', percentage);
    }
    
    const progressText = document.querySelector('.progress-text');
    if (progressText) {
        progressText.textContent = message;
    }
}

/**
 * 確認ダイアログの表示
 */
function showConfirmDialog(message, callback) {
    const overlay = document.createElement('div');
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        z-index: 2000;
        display: flex;
        align-items: center;
        justify-content: center;
    `;
    
    const dialog = document.createElement('div');
    dialog.style.cssText = `
        background: var(--white);
        border-radius: var(--radius-lg);
        padding: var(--spacing-2xl);
        max-width: 400px;
        width: 90%;
        text-align: center;
        box-shadow: var(--shadow-xl);
    `;
    
    dialog.innerHTML = `
        <h3 style="margin-bottom: var(--spacing-lg); color: var(--gray-800);">
            ${message}
        </h3>
        <div style="display: flex; gap: var(--spacing-md); justify-content: center;">
            <button class="btn btn-secondary" onclick="this.closest('.confirm-overlay').remove()">キャンセル</button>
            <button class="btn btn-danger confirm-yes">確認</button>
        </div>
    `;
    
    overlay.className = 'confirm-overlay';
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);
    
    // クリックイベント
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) {
            overlay.remove();
        }
    });
    
    dialog.querySelector('.confirm-yes').addEventListener('click', () => {
        overlay.remove();
        if (callback) callback(true);
    });
}

/**
 * キーボードショートカット
 */
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter でフォーム送信
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const activeForm = document.activeElement.closest('form');
        if (activeForm) {
            const submitBtn = activeForm.querySelector('button[type="submit"]');
            if (submitBtn && !submitBtn.disabled) {
                submitBtn.click();
            }
        }
    }
    
    // Escape キーで通知を閉じる
    if (e.key === 'Escape') {
        const notifications = document.querySelectorAll('.notification');
        notifications.forEach(notification => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => notification.remove(), 300);
        });
    }
});

/**
 * レスポンシブ対応のヘルパー
 */
function isMobile() {
    return window.innerWidth <= 768;
}

function isTablet() {
    return window.innerWidth > 768 && window.innerWidth <= 1024;
}

/**
 * ローカルストレージヘルパー
 */
const StorageHelper = {
    set: function(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
        } catch (e) {
            console.warn('localStorage set failed:', e);
        }
    },
    
    get: function(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (e) {
            console.warn('localStorage get failed:', e);
            return defaultValue;
        }
    },
    
    remove: function(key) {
        try {
            localStorage.removeItem(key);
        } catch (e) {
            console.warn('localStorage remove failed:', e);
        }
    }
};

// === エクスポート ===
window.MyVoicegerApp = {
    showNotification,
    showConfirmDialog,
    updateProgress,
    validateForm,
    StorageHelper,
    isMobile,
    isTablet
};