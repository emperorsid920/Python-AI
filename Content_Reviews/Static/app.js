// Main Application JavaScript
class ContentModerationApp {
    constructor() {
        this.currentTab = 'dashboard';
        this.currentReviewId = null;
        this.processing = false;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadSystemStatus();
        this.loadDashboardData();

        // Initialize tabs
        this.showTab('dashboard');

        // Start status polling
        this.startStatusPolling();
    }

    setupEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tabName = e.target.dataset.tab;
                this.showTab(tabName);
            });
        });

        // Modal close
        document.querySelector('.close').addEventListener('click', () => {
            this.closeModal();
        });

        // Close modal when clicking outside
        document.getElementById('review-modal').addEventListener('click', (e) => {
            if (e.target.id === 'review-modal') {
                this.closeModal();
            }
        });

        // Download button
        document.getElementById('download-btn').addEventListener('click', () => {
            this.downloadResults();
        });

        // Filter changes
        document.getElementById('sentiment-filter').addEventListener('change', () => {
            this.loadResults();
        });

        document.getElementById('spam-filter').addEventListener('change', () => {
            this.loadResults();
        });

        document.getElementById('moderation-filter').addEventListener('change', () => {
            this.loadResults();
        });
    }

    showTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(tabName).classList.add('active');

        this.currentTab = tabName;

        // Load tab-specific data
        switch (tabName) {
            case 'dashboard':
                this.loadDashboardData();
                break;
            case 'results':
                this.loadResults();
                break;
            case 'analytics':
                this.loadAnalytics();
                break;
        }
    }

    async loadSystemStatus() {
        try {
            const response = await fetch('/health');
            const data = await response.json();

            // Update status indicators
            this.updateStatusIndicator('db-status', data.database_status);
            this.updateStatusIndicator('model-status', data.model_status);
            this.updateStatusIndicator('processing-status', this.processing ? 'processing' : 'ready');

        } catch (error) {
            console.error('Error loading system status:', error);
            this.updateStatusIndicator('db-status', 'error');
            this.updateStatusIndicator('model-status', 'error');
            this.updateStatusIndicator('processing-status', 'error');
        }
    }

    updateStatusIndicator(elementId, status) {
        const indicator = document.getElementById(elementId);
        indicator.className = 'status-indicator';

        switch (status) {
            case 'healthy':
            case 'ready':
            case 'loaded':
                indicator.classList.add('online');
                break;
            case 'processing':
            case 'loading':
                indicator.classList.add('warning');
                break;
            case 'error':
            case 'unhealthy':
            case 'failed':
                indicator.classList.add('offline');
                break;
            default:
                indicator.classList.add('offline');
        }
    }

    async loadDashboardData() {
        try {
            const response = await fetch('/analytics');
            const data = await response.json();

            // Update stats
            document.getElementById('total-reviews').textContent = data.total_reviews || 0;
            document.getElementById('approved-reviews').textContent = data.approved_count || 0;
            document.getElementById('flagged-reviews').textContent = data.flagged_count || 0;
            document.getElementById('pending-reviews').textContent = data.pending_count || 0;

            // Update activity feed
            this.updateActivityFeed(data);

        } catch (error) {
            console.error('Error loading dashboard data:', error);
            this.showToast('Error loading dashboard data', 'error');
        }
    }

    updateActivityFeed(data) {
        const feed = document.getElementById('activity-feed');
        const activities = [];

        if (data.recent_uploads) {
            data.recent_uploads.forEach(upload => {
                activities.push({
                    icon: '📁',
                    text: `Processed ${upload.count} reviews from ${upload.filename}`,
                    time: this.formatTime(upload.timestamp)
                });
            });
        }

        if (data.recent_moderations) {
            data.recent_moderations.forEach(mod => {
                activities.push({
                    icon: '✅',
                    text: `Review ${mod.review_id} marked as ${mod.status}`,
                    time: this.formatTime(mod.timestamp)
                });
            });
        }

        // Sort by time and show latest 10
        activities.sort((a, b) => new Date(b.time) - new Date(a.time));
        const recentActivities = activities.slice(0, 10);

        if (recentActivities.length === 0) {
            feed.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">📊</div>
                    <h3>No recent activity</h3>
                    <p>Upload some reviews to get started</p>
                </div>
            `;
        } else {
            feed.innerHTML = recentActivities.map(activity => `
                <div class="activity-item">
                    <div class="activity-icon">${activity.icon}</div>
                    <div class="activity-content">
                        <p>${activity.text}</p>
                        <span class="activity-time">${activity.time}</span>
                    </div>
                </div>
            `).join('');
        }
    }

    async loadResults(page = 1) {
        const sentimentFilter = document.getElementById('sentiment-filter').value;
        const spamFilter = document.getElementById('spam-filter').value;
        const moderationFilter = document.getElementById('moderation-filter').value;

        const params = new URLSearchParams({
            page: page,
            sentiment: sentimentFilter,
            spam: spamFilter,
            moderation_status: moderationFilter
        });

        try {
            const response = await fetch(`/results?${params}`);
            const data = await response.json();

            this.renderResults(data.reviews);
            this.renderPagination(data.pagination);

        } catch (error) {
            console.error('Error loading results:', error);
            this.showToast('Error loading results', 'error');
        }
    }

    renderResults(reviews) {
        const tbody = document.getElementById('results-tbody');

        if (reviews.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="7" class="text-center">
                        <div class="empty-state">
                            <div class="empty-state-icon">📋</div>
                            <h3>No reviews found</h3>
                            <p>Try adjusting your filters or upload some reviews</p>
                        </div>
                    </td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = reviews.map(review => `
            <tr>
                <td>${review.id}</td>
                <td class="review-text" title="${this.escapeHtml(review.text)}">
                    ${this.truncateText(this.escapeHtml(review.text), 100)}
                </td>
                <td>
                    <span class="badge badge-${this.getSentimentBadgeClass(review.sentiment)}">
                        ${review.sentiment}
                    </span>
                    <div class="score">${review.sentiment_score ? review.sentiment_score.toFixed(2) : 'N/A'}</div>
                </td>
                <td>
                    <span class="badge badge-${review.is_spam ? 'danger' : 'success'}">
                        ${review.is_spam ? 'Spam' : 'Not Spam'}
                    </span>
                    <div class="score">${review.spam_score ? review.spam_score.toFixed(2) : 'N/A'}</div>
                </td>
                <td>
                    <span class="badge badge-${this.getToxicityBadgeClass(review.toxicity_score)}">
                        ${review.toxicity_score > 0.7 ? 'High' : review.toxicity_score > 0.3 ? 'Medium' : 'Low'}
                    </span>
                    <div class="score">${review.toxicity_score ? review.toxicity_score.toFixed(2) : 'N/A'}</div>
                </td>
                <td>
                    <span class="badge badge-${this.getStatusBadgeClass(review.moderation_status)}">
                        ${review.moderation_status}
                    </span>
                </td>
                <td class="actions">
                    <button class="btn btn-small btn-primary" onclick="app.viewReview(${review.id})">
                        View
                    </button>
                </td>
            </tr>
        `).join('');
    }

    renderPagination(pagination) {
        const paginationDiv = document.getElementById('pagination');

        if (pagination.total_pages <= 1) {
            paginationDiv.innerHTML = '';
            return;
        }

        let paginationHTML = '';

        // Previous button
        if (pagination.current_page > 1) {
            paginationHTML += `<button onclick="app.loadResults(${pagination.current_page - 1})">Previous</button>`;
        }

        // Page numbers
        for (let i = 1; i <= pagination.total_pages; i++) {
            const activeClass = i === pagination.current_page ? 'active' : '';
            paginationHTML += `<button class="${activeClass}" onclick="app.loadResults(${i})">${i}</button>`;
        }

        // Next button
        if (pagination.current_page < pagination.total_pages) {
            paginationHTML += `<button onclick="app.loadResults(${pagination.current_page + 1})">Next</button>`;
        }

        paginationDiv.innerHTML = paginationHTML;
    }

    async viewReview(reviewId) {
        try {
            const response = await fetch(`/review/${reviewId}`);
            const review = await response.json();

            this.currentReviewId = reviewId;
            this.showReviewModal(review);

        } catch (error) {
            console.error('Error loading review:', error);
            this.showToast('Error loading review details', 'error');
        }
    }

    showReviewModal(review) {
        const modal = document.getElementById('review-modal');
        const details = document.getElementById('review-details');

        details.innerHTML = `
            <div class="review-detail">
                <h3>Review #${review.id}</h3>
                <div class="review-content">
                    <h4>Review Text:</h4>
                    <p class="review-text-full">${this.escapeHtml(review.text)}</p>
                </div>
                <div class="review-meta">
                    <div class="meta-item">
                        <strong>Author:</strong> ${review.author || 'Anonymous'}
                    </div>
                    <div class="meta-item">
                        <strong>Submitted:</strong> ${this.formatDate(review.created_at)}
                    </div>
                    <div class="meta-item">
                        <strong>Sentiment:</strong>
                        <span class="badge badge-${this.getSentimentBadgeClass(review.sentiment)}">
                            ${review.sentiment} (${review.sentiment_score ? review.sentiment_score.toFixed(2) : 'N/A'})
                        </span>
                    </div>
                    <div class="meta-item">
                        <strong>Spam Detection:</strong>
                        <span class="badge badge-${review.is_spam ? 'danger' : 'success'}">
                            ${review.is_spam ? 'Spam' : 'Not Spam'} (${review.spam_score ? review.spam_score.toFixed(2) : 'N/A'})
                        </span>
                    </div>
                    <div class="meta-item">
                        <strong>Toxicity:</strong>
                        <span class="badge badge-${this.getToxicityBadgeClass(review.toxicity_score)}">
                            ${review.toxicity_score > 0.7 ? 'High' : review.toxicity_score > 0.3 ? 'Medium' : 'Low'} (${review.toxicity_score ? review.toxicity_score.toFixed(2) : 'N/A'})
                        </span>
                    </div>
                    <div class="meta-item">
                        <strong>Current Status:</strong>
                        <span class="badge badge-${this.getStatusBadgeClass(review.moderation_status)}">
                            ${review.moderation_status}
                        </span>
                    </div>
                </div>
            </div>
        `;

        modal.style.display = 'block';
    }

    closeModal() {
        document.getElementById('review-modal').style.display = 'none';
        this.currentReviewId = null;
    }

    async updateModerationStatus(status) {
        if (!this.currentReviewId) return;

        try {
            const response = await fetch(`/moderate/${this.currentReviewId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ status: status })
            });

            if (response.ok) {
                this.showToast(`Review ${status} successfully`, 'success');
                this.closeModal();
                this.loadResults();
                this.loadDashboardData();
            } else {
                throw new Error('Failed to update status');
            }
        } catch (error) {
            console.error('Error updating moderation status:', error);
            this.showToast('Error updating review status', 'error');
        }
    }

    async downloadResults() {
        try {
            const response = await fetch('/download_processed');
            const blob = await response.blob();

            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'processed_reviews.csv';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

            this.showToast('Results downloaded successfully', 'success');
        } catch (error) {
            console.error('Error downloading results:', error);
            this.showToast('Error downloading results', 'error');
        }
    }

    startStatusPolling() {
        setInterval(() => {
            this.loadSystemStatus();
            if (this.processing) {
                this.checkProcessingStatus();
            }
        }, 5000); // Poll every 5 seconds
    }

    async checkProcessingStatus() {
        try {
            const response = await fetch('/status');
            const status = await response.json();

            if (status.processing) {
                this.processing = true;
                this.updateProcessingProgress(status);
            } else {
                this.processing = false;
                this.hideProgressSection();
                if (status.completed) {
                    this.showToast('File processing completed', 'success');
                    this.loadDashboardData();
                    this.loadResults();
                }
            }
        } catch (error) {
            console.error('Error checking processing status:', error);
        }
    }

    updateProcessingProgress(status) {
        const progressSection = document.getElementById('progress-section');
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        const progressPercent = document.getElementById('progress-percent');
        const processingDetails = document.getElementById('processing-details');

        progressSection.style.display = 'block';

        const percent = Math.round((status.processed_rows / status.total_rows) * 100);
        progressFill.style.width = `${percent}%`;
        progressPercent.textContent = `${percent}%`;

        progressText.textContent = status.current_step || 'Processing...';

        processingDetails.innerHTML = `
            <h4>Processing Details:</h4>
            <p>File: ${status.current_file || 'Unknown'}</p>
            <p>Processed: ${status.processed_rows || 0} of ${status.total_rows || 0} rows</p>
            <p>Status: ${status.status || 'Running'}</p>
            ${status.error ? `<p class="text-danger">Error: ${status.error}</p>` : ''}
        `;
    }

    hideProgressSection() {
        document.getElementById('progress-section').style.display = 'none';
    }

    showToast(message, type = 'info') {
        const toastContainer = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <strong>${type.charAt(0).toUpperCase() + type.slice(1)}</strong>
                <p>${message}</p>
            </div>
        `;

        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.remove();
        }, 5000);
    }

    // Utility functions
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }

    formatDate(dateString) {
        return new Date(dateString).toLocaleString();
    }

    formatTime(dateString) {
        return new Date(dateString).toLocaleTimeString();
    }

    getSentimentBadgeClass(sentiment) {
        switch (sentiment) {
            case 'positive': return 'success';
            case 'negative': return 'danger';
            case 'neutral': return 'secondary';
            default: return 'secondary';
        }
    }

    getToxicityBadgeClass(score) {
        if (score > 0.7) return 'danger';
        if (score > 0.3) return 'warning';
        return 'success';
    }

    getStatusBadgeClass(status) {
        switch (status) {
            case 'approved': return 'success';
            case 'rejected': return 'danger';
            case 'pending': return 'warning';
            case 'flagged': return 'danger';
            default: return 'secondary';
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ContentModerationApp();
});

// Make functions globally available
window.updateModerationStatus = function(status) {
    window.app.updateModerationStatus(status);
};