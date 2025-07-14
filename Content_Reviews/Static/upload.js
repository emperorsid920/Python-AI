// Upload functionality
class UploadManager {
    constructor() {
        this.uploadArea = document.getElementById('upload-area');
        this.fileInput = document.getElementById('file-input');
        this.progressSection = document.getElementById('progress-section');
        this.setupEventListeners();
    }

    setupEventListeners() {
        // File input change
        this.fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.handleFileSelection(file);
            }
        });

        // Drag and drop
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('drag-over');
        });

        this.uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('drag-over');
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('drag-over');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelection(files[0]);
            }
        });

        // Click to upload
        this.uploadArea.addEventListener('click', (e) => {
            if (e.target.tagName !== 'BUTTON') {
                this.fileInput.click();
            }
        });
    }

    handleFileSelection(file) {
        // Validate file
        if (!this.validateFile(file)) {
            return;
        }

        // Show file info
        this.showFileInfo(file);

        // Upload file
        this.uploadFile(file);
    }

    validateFile(file) {
        // Check file type
        if (!file.name.toLowerCase().endsWith('.csv')) {
            app.showToast('Please select a CSV file', 'error');
            return false;
        }

        // Check file size (16MB limit)
        const maxSize = 16 * 1024 * 1024; // 16MB
        if (file.size > maxSize) {
            app.showToast('File size exceeds 16MB limit', 'error');
            return false;
        }

        return true;
    }

    showFileInfo(file) {
        const fileSize = this.formatFileSize(file.size);
        const uploadIcon = this.uploadArea.querySelector('.upload-icon');
        const uploadText = this.uploadArea.querySelector('h3');
        const uploadSubtext = this.uploadArea.querySelector('p');

        uploadIcon.textContent = '📄';
        uploadText.textContent = file.name;
        uploadSubtext.textContent = `Size: ${fileSize} | Ready to upload`;

        // Add upload button
        const existingBtn = this.uploadArea.querySelector('.upload-btn');
        if (existingBtn) {
            existingBtn.remove();
        }

        const uploadBtn = document.createElement('button');
        uploadBtn.className = 'btn btn-primary upload-btn';
        uploadBtn.textContent = 'Upload File';
        uploadBtn.onclick = () => this.uploadFile(file);

        this.uploadArea.appendChild(uploadBtn);
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            // Show progress
            this.showProgress();
            app.processing = true;

            // Upload request
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();

                // Hide progress
                this.hideProgress();

                // Handle successful upload
                this.handleUploadSuccess(result);

                app.showToast('File uploaded successfully!', 'success');
            } else {
                throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
            }

        } catch (error) {
            console.error('Upload error:', error);
            this.hideProgress();
            app.showToast(`Upload failed: ${error.message}`, 'error');
        } finally {
            app.processing = false;
        }
    }

    showProgress() {
        this.progressSection.style.display = 'block';
        this.progressSection.innerHTML = `
            <div class="progress-bar">
                <div class="progress-fill" style="width: 0%"></div>
            </div>
            <p>Uploading file...</p>
        `;

        // Simulate progress animation
        const progressFill = this.progressSection.querySelector('.progress-fill');
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) {
                clearInterval(interval);
                progress = 90;
            }
            progressFill.style.width = `${progress}%`;
        }, 200);

        // Store interval for cleanup
        this.progressInterval = interval;
    }

    hideProgress() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }

        const progressFill = this.progressSection.querySelector('.progress-fill');
        if (progressFill) {
            progressFill.style.width = '100%';
        }

        setTimeout(() => {
            this.progressSection.style.display = 'none';
        }, 500);
    }

    handleUploadSuccess(result) {
        // Update UI to show upload success
        const uploadIcon = this.uploadArea.querySelector('.upload-icon');
        const uploadText = this.uploadArea.querySelector('h3');
        const uploadSubtext = this.uploadArea.querySelector('p');
        const uploadBtn = this.uploadArea.querySelector('.upload-btn');

        uploadIcon.textContent = '✅';
        uploadText.textContent = 'Upload Complete';
        uploadSubtext.textContent = `${result.filename} processed successfully`;

        if (uploadBtn) {
            uploadBtn.remove();
        }

        // Store upload result for other parts of the app
        if (typeof app !== 'undefined') {
            app.uploadResult = result;

            // Trigger any post-upload processing
            if (app.onUploadComplete) {
                app.onUploadComplete(result);
            }
        }

        // Reset upload area after delay
        setTimeout(() => {
            this.resetUploadArea();
        }, 3000);
    }

    resetUploadArea() {
        const uploadIcon = this.uploadArea.querySelector('.upload-icon');
        const uploadText = this.uploadArea.querySelector('h3');
        const uploadSubtext = this.uploadArea.querySelector('p');
        const uploadBtn = this.uploadArea.querySelector('.upload-btn');

        uploadIcon.textContent = '📁';
        uploadText.textContent = 'Drop your CSV file here';
        uploadSubtext.textContent = 'or click to browse files';

        if (uploadBtn) {
            uploadBtn.remove();
        }

        // Reset file input
        this.fileInput.value = '';

        // Remove drag-over class if present
        this.uploadArea.classList.remove('drag-over');
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';

        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));

        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Method to programmatically trigger file selection
    selectFile() {
        this.fileInput.click();
    }

    // Method to clear current upload state
    clearUpload() {
        this.resetUploadArea();
        this.hideProgress();

        if (typeof app !== 'undefined') {
            app.uploadResult = null;
        }
    }
}