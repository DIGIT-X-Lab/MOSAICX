// MOSAICX WebApp - Ultra Modern Frontend JavaScript
// Enhanced with smooth animations, modern UX patterns, and robust error handling

// Global state management
const AppState = {
    currentTab: 'schema',
    isLoading: false,
    generatedSchemas: []
};

// Schema Storage
const SchemaStore = {
    addSchema(name, content, prompt, filename = null) {
        const timestamp = new Date().toISOString();
        const id = Date.now().toString();
        
        // Generate server filename if not provided
        if (!filename) {
            const sanitizedDescription = prompt.toLowerCase()
                .replace(/[^a-z0-9\s]/g, '')
                .replace(/\s+/g, '_')
                .substring(0, 50);
            const dateStr = new Date().toISOString().replace(/[-:]/g, '').replace(/T.*/, '').substring(0, 8) + '_' + 
                           new Date().toTimeString().replace(/[:.]/g, '').substring(0, 6);
            filename = `generatedmodel_${sanitizedDescription}_${dateStr}.py`;
        }
        
        const schema = { id, name, content, prompt, timestamp, filename };
        AppState.generatedSchemas.push(schema);
        this.updateSchemaDropdowns();
        try { localStorage.setItem('mosaicx_schemas', JSON.stringify(AppState.generatedSchemas)); } catch (e) {}
        return schema.id;
    },
    loadSchemas() {
        try { 
            const stored = localStorage.getItem('mosaicx_schemas');
            if (stored) { AppState.generatedSchemas = JSON.parse(stored); this.updateSchemaDropdowns(); }
        } catch (e) {}
    },
    updateSchemaDropdowns() {
        const dropdown = document.getElementById('extractSchema');
        if (dropdown) {
            dropdown.innerHTML = '<option value="">Select a previously generated schema...</option>';
            AppState.generatedSchemas.forEach(schema => {
                const option = document.createElement('option');
                option.value = schema.id;
                option.textContent = `${schema.name} (${new Date(schema.timestamp).toLocaleDateString()})`;
                dropdown.appendChild(option);
            });
        }
    },
    getSchema(id) { return AppState.generatedSchemas.find(s => s.id === id); }
};

// API Configuration
const API_BASE = '/api/v1';
const ENDPOINTS = {
    generateSchema: `${API_BASE}/generate-schema`,
    extractPDF: `${API_BASE}/extract-pdf`,
    summarizeFiles: `${API_BASE}/summarize-files`
};

// Utility Functions
const Utils = {
    // Show loading state with modern spinner
    showLoading(message = 'Processing...', containerId = null) {
        AppState.isLoading = true;
        const spinner = `
            <div class="loading-container" style="display: flex; align-items: center; justify-content: center; padding: 2rem; gap: 1rem;">
                <div class="spinner"></div>
                <span style="color: var(--text-secondary); font-weight: 500;">${message}</span>
            </div>
        `;
        
        if (containerId) {
            const container = document.getElementById(containerId);
            if (container) {
                container.innerHTML = spinner;
                container.classList.remove('hidden');
            }
        }
        return spinner;
    },

    // Hide loading state
    hideLoading(containerId = null) {
        AppState.isLoading = false;
        if (containerId) {
            const container = document.getElementById(containerId);
            if (container) {
                // Clear the loading content and hide container
                container.innerHTML = '';
                container.classList.add('hidden');
            }
        }
    },

    // Show modern notification
    showNotification(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        const colors = {
            success: 'var(--accent-primary)',
            error: 'var(--accent-tertiary)',
            info: 'var(--accent-secondary)',
            warning: 'var(--accent-tertiary)'
        };
        
        notification.style.cssText = `
            position: fixed;
            top: 2rem;
            right: 2rem;
            background: var(--bg-card);
            border: 1px solid ${colors[type]};
            border-radius: var(--radius-lg);
            padding: 1rem 1.5rem;
            color: var(--text-primary);
            box-shadow: var(--shadow-deep);
            backdrop-filter: blur(20px);
            z-index: 1000;
            animation: slideInRight 0.3s ease-out;
            max-width: 400px;
            font-weight: 500;
        `;
        
        notification.innerHTML = `
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="color: ${colors[type]}; font-size: 1.2rem;">
                    ${type === 'success' ? '✅' : type === 'error' ? '❌' : type === 'warning' ? '⚠️' : 'ℹ️'}
                </span>
                <span>${message}</span>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-in forwards';
            setTimeout(() => notification.remove(), 300);
        }, duration);
    },

    // Format file size
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    // Animate element appearance
    animateIn(element) {
        if (!element) return;
        element.style.opacity = '0';
        element.style.transform = 'translateY(20px)';
        element.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
        
        requestAnimationFrame(() => {
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        });
    }
};

// Tab Management
function showTab(tabName) {
    // Update state
    AppState.currentTab = tabName;
    
    // Update tab buttons with smooth transitions
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(btn => {
        btn.classList.remove('active');
        btn.setAttribute('aria-selected', 'false');
    });
    
    tabContents.forEach(content => {
        content.classList.remove('active');
    });
    
    // Activate selected tab
    const activeButton = document.querySelector(`.tab-btn[onclick*="${tabName}"]`);
    const activeContent = document.getElementById(tabName);
    
    if (activeButton && activeContent) {
        activeButton.classList.add('active');
        activeButton.setAttribute('aria-selected', 'true');
        activeContent.classList.add('active');
        
        // Animate content appearance
        Utils.animateIn(activeContent);
    }
}

// Schema Generator Functions
async function generateSchema() {
    const description = document.getElementById('schemaDescription').value.trim();
    const name = document.getElementById('schemaName').value.trim();
    const model = document.getElementById('schemaModel').value;
    
    if (!description) {
        Utils.showNotification('Please provide a schema description', 'warning');
        return;
    }
    
    if (!model) {
        Utils.showNotification('Please select an AI model', 'warning');
        return;
    }
    
    const resultsContainer = document.getElementById('schemaResults');
    const outputElement = document.getElementById('schemaOutput');
    
    try {
        // Show loading state
        Utils.showLoading('Generating your custom schema...', 'schemaResults');
        
        const requestBody = {
            description: description,
            model: model,
            base_url: "http://host.docker.internal:11434/v1",
            api_key: "ollama"
        };
        
        // Add schema name if provided
        if (name) {
            requestBody.schema_name = name;
        }
        
        const response = await fetch(ENDPOINTS.generateSchema, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            const errorMsg = typeof data.detail === 'string' ? data.detail : 
                            typeof data.detail === 'object' ? JSON.stringify(data.detail) :
                            data.message || data.error || 'Failed to generate schema';
            throw new Error(errorMsg);
        }
        
        // Debug logging
        console.log('Schema generation response:', data);
        console.log('Response keys:', Object.keys(data));
        console.log('python_code:', data.python_code);
        console.log('schema:', data.schema);
        
        // Display results with animation
        const schemaCode = data.python_code || data.schema || data.code || JSON.stringify(data, null, 2);
        console.log('Final schema code to display:', schemaCode);
        console.log('Output element:', outputElement);
        console.log('Results container:', resultsContainer);
        
        if (schemaCode) {
            // SIMPLE WORKING SOLUTION - Replace all the complex code with this
            Utils.hideLoading('schemaResults');
            
            resultsContainer.innerHTML = `
                <div style="margin: 24px 0; padding: 20px; background: rgba(255,255,255,0.03); 
                           border: 1px solid rgba(0,245,255,0.3); border-radius: 12px;">
                    <h3 style="color: #00f5ff; margin: 0 0 16px 0;">Generated Schema</h3>
                    <pre id="schemaOutput" style="background: rgba(15,23,42,0.8); border: 1px solid rgba(0,245,255,0.2); 
                               border-radius: 8px; padding: 16px; font-family: monospace; 
                               color: #e2e8f0; white-space: pre-wrap; margin: 0;">${schemaCode}</pre>
                </div>
            `;
            
            resultsContainer.classList.remove('hidden');
            resultsContainer.style.display = 'block';
            Utils.showNotification('Schema generated successfully! 🎉', 'success');
            
            // Scroll to results
            resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            
            outputElement.textContent = schemaCode;
            console.log('Set output element text content:', outputElement.textContent.substring(0, 100));
            console.log('Output element after setting content:', outputElement);
            console.log('Output element innerHTML:', outputElement.innerHTML);
            console.log('Output element textContent length:', outputElement.textContent.length);
            
            // Ensure proper display with beautiful styling
            resultsContainer.style.setProperty('display', 'block', 'important');
            resultsContainer.style.setProperty('visibility', 'visible', 'important');
            resultsContainer.style.setProperty('opacity', '1', 'important');
            resultsContainer.style.setProperty('margin-top', 'var(--space-lg)', 'important');
            resultsContainer.style.setProperty('padding', 'var(--space-md)', 'important');
            
            // Try alternative methods of setting content
            outputElement.innerHTML = schemaCode;
            console.log('Also tried innerHTML method');
            
            // Try multiple methods to set content and force visibility
            outputElement.textContent = schemaCode;
            outputElement.innerHTML = schemaCode; // Also try innerHTML in case textContent is blocked
            
            // Add a fallback visible test inside
            if (!outputElement.textContent || outputElement.textContent.trim() === '') {
                outputElement.innerHTML = '<span style="color: red;">ERROR: Content not displaying</span>';
            }
            
            // Override any problematic CSS but make it beautiful
            resultsContainer.style.setProperty('background', 'rgba(255, 255, 255, 0.03)', 'important');
            resultsContainer.style.setProperty('border', '1px solid rgba(0, 245, 255, 0.3)', 'important');
            resultsContainer.style.setProperty('border-radius', '12px', 'important');
            resultsContainer.style.setProperty('backdrop-filter', 'blur(10px)', 'important');
            
            outputElement.style.setProperty('color', '#e2e8f0', 'important');
            outputElement.style.setProperty('background', 'rgba(15, 23, 42, 0.8)', 'important');
            outputElement.style.setProperty('border', '1px solid rgba(0, 245, 255, 0.2)', 'important');
            outputElement.style.setProperty('border-radius', '8px', 'important');
            outputElement.style.setProperty('font-family', '"JetBrains Mono", "Fira Code", monospace', 'important');
            outputElement.style.setProperty('font-size', '0.9rem', 'important');
            outputElement.style.setProperty('line-height', '1.6', 'important');
            outputElement.style.setProperty('white-space', 'pre-wrap', 'important');
            outputElement.style.setProperty('min-height', '60px', 'important');
            outputElement.style.setProperty('padding', '15px', 'important');
            
            // Debug: Add placeholder text to see if styling works
            setTimeout(() => {
                if (outputElement.offsetHeight === 0 || !outputElement.textContent) {
                    outputElement.innerHTML = `<div style="color: #00f5ff; padding: 10px;">
                        ✅ Schema Generated Successfully!<br><br>
                        <code style="color: #e2e8f0; display: block; white-space: pre-wrap;">${schemaCode}</code>
                    </div>`;
                }
            }, 100);
            
            console.log('Applied beautiful theme-matching styles and fallbacks');
            
            resultsContainer.classList.remove('hidden');
            console.log('Removed hidden class from results container. Classes now:', resultsContainer.className);
            console.log('Results container display style:', window.getComputedStyle(resultsContainer).display);
            console.log('Results container visibility:', window.getComputedStyle(resultsContainer).visibility);
            
            Utils.animateIn(resultsContainer);
            console.log('Animated results container in');
            
            // Store schema locally using the server's actual filename
            const schemaName = description.substring(0, 50) + (description.length > 50 ? '...' : '');
            
            // Use the exact filename from CLI-style response
            console.log('🔍 Full server response:', data);
            const serverFilename = data.file_path ? data.file_path.split('/').pop() : null;
            console.log('🔍 Extracted filename:', serverFilename);
            console.log('🔍 This filename will be used for extraction:', serverFilename);
            
            // Store schema locally with the server's actual filename
            const schemaId = SchemaStore.addSchema(schemaName, schemaCode, description, serverFilename);
            
            Utils.showNotification('Schema generated successfully! 🎉 Available in PDF Extractor.', 'success');
            
            // Scroll to results
            resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            console.log('Scrolled to results');
        } else {
            Utils.hideLoading('schemaResults');
            throw new Error('No schema data received from server');
        }
        
    } catch (error) {
        console.error('Schema generation error:', error);
        Utils.hideLoading('schemaResults');
        
        // Improved error message handling
        let errorMessage = 'Unknown error occurred';
        if (error instanceof Error) {
            errorMessage = error.message;
        } else if (typeof error === 'string') {
            errorMessage = error;
        } else if (typeof error === 'object' && error !== null) {
            errorMessage = JSON.stringify(error);
        }
        
        Utils.showNotification(`Error: ${errorMessage}`, 'error');
    }
}

// Schema is now saved automatically by the backend during generation

// PDF Extractor Functions
async function extractPDF() {
    console.log('🔍 PDF Extract function called');
    console.log('🚨 EXTRACT PDF FUNCTION IS RUNNING!');
    const fileInput = document.getElementById('extractFiles');
    console.log('🔍 PDF Extract - File input element:', fileInput);
    console.log('🔍 PDF Extract - All input properties:', Object.getOwnPropertyNames(fileInput));
    console.log('🔍 PDF Extract - input.files:', fileInput.files, fileInput.files ? fileInput.files.length : 'null');
    console.log('🔍 PDF Extract - input._droppedFiles:', fileInput._droppedFiles, fileInput._droppedFiles ? fileInput._droppedFiles.length : 'null');
    
    // Try to get files from multiple sources
    let files = null;
    if (fileInput.files && fileInput.files.length > 0) {
        files = fileInput.files;
        console.log('✅ Using input.files');
    } else if (fileInput._droppedFiles && fileInput._droppedFiles.length > 0) {
        files = fileInput._droppedFiles;
        console.log('✅ Using input._droppedFiles');
    } else {
        // Try to find files in other ways
        console.log('🔍 Searching for files in other locations...');
        console.log('🔍 Window._lastDroppedFiles:', window._lastDroppedFiles);
        files = window._lastDroppedFiles;
    }
    
    console.log('🔍 PDF Extract - Final files chosen:', files, files ? files.length : 'null', files ? Array.from(files).map(f => f.name) : 'no files');
    
    const schemaId = document.getElementById('extractSchema').value;
    const model = document.getElementById('extractModel').value;
    console.log('🔍 PDF Extract - Schema ID:', schemaId);
    console.log('🔍 PDF Extract - Model:', model);
    
    if (!files || files.length === 0) {
        console.error('❌ No files found! Debugging file input state...');
        console.log('📋 File input debugging:');
        console.log('- fileInput exists:', !!fileInput);
        console.log('- fileInput.id:', fileInput?.id);
        console.log('- fileInput.files exists:', !!fileInput?.files);
        console.log('- fileInput._droppedFiles exists:', !!fileInput?._droppedFiles);
        
        Utils.showNotification('Please select at least one PDF file', 'warning');
        return;
    }
    
    if (!schemaId) {
        Utils.showNotification('Please select a generated schema', 'warning');
        return;
    }
    
    const selectedSchema = SchemaStore.getSchema(schemaId);
    if (!selectedSchema) {
        Utils.showNotification('Selected schema not found', 'error');
        return;
    }
    
    if (!model) {
        Utils.showNotification('Please select an AI model', 'warning');
        return;
    }
    
    const resultsContainer = document.getElementById('extractResults');
    const outputElement = document.getElementById('extractOutput');
    
    try {
        // Show loading state
        Utils.showLoading('Extracting data from your PDFs...', 'extractResults');
        
        const formData = new FormData();
        // Server expects 'file' (singular), not 'files' (plural)
        formData.append('file', files[0]); // Use first file for now
        
        // Use the full relative path to the schema file
        console.log('🔍 Selected schema object:', selectedSchema);
        console.log('🔍 Schema filename being sent:', selectedSchema.filename);
        console.log('🔍 Schema ID:', selectedSchema.id);
        console.log('🔍 Schema prompt:', selectedSchema.prompt);
        
        let schemaPath;
        if (selectedSchema.filename) {
            // Use full relative path that MOSAICX expects
            schemaPath = `mosaicx/schema/pyd/${selectedSchema.filename}`;
        } else {
            // Fallback for older schemas - use the schema ID as name
            schemaPath = selectedSchema.id;
        }
        
        console.log('🔍 Final schema path/identifier being sent:', schemaPath);
        formData.append('schema_identifier', schemaPath);
        formData.append('description', selectedSchema.prompt);
        formData.append('prompt', selectedSchema.prompt);
        formData.append('model', model);
        formData.append('base_url', 'http://host.docker.internal:11434/v1');
        formData.append('api_key', 'ollama');
        
        console.log('🚀 Sending PDF extraction request with:');
        console.log('- Files:', Array.from(files).map(f => ({
            name: f.name, 
            size: f.size, 
            type: f.type,
            lastModified: f.lastModified
        })));
        console.log('- Schema:', selectedSchema.name);
        console.log('- Model:', model);
        
        // Debug the actual file object
        const firstFile = files[0];
        console.log('🔍 First file details:', {
            name: firstFile.name,
            size: firstFile.size,
            type: firstFile.type,
            isFile: firstFile instanceof File,
            hasContent: firstFile.size > 0
        });
        
        const response = await fetch(ENDPOINTS.extractPDF, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        console.log('📥 Server response:', data);
        
        if (!response.ok) {
            const errorMsg = typeof data.detail === 'string' ? data.detail : 
                            typeof data.detail === 'object' ? JSON.stringify(data.detail, null, 2) :
                            data.message || data.error || `Server error ${response.status}`;
            throw new Error(errorMsg);
        }
        
        // Hide loading and display results
        Utils.hideLoading('extractResults');
        
        // Display results in a beautiful modal popup
        const modal = document.createElement('div');
        modal.innerHTML = `
            <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
                        background: rgba(0,0,0,0.8); z-index: 9999; display: flex; align-items: center; justify-content: center;">
                <div style="background: linear-gradient(135deg, rgba(15,23,42,0.95) 0%, rgba(30,41,59,0.95) 100%); 
                            border: 1px solid rgba(0,245,255,0.3); border-radius: 16px; padding: 30px; 
                            max-width: 90%; max-height: 90%; overflow: auto; backdrop-filter: blur(10px);
                            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                        <h2 style="color: #00f5ff; margin: 0; font-size: 1.5rem;">🎉 Extraction Complete</h2>
                        <button id="closeModal" 
                                style="background: none; border: 1px solid rgba(0,245,255,0.5); color: #00f5ff; 
                                       padding: 8px 12px; border-radius: 8px; cursor: pointer; font-size: 14px;">✕</button>
                    </div>
                    <div style="margin-bottom: 15px;">
                        <div style="color: rgba(226,232,240,0.8); font-size: 14px;">📄 ${data.file_name}</div>
                        <div style="color: rgba(226,232,240,0.6); font-size: 12px; margin-top: 5px;">
                            Schema: ${data.schema_used.split('/').pop()} | Model: ${data.model_used}
                        </div>
                    </div>
                    <div style="background: rgba(0,245,255,0.05); border: 1px solid rgba(0,245,255,0.2); 
                                border-radius: 12px; padding: 20px; font-family: 'JetBrains Mono', monospace; 
                                color: #e2e8f0; font-size: 14px; line-height: 1.6; white-space: pre-wrap;">${JSON.stringify(data.extracted_data, null, 2)}</div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
        
        // Add close functionality
        const closeBtn = modal.querySelector('#closeModal');
        const modalBackground = modal.querySelector('[style*="position: fixed"]');
        
        closeBtn.addEventListener('click', () => {
            modal.remove();
        });
        
        // Also close when clicking outside the modal
        modalBackground.addEventListener('click', (e) => {
            if (e.target === modalBackground) {
                modal.remove();
            }
        });
        
        // Close with Escape key
        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                modal.remove();
                document.removeEventListener('keydown', handleEscape);
            }
        };
        document.addEventListener('keydown', handleEscape);
        
        console.log('📊 Results container after setting:', resultsContainer);
        console.log('📊 Output element innerHTML length:', outputElement.innerHTML.length);
        
        Utils.animateIn(resultsContainer);
        Utils.showNotification(`Successfully extracted data! 🚀`, 'success');
        
        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        
    } catch (error) {
        console.error('PDF extraction error:', error);
        Utils.hideLoading('extractResults');
        Utils.showNotification(`Error: ${error.message}`, 'error');
    }
}

// Report Summarizer Functions
async function summarizeDirectory() {
    const files = document.getElementById('summarizeFiles').files;
    const model = document.getElementById('summarizeModel').value;
    
    if (files.length === 0) {
        Utils.showNotification('Please select files to summarize', 'warning');
        return;
    }
    
    if (!model) {
        Utils.showNotification('Please select an AI model', 'warning');
        return;
    }
    
    const resultsContainer = document.getElementById('summarizeResults');
    const overallSummaryElement = document.getElementById('overallSummary');
    const timelineElement = document.getElementById('timelineOutput');
    
    try {
        // Show loading state
        Utils.showLoading('Analyzing and summarizing your reports...', 'summarizeResults');
        
        const formData = new FormData();
        Array.from(files).forEach(file => {
            formData.append('pdf_files', file);
        });
        formData.append('patient_id', 'patient');
        formData.append('model', model);
        formData.append('temperature', '0.2');
        
        const response = await fetch(ENDPOINTS.summarizeFiles, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to summarize files');
        }
        
        // Display timeline in a beautiful modal
        let timelineHTML = '';
        if (data.timeline && data.timeline.length > 0) {
            data.timeline.forEach(item => {
                timelineHTML += `
                    <div style="margin-bottom: 15px; padding: 15px; background: rgba(0,245,255,0.05); 
                                border: 1px solid rgba(0,245,255,0.2); border-radius: 8px;">
                        <div style="color: #00f5ff; font-weight: 600; margin-bottom: 8px;">
                            ${item.filename || item.date || 'Clinical Report'}
                        </div>
                        <div style="color: #e2e8f0; line-height: 1.6;">
                            ${item.note || item.summary || item.content || 'No content available'}
                        </div>
                    </div>
                `;
            });
        } else {
            timelineHTML = `
                <div style="padding: 20px; text-align: center; color: rgba(226,232,240,0.6);">
                    No timeline data available
                </div>
            `;
        }
        
        // Display results in a beautiful modal popup
        const modal = document.createElement('div');
        modal.innerHTML = `
            <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
                        background: rgba(0,0,0,0.8); z-index: 9999; display: flex; align-items: center; justify-content: center;">
                <div style="background: linear-gradient(135deg, rgba(15,23,42,0.95) 0%, rgba(30,41,59,0.95) 100%); 
                            border: 1px solid rgba(0,245,255,0.3); border-radius: 16px; padding: 30px; 
                            max-width: 90%; max-height: 90%; overflow: auto; backdrop-filter: blur(10px);
                            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5); min-width: 600px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                        <h2 style="color: #00f5ff; margin: 0; font-size: 1.5rem;">📊 Summarization Complete</h2>
                        <button id="closeModal" 
                                style="background: none; border: 1px solid rgba(0,245,255,0.5); color: #00f5ff; 
                                       padding: 8px 12px; border-radius: 8px; cursor: pointer; font-size: 14px;">✕</button>
                    </div>
                    <div style="margin-bottom: 20px;">
                        <div style="color: rgba(226,232,240,0.8); font-size: 14px; margin-bottom: 10px;">
                            📄 Processed ${files.length} file(s) | Patient: ${data.patient_id || 'Unknown'}
                        </div>
                        ${data.overall_summary ? `
                        <div style="background: rgba(124,58,237,0.1); border: 1px solid rgba(124,58,237,0.3); 
                                    border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                            <h3 style="color: #a855f7; margin: 0 0 10px 0; font-size: 16px;">📋 Overall Summary</h3>
                            <div style="color: #e2e8f0; line-height: 1.6;">${data.overall_summary}</div>
                        </div>
                        ` : ''}
                    </div>
                    <div style="max-height: 400px; overflow-y: auto;">
                        <h3 style="color: #00f5ff; margin: 0 0 15px 0; font-size: 16px;">📅 Timeline</h3>
                        ${timelineHTML}
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
        
        // Add close functionality
        const closeBtn = modal.querySelector('#closeModal');
        const modalBackground = modal.querySelector('[style*="position: fixed"]');
        
        closeBtn.addEventListener('click', () => {
            modal.remove();
        });
        
        // Also close when clicking outside the modal
        modalBackground.addEventListener('click', (e) => {
            if (e.target === modalBackground) {
                modal.remove();
            }
        });
        
        // Close with Escape key
        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                modal.remove();
                document.removeEventListener('keydown', handleEscape);
            }
        };
        document.addEventListener('keydown', handleEscape);
        
        // Hide loading spinner since modal is now displayed
        Utils.hideLoading('summarizeResults');
        
        Utils.showNotification(`Successfully analyzed ${files.length} file(s)! ⭐`, 'success');
        
    } catch (error) {
        console.error('Summarization error:', error);
        Utils.hideLoading('summarizeResults');
        Utils.showNotification(`Error: ${error.message}`, 'error');
    }
}

// File List Display Function
function displayFileList(inputId, files) {
    console.log('📋 displayFileList called for', inputId, 'with', files.length, 'files');
    
    // Generate file list container ID from input ID
    // HTML has: extractFileList, summarizeFileList (without 's')
    const fileListId = inputId.replace('Files', 'FileList');
    console.log('Looking for file list container:', fileListId);
    
    const fileListContainer = document.getElementById(fileListId);
    console.log('File list container found:', !!fileListContainer, fileListContainer);
    
    if (!fileListContainer) {
        console.warn('File list container not found:', fileListId);
        return;
    }
    
    if (files.length === 0) {
        fileListContainer.innerHTML = '';
        return;
    }
    
    console.log('Creating file list HTML for', files.length, 'files');
    let html = '<div style="margin-top: 1rem; padding: 0.75rem; background: rgba(0,245,255,0.05); border: 1px solid rgba(0,245,255,0.2); border-radius: 8px;">';
    html += '<div style="color: #00f5ff; font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem;">📁 Selected Files:</div>';
    
    Array.from(files).forEach((file, index) => {
        const sizeText = Utils.formatFileSize ? Utils.formatFileSize(file.size) : (file.size / 1024).toFixed(1) + ' KB';
        const fileIcon = file.name.toLowerCase().endsWith('.pdf') ? '📄' : 
                        file.name.toLowerCase().endsWith('.txt') ? '📝' : 
                        file.name.toLowerCase().endsWith('.docx') ? '📋' : '📄';
        
        html += `
            <div style="display: flex; align-items: center; padding: 0.5rem; background: rgba(255,255,255,0.05); 
                        border-radius: 6px; margin: 0.25rem 0; color: var(--text-primary); font-size: 0.9rem;">
                <span style="margin-right: 0.5rem; font-size: 1.1rem;">${fileIcon}</span>
                <div style="flex: 1;">
                    <div style="font-weight: 500;">${file.name}</div>
                    <div style="color: rgba(255,255,255,0.6); font-size: 0.8rem;">${sizeText}</div>
                </div>
                <div style="color: rgba(0,245,255,0.8); font-size: 0.8rem;">#${index + 1}</div>
            </div>
        `;
        console.log(`Added file ${index + 1} to list:`, file.name, file.size);
    });
    
    html += '</div>';
    fileListContainer.innerHTML = html;
    console.log('✅ File list HTML set successfully');
}

// Form Event Handlers
function setupFormHandlers() {
    // Schema form
    const schemaForm = document.getElementById('schemaForm');
    if (schemaForm) {
        schemaForm.addEventListener('submit', function(e) {
            e.preventDefault();
            generateSchema();
        });
    }
    
    // Extract form
    const extractForm = document.getElementById('extractForm');
    if (extractForm) {
        extractForm.addEventListener('submit', function(e) {
            e.preventDefault();
            extractPDF();
        });
    }
    
    // Summarize form
    const summarizeForm = document.getElementById('summarizeForm');
    if (summarizeForm) {
        summarizeForm.addEventListener('submit', function(e) {
            e.preventDefault();
            summarizeDirectory();
        });
    }
}

// File Upload Enhancement
function enhanceFileUploads() {
    console.log('🔧 enhanceFileUploads() called');
    const fileInputs = document.querySelectorAll('input[type="file"]');
    console.log('Found file inputs:', fileInputs.length, fileInputs);
    
    fileInputs.forEach((input, index) => {
        console.log(`Processing input ${index}: id=${input.id}`);
        
        const parentElement = input.parentElement;
        console.log('Parent element:', parentElement);
        
        const label = parentElement.querySelector('label');
        console.log('Found label:', !!label, label);
        
        if (!label) {
            console.log('No label found for input:', input.id);
            return;
        }
        
        console.log('Setting up file selection and drag & drop for:', input.id);
        
        // File selection change handler (for regular file picker)
        input.addEventListener('change', function(e) {
            console.log('📁 File selection changed on', input.id);
            const files = e.target.files;
            console.log('Files selected:', files.length, Array.from(files).map(f => f.name));
            
            // Update label text
            const labelText = label.querySelector('.file-upload-text');
            if (labelText) {
                if (files.length > 0) {
                    labelText.textContent = `${files.length} file(s) selected`;
                } else {
                    labelText.textContent = 'Choose Report Files';
                }
            }
            
            // Show file list
            displayFileList(input.id, files);
        });
        
        // Drag and drop functionality with detailed logging
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            label.addEventListener(eventName, preventDefaults, false);
            console.log(`Added ${eventName} preventDefault listener to`, input.id);
        });
        
        ['dragenter', 'dragover'].forEach(eventName => {
            label.addEventListener(eventName, highlight, false);
            console.log(`Added ${eventName} highlight listener to`, input.id);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            label.addEventListener(eventName, unhighlight, false);
            console.log(`Added ${eventName} unhighlight listener to`, input.id);
        });
        
        label.addEventListener('drop', handleDrop, false);
        console.log(`Added drop handleDrop listener to`, input.id);
        
        // Also add events to the parent container for better coverage
        const container = input.parentElement;
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            container.addEventListener(eventName, preventDefaults, false);
        });
        ['dragenter', 'dragover'].forEach(eventName => {
            container.addEventListener(eventName, highlight, false);
        });
        ['dragleave', 'drop'].forEach(eventName => {
            container.addEventListener(eventName, unhighlight, false);
        });
        container.addEventListener('drop', handleDrop, false);
        console.log(`Added container events to`, input.id);
        
        function preventDefaults(e) {
            console.log('🛑 preventDefaults called for', e.type, 'on', input.id);
            e.preventDefault();
            e.stopPropagation();
        }
        
        function highlight(e) {
            console.log('🔆 Highlight triggered on', input.id, 'event:', e.type);
            label.style.borderColor = 'var(--accent-primary)';
            label.style.background = 'rgba(0, 245, 255, 0.1)';
        }
        
        function unhighlight(e) {
            console.log('🔅 Unhighlight triggered on', input.id, 'event:', e.type);
            label.style.borderColor = 'rgba(255, 255, 255, 0.3)';
            label.style.background = 'var(--bg-glass)';
        }
        
        function handleDrop(e) {
            e.preventDefault();
            e.stopPropagation();
            console.log('Drop detected on', input.id, '!');
            const files = e.dataTransfer.files;
            console.log('Files:', files.length, Array.from(files).map(f => f.name));
            
            // Store files in multiple places for reliability
            input._droppedFiles = files;
            window._lastDroppedFiles = files; // Global backup
            
            console.log('✅ Stored files in input._droppedFiles:', input._droppedFiles);
            console.log('✅ Stored files in window._lastDroppedFiles:', window._lastDroppedFiles);
            
            // Update label text
            const labelText = label.querySelector('.file-upload-text');
            if (labelText) {
                labelText.textContent = `${files.length} file(s) selected`;
            }
            
            // Show file list using shared function
            displayFileList(input.id, files);
            
            Utils.showNotification(`${files.length} file(s) selected! 📎`, 'success');
        }
    });
    
    // Add global drag and drop as backup - only if not handled by file inputs
    document.body.addEventListener('dragover', function(e) {
        // Only prevent default if not over a file upload area
        if (!e.target.closest('.file-upload')) {
            e.preventDefault();
        }
    });
    
    document.body.addEventListener('drop', function(e) {
        // Only prevent default if not over a file upload area
        if (!e.target.closest('.file-upload')) {
            e.preventDefault();
            console.log('Global drop detected - preventing default behavior (not over file upload)');
        }
    });
}

// Keyboard Navigation
function setupKeyboardNavigation() {
    document.addEventListener('keydown', function(e) {
        // Tab navigation with arrow keys
        if (e.altKey && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
            e.preventDefault();
            
            const tabs = ['schema', 'extract', 'summarize'];
            const currentIndex = tabs.indexOf(AppState.currentTab);
            let nextIndex;
            
            if (e.key === 'ArrowLeft') {
                nextIndex = currentIndex > 0 ? currentIndex - 1 : tabs.length - 1;
            } else {
                nextIndex = currentIndex < tabs.length - 1 ? currentIndex + 1 : 0;
            }
            
            showTab(tabs[nextIndex]);
        }
        
        // ESC to close notifications
        if (e.key === 'Escape') {
            const notifications = document.querySelectorAll('[style*="position: fixed"][style*="top: 2rem"]');
            notifications.forEach(notification => notification.remove());
        }
    });
}

// Create floating particles background
function createParticles() {
    const particlesContainer = document.getElementById('particles');
    if (!particlesContainer) return;
    
    const particleCount = 50;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 15 + 's';
        particle.style.animationDuration = (Math.random() * 10 + 15) + 's';
        particlesContainer.appendChild(particle);
    }
}

// Add CSS animations
function addCustomStyles() {
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideInRight {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        @keyframes slideOutRight {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }
        
        .fade-in {
            animation: fadeIn 0.3s ease-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    `;
    document.head.appendChild(style);
}

// Load available models from Ollama
async function loadAvailableModels() {
    try {
        const response = await fetch('http://localhost:11434/api/tags');
        if (response.ok) {
            const data = await response.json();
            const models = data.models.map(model => model.name);
            
            // Update all model select dropdowns
            const modelSelects = ['schemaModel', 'extractModel', 'summarizeModel'];
            modelSelects.forEach(selectId => {
                const select = document.getElementById(selectId);
                if (select) {
                    select.innerHTML = ''; // Clear existing options
                    
                    if (models.length > 0) {
                        models.forEach(model => {
                            const option = document.createElement('option');
                            option.value = model;
                            option.textContent = model;
                            select.appendChild(option);
                        });
                        // Set first model as default
                        select.value = models[0];
                    } else {
                        const option = document.createElement('option');
                        option.value = '';
                        option.textContent = 'No models available';
                        select.appendChild(option);
                    }
                }
            });
            
            Utils.showNotification(`Loaded ${models.length} AI models successfully! 🤖`, 'success');
        }
    } catch (error) {
        console.warn('Could not load models from Ollama:', error);
        // Set fallback options if Ollama is not accessible
        const modelSelects = ['schemaModel', 'extractModel', 'summarizeModel'];
        modelSelects.forEach(selectId => {
            const select = document.getElementById(selectId);
            if (select) {
                select.innerHTML = '<option value="llama3.2">llama3.2 (default)</option>';
                select.value = 'llama3.2';
            }
        });
        Utils.showNotification('Using default model (Ollama not accessible)', 'warning');
    }
}

// Test function to verify UI works
function testSchemaDisplay() {
    const resultsContainer = document.getElementById('schemaResults');
    const outputElement = document.getElementById('schemaOutput');
    
    if (resultsContainer && outputElement) {
        outputElement.textContent = `# Test Schema Generation
from pydantic import BaseModel
from typing import Optional

class PatientDemographics(BaseModel):
    name: str
    age: int
    gender: str
    medical_history: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "name": "John Doe",
                "age": 45,
                "gender": "Male",
                "medical_history": "Hypertension, Diabetes"
            }
        }`;
        
        resultsContainer.classList.remove('hidden');
        Utils.animateIn(resultsContainer);
        Utils.showNotification('Test schema displayed successfully! The UI is working.', 'success');
    } else {
        console.error('Could not find schema results elements:', {
            resultsContainer,
            outputElement
        });
        Utils.showNotification('Error: Could not find schema display elements', 'error');
    }
}

// Initialize Application
function initializeApp() {
    console.log('🧬 MOSAICX WebApp - Initializing ultra-modern interface...');
    
    // Add custom styles
    addCustomStyles();
    
    // Create floating particles background
    createParticles();
    
    // Setup form handlers
    setupFormHandlers();
    
    // Enhance file uploads
    enhanceFileUploads();
    
    // Setup keyboard navigation
    setupKeyboardNavigation();
    
    // Load available models
    loadAvailableModels();
    
    // Initialize schema storage
    SchemaStore.loadSchemas();
    
    // Show welcome message
    setTimeout(() => {
        Utils.showNotification('Welcome to MOSAICX! 🚀 Ready for AI-powered medical data processing.', 'success');
    }, 1000);
    
    // Add test functions to window for debugging
    window.testSchemaDisplay = testSchemaDisplay;
    window.testDragDrop = function() {
        console.log('🧪 Testing drag and drop setup...');
        const fileInputs = document.querySelectorAll('input[type="file"]');
        console.log('Found file inputs:', fileInputs.length);
        fileInputs.forEach((input, i) => {
            console.log(`Input ${i}: id=${input.id}, exists=${!!input}`);
            const label = input.parentElement.querySelector('label');
            console.log(`Label ${i}:`, label ? 'found' : 'not found');
        });
        
        // Try to manually trigger the setup again
        enhanceFileUploads();
        
        // Test direct event
        const extractInput = document.getElementById('extractFiles');
        if (extractInput) {
            console.log('Testing direct drag events on extractFiles...');
            const label = extractInput.parentElement.querySelector('label');
            if (label) {
                label.addEventListener('dragover', function(e) {
                    e.preventDefault();
                    console.log('MANUAL DRAGOVER detected!');
                    label.style.borderColor = '#ff0000';
                });
                label.addEventListener('drop', function(e) {
                    e.preventDefault();
                    console.log('MANUAL DROP detected!');
                    console.log('Files:', e.dataTransfer.files.length);
                });
                console.log('Added manual event listeners to label');
            }
        }
    };
    
    // DIRECT EXTRACTION BYPASS - CALL THIS TO SKIP ALL VALIDATION
    window.directExtract = function() {
        console.log('🚨 DIRECT EXTRACT CALLED - BYPASSING ALL VALIDATION');
        extractPDF();
    };

    // Test function to check current file status
    window.checkFileStatus = function() {
        console.log('🧪 Checking file status...');
        const fileInput = document.getElementById('extractFiles');
        console.log('File input:', fileInput);
        console.log('input.files:', fileInput.files, fileInput.files ? fileInput.files.length : 'null');
        console.log('input._droppedFiles:', fileInput._droppedFiles, fileInput._droppedFiles ? fileInput._droppedFiles.length : 'null');
        console.log('window._lastDroppedFiles:', window._lastDroppedFiles, window._lastDroppedFiles ? window._lastDroppedFiles.length : 'null');
        
        // Test if extractPDF function exists and can be called
        console.log('extractPDF function exists:', typeof extractPDF);
        
        return {
            inputFiles: fileInput.files ? fileInput.files.length : 0,
            droppedFiles: fileInput._droppedFiles ? fileInput._droppedFiles.length : 0,
            globalFiles: window._lastDroppedFiles ? window._lastDroppedFiles.length : 0
        };
    };

    // Test function to simulate file selection
    window.simulateFileSelection = function() {
        console.log('🧪 Simulating file selection...');
        const extractInput = document.getElementById('extractFiles');
        if (extractInput) {
            // Create a mock file
            const mockFile = new File(['test content'], 'test.pdf', { type: 'application/pdf' });
            const dt = new DataTransfer();
            dt.items.add(mockFile);
            
            // Store in custom property
            extractInput._droppedFiles = dt.files;
            
            // Update UI manually
            const label = document.querySelector('label[for="extractFiles"]');
            if (label) {
                const labelText = label.querySelector('.file-upload-text');
                if (labelText) {
                    labelText.textContent = '1 file(s) selected';
                }
            }
            
            // Show file list
            const fileListContainer = document.getElementById('extractFilesList');
            if (fileListContainer) {
                fileListContainer.innerHTML = '<div style="margin-top: 1rem; padding: 0.5rem; background: rgba(0,245,255,0.1); border-radius: 4px; color: var(--text-primary);">📄 test.pdf (12 Bytes)</div>';
            }
            
            console.log('✅ Simulated file selection complete!');
            Utils.showNotification('Simulated file selection! 📎', 'success');
        }
    };
    window.testShowResults = function() {
        console.log('🧪 Testing schema results display...');
        
        const resultsContainer = document.getElementById('schemaResults');
        const outputElement = document.getElementById('schemaOutput');
        
        console.log('Results container:', resultsContainer);
        console.log('Output element:', outputElement);
        
        if (!resultsContainer) {
            console.error('❌ schemaResults element not found!');
            alert('ERROR: schemaResults element not found in DOM');
            return;
        }
        
        if (!outputElement) {
            console.error('❌ schemaOutput element not found!');
            alert('ERROR: schemaOutput element not found in DOM');
            return;
        }
        
        // Set test content
        const testSchema = `from pydantic import BaseModel, Field

class PatientInfo(BaseModel):
    name: str = Field(..., description="Patient full name")
    age: int = Field(..., gt=0, lt=120, description="Patient age in years")`;
        
        outputElement.textContent = testSchema;
        console.log('✅ Set output text content');
        
        // Remove hidden class
        resultsContainer.classList.remove('hidden');
        console.log('✅ Removed hidden class');
        console.log('Current classes:', resultsContainer.className);
        
        // Check computed styles
        const computedStyle = window.getComputedStyle(resultsContainer);
        console.log('Display style:', computedStyle.display);
        console.log('Visibility style:', computedStyle.visibility);
        
        alert('Test completed! Check if schema is visible below the form.');
        Utils.showNotification('Test schema displayed! 🎉', 'success');
    };
    
    console.log('✅ MOSAICX WebApp initialized successfully!');
    console.log('💡 To test schema display, run: testSchemaDisplay() in console');
}

// Start the application when DOM is ready
document.addEventListener('DOMContentLoaded', initializeApp);