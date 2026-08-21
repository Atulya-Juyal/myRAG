/**
 * Client-side script for the AI-Semantic RAG Workspace application.
 * Integrates chat feeds, dynamic workspaces, file uploads, and background indexing polling.
 */

// Global State
let state = {
    workspaces: [],
    activeWorkspaceId: '',
    activeWorkspaceTitle: '',
    documents: [],
    localHistory: {}, // in-memory cache for message histories per workspace
    isSending: false,
    pollingTimer: null,
    healthTimer: null
};

// DOM Cache
const dom = {
    workspaceList: document.getElementById('workspace-list'),
    chatMessages: document.getElementById('chat-messages'),
    chatInput: document.getElementById('chat-input'),
    btnSend: document.getElementById('btn-send'),
    charCount: document.getElementById('char-count'),
    activeTitle: document.getElementById('active-chat-title'),
    activeSubtitle: document.getElementById('active-chat-subtitle'),
    documentList: document.getElementById('document-list'),
    btnNewChat: document.getElementById('btn-new-chat'),
    btnToggleDocs: document.getElementById('btn-toggle-docs'),
    btnToggleWorkspaces: document.getElementById('btn-toggle-workspaces'),
    leftSidebar: document.querySelector('.left-sidebar'),
    docSidebar: document.getElementById('doc-sidebar'),
    sidebarBackdrop: document.getElementById('sidebar-backdrop'),
    btnCloseLeftSidebar: document.getElementById('btn-close-left-sidebar'),
    btnCloseRightSidebar: document.getElementById('btn-close-right-sidebar'),
    uploadZone: document.getElementById('upload-zone'),
    fileInput: document.getElementById('file-input'),
    statusIndicator: document.getElementById('status-indicator'),
    statusText: document.getElementById('status-text'),
    modalContainer: document.getElementById('modal-container'),
    modalInput: document.getElementById('workspace-name-input'),
    btnModalCancel: document.getElementById('btn-modal-cancel'),
    btnModalCreate: document.getElementById('btn-modal-create')
};

// ----------------------------------------------------
// INITIALIZATION
// ----------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    fetchWorkspaces();
    startHealthCheck();
});

function closeMobileSidebars() {
    if (dom.leftSidebar) dom.leftSidebar.classList.remove('open');
    if (dom.docSidebar) dom.docSidebar.classList.remove('open');
    if (dom.sidebarBackdrop) dom.sidebarBackdrop.classList.remove('active');
}

function toggleLeftSidebar() {
    if (!dom.leftSidebar) return;
    const isOpen = dom.leftSidebar.classList.toggle('open');
    if (dom.docSidebar) dom.docSidebar.classList.remove('open');
    if (dom.sidebarBackdrop) {
        if (isOpen) dom.sidebarBackdrop.classList.add('active');
        else dom.sidebarBackdrop.classList.remove('active');
    }
}

function isMobileView() {
    return window.matchMedia('(max-width: 1280px), (orientation: portrait)').matches;
}

function toggleRightSidebar() {
    if (!dom.docSidebar) return;
    if (isMobileView()) {
        const isOpen = dom.docSidebar.classList.toggle('open');
        if (dom.leftSidebar) dom.leftSidebar.classList.remove('open');
        if (dom.sidebarBackdrop) {
            if (isOpen) dom.sidebarBackdrop.classList.add('active');
            else dom.sidebarBackdrop.classList.remove('active');
        }
    } else {
        dom.docSidebar.classList.toggle('collapsed');
    }
}

function setupEventListeners() {
    // Mobile Drawer Triggers
    if (dom.btnToggleWorkspaces) {
        dom.btnToggleWorkspaces.addEventListener('click', toggleLeftSidebar);
    }
    if (dom.btnCloseLeftSidebar) {
        dom.btnCloseLeftSidebar.addEventListener('click', closeMobileSidebars);
    }
    if (dom.btnCloseRightSidebar) {
        dom.btnCloseRightSidebar.addEventListener('click', closeMobileSidebars);
    }
    if (dom.sidebarBackdrop) {
        dom.sidebarBackdrop.addEventListener('click', closeMobileSidebars);
    }
    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeMobileSidebars();
    });

    // Modal Workspace Trigger
    dom.btnNewChat.addEventListener('click', () => {
        dom.modalInput.value = '';
        dom.modalContainer.style.display = 'flex';
        dom.modalInput.focus();
    });

    dom.btnModalCancel.addEventListener('click', () => {
        dom.modalContainer.style.display = 'none';
    });

    dom.btnModalCreate.addEventListener('click', handleCreateWorkspace);
    dom.modalInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') handleCreateWorkspace();
    });

    // Chat Send Triggers
    dom.btnSend.addEventListener('click', handleSendMessage);
    dom.chatInput.addEventListener('keydown', (e) => {
        const isEnter = e.key === 'Enter' || e.keyCode === 13 || e.which === 13;
        if (isEnter && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });

    // Chat Input Char Limit & Dynamic Height adjustment
    dom.chatInput.addEventListener('input', () => {
        const textLength = dom.chatInput.value.length;
        dom.charCount.textContent = textLength;
        dom.btnSend.disabled = textLength === 0 || state.isSending;

        // Auto growth height mapping
        dom.chatInput.style.height = 'auto';
        dom.chatInput.style.height = (dom.chatInput.scrollHeight) + 'px';
    });

    // Toggle Right Sidebar (Desktop collapse or Mobile drawer)
    if (dom.btnToggleDocs) {
        dom.btnToggleDocs.addEventListener('click', toggleRightSidebar);
    }

    // Ingest Drag and Drop Setup
    dom.uploadZone.addEventListener('click', () => dom.fileInput.click());
    dom.fileInput.addEventListener('change', handleFileSelected);

    dom.uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dom.uploadZone.classList.add('dragover');
    });

    dom.uploadZone.addEventListener('dragleave', () => {
        dom.uploadZone.classList.remove('dragover');
    });

    dom.uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dom.uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleUploadFile(e.dataTransfer.files[0]);
        }
    });
}

// ----------------------------------------------------
// API CLIENT IMPLEMENTATION
// ----------------------------------------------------

// Health Check API
function startHealthCheck() {
    const runHealthCheck = async () => {
        try {
            const res = await fetch('/health');
            if (res.ok) {
                const data = await res.json();
                if (data.status === 'healthy') {
                    dom.statusIndicator.className = 'status-dot connected';
                    dom.statusText.textContent = 'Backend Online';
                    return;
                }
            }
            dom.statusIndicator.className = 'status-dot disconnected';
            dom.statusText.textContent = 'Degraded State';
        } catch (err) {
            dom.statusIndicator.className = 'status-dot disconnected';
            dom.statusText.textContent = 'Backend Offline';
        }
    };

    runHealthCheck();
    state.healthTimer = setInterval(runHealthCheck, 10000);
}

// Workspace API (List, Create, Delete, Rename)
async function fetchWorkspaces() {
    try {
        const res = await fetch('/chats');
        if (res.ok) {
            state.workspaces = await res.json();
            
            // Automatically seed a 'New Chat' workspace if list is empty
            if (state.workspaces.length === 0) {
                await createWorkspace('New Chat');
                return;
            }
            
            renderWorkspaces();
            
            // Default select active workspaces if not set
            const exists = state.workspaces.some(w => w.chat_id === state.activeWorkspaceId);
            if (!exists) {
                state.activeWorkspaceId = state.workspaces.length > 0 ? state.workspaces[0].chat_id : '';
            }
            
            const active = state.workspaces.find(w => w.chat_id === state.activeWorkspaceId);
            if (active) {
                state.activeWorkspaceTitle = active.title;
            } else {
                state.activeWorkspaceTitle = '';
            }
            
            if (state.activeWorkspaceId) {
                selectWorkspace(state.activeWorkspaceId);
            }
        }
    } catch (err) {
        console.error('Failed to load workspaces:', err);
    }
}

async function createWorkspace(title) {
    try {
        const res = await fetch('/chats', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title })
        });
        if (res.ok) {
            const chat = await res.json();
            await fetchWorkspaces();
            selectWorkspace(chat.chat_id);
            return chat;
        }
    } catch (err) {
        console.error('Workspace creation failed:', err);
    }
}

async function handleCreateWorkspace() {
    const title = dom.modalInput.value.trim();
    if (!title) return;
    dom.modalContainer.style.display = 'none';
    await createWorkspace(title);
}

async function deleteWorkspace(chatId, event) {
    event.stopPropagation();
    if (!confirm('Are you sure you want to delete this workspace? All uploaded documents and vectors will be permanently destroyed.')) {
        return;
    }

    try {
        const res = await fetch(`/chats/${chatId}`, { method: 'DELETE' });
        if (res.ok) {
            // Delete messages from local history
            delete state.localHistory[chatId];
            
            // Switch active workspace if deleted active workspace
            if (state.activeWorkspaceId === chatId) {
                state.activeWorkspaceId = '';
            }
            await fetchWorkspaces();
        }
    } catch (err) {
        console.error('Failed to delete workspace:', err);
    }
}

async function renameWorkspace(chatId, currentTitle, event) {
    event.stopPropagation();
    const newTitle = prompt('Rename Workspace:', currentTitle);
    if (newTitle === null) return;
    const trimmed = newTitle.trim();
    if (!trimmed || trimmed === currentTitle) return;

    try {
        const res = await fetch(`/chats/${chatId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: trimmed })
        });
        if (res.ok) {
            await fetchWorkspaces();
        }
    } catch (err) {
        console.error('Failed to rename workspace:', err);
    }
}

async function fetchChatHistory(chatId) {
    try {
        // Show loading indicator in chat history area
        if (state.activeWorkspaceId === chatId) {
            dom.chatMessages.innerHTML = `
                <div class="welcome-message-card">
                    <div class="typing-indicator" style="margin: 0 auto; display: flex; justify-content: center; gap: 4px;">
                        <span style="background: var(--text-secondary); width: 8px; height: 8px; border-radius: 50%; display: inline-block;"></span>
                        <span style="background: var(--text-secondary); width: 8px; height: 8px; border-radius: 50%; display: inline-block;"></span>
                        <span style="background: var(--text-secondary); width: 8px; height: 8px; border-radius: 50%; display: inline-block;"></span>
                    </div>
                    <p style="margin-top: 10px; color: var(--text-secondary);">Loading chat history...</p>
                </div>
            `;
        }
        const res = await fetch(`/chats/${chatId}/history`);
        if (res.ok) {
            state.localHistory[chatId] = await res.json();
        } else {
            state.localHistory[chatId] = [];
        }
    } catch (err) {
        console.error('Failed to fetch chat history:', err);
        state.localHistory[chatId] = [];
    } finally {
        if (state.activeWorkspaceId === chatId) {
            renderMessages();
        }
    }
}

function selectWorkspace(chatId) {
    state.activeWorkspaceId = chatId;
    closeMobileSidebars();
    
    // Clear document polling timer
    if (state.pollingTimer) {
        clearTimeout(state.pollingTimer);
        state.pollingTimer = null;
    }

    // Update UI active styling
    const items = dom.workspaceList.querySelectorAll('.workspace-item');
    items.forEach(el => {
        if (el.dataset.id === chatId) {
            el.classList.add('active');
        } else {
            el.classList.remove('active');
        }
    });

    const activeWorkspace = state.workspaces.find(w => w.chat_id === chatId);
    if (activeWorkspace) {
        state.activeWorkspaceTitle = activeWorkspace.title;
        dom.activeTitle.textContent = activeWorkspace.title;
        const dateStr = new Date(activeWorkspace.created_at).toLocaleDateString();
        dom.activeSubtitle.textContent = `Workspace created on ${dateStr}`;
    } else {
        dom.activeTitle.textContent = 'No Workspace Selected';
        dom.activeSubtitle.textContent = 'Please select or create a workspace to begin';
    }

    // Refresh Documents List
    fetchDocuments();

    // Fetch history and render messages
    fetchChatHistory(chatId);
}

// Documents Ingestion API (Upload, List, Delete)
async function fetchDocuments(isPolling = false) {
    try {
        const res = await fetch(`/documents?chat_id=${state.activeWorkspaceId}`);
        if (res.ok) {
            state.documents = await res.json();
            renderDocuments();

            // Set up polling if any documents are currently in processing state
            const hasOngoingProcessing = state.documents.some(doc => 
                doc.status.startsWith('Processing') || doc.status.includes('Uploading')
            );
            
            if (hasOngoingProcessing) {
                if (state.pollingTimer) clearTimeout(state.pollingTimer);
                state.pollingTimer = setTimeout(() => fetchDocuments(true), 2500);
            }
        }
    } catch (err) {
        console.error('Failed to fetch documents:', err);
    }
}

function handleFileSelected(e) {
    if (e.target.files.length > 0) {
        handleUploadFile(e.target.files[0]);
    }
}

async function handleUploadFile(file) {
    const validExtensions = ['.pdf', '.txt', '.md'];
    const filename = file.name.toLowerCase();
    const isValid = validExtensions.some(ext => filename.endsWith(ext));

    if (!isValid) {
        alert('Invalid file format. Only PDF, TXT, or MD documents are allowed.');
        return;
    }

    if (file.size > 25 * 1024 * 1024) {
        alert('File is too large. Maximum supported limit is 25MB.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // Render temporary uploading item in the list
    const tempDocId = 'temp-' + Date.now();
    state.documents.push({
        id: tempDocId,
        filename: file.name,
        status: 'Processing: Ingesting file bytes...',
        char_count: 0,
        chunk_count: 0,
        uploaded_at: new Date().toISOString()
    });
    renderDocuments();

    try {
        const res = await fetch(`/documents/upload?chat_id=${state.activeWorkspaceId}`, {
            method: 'POST',
            body: formData
        });
        if (res.ok) {
            fetchDocuments();
        } else {
            const errData = await res.json();
            alert(`Upload failed: ${errData.detail || 'Server error'}`);
            fetchDocuments();
        }
    } catch (err) {
        console.error('Failed to upload document:', err);
        alert('Failed to connect to the server for upload.');
        fetchDocuments();
    }
}

async function deleteDocument(docId) {
    if (!confirm('Are you sure you want to remove this document context? It will be deleted from the vector index.')) {
        return;
    }

    try {
        const res = await fetch(`/documents/${state.activeWorkspaceId}/${docId}`, {
            method: 'DELETE'
        });
        if (res.ok) {
            fetchDocuments();
        }
    } catch (err) {
        console.error('Failed to delete document:', err);
    }
}

// Chat API (Invoke, Render)
async function handleSendMessage() {
    const text = dom.chatInput.value.trim();
    if (!text || state.isSending) return;

    state.isSending = true;
    dom.btnSend.disabled = true;
    dom.chatInput.value = '';
    dom.chatInput.style.height = 'auto';
    dom.charCount.textContent = '0';

    // Store user message in state
    if (!state.localHistory[state.activeWorkspaceId]) {
        state.localHistory[state.activeWorkspaceId] = [];
    }
    
    const userMsg = { sender: 'user', text, timestamp: new Date().toISOString() };
    state.localHistory[state.activeWorkspaceId].push(userMsg);
    
    // Clear welcome card if it exists
    const welcome = dom.chatMessages.querySelector('.welcome-message-card');
    if (welcome) welcome.remove();

    appendMessage('user', text);

    // Create typing indicator placeholder
    const typingId = 'typing-' + Date.now();
    appendTypingPlaceholder(typingId);

    try {
        const res = await fetch('/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                chat_id: state.activeWorkspaceId,
                thread_id: state.activeWorkspaceId
            })
        });

        if (!res.ok) {
            const typingEl = document.getElementById(typingId);
            if (typingEl) typingEl.remove();
            const errData = await res.json();
            const botErrorMsg = {
                sender: 'bot',
                text: `⚠️ Error: ${errData.detail || 'An error occurred while generating response.'}`,
                sources: [],
                timestamp: new Date().toISOString()
            };
            state.localHistory[state.activeWorkspaceId].push(botErrorMsg);
            appendMessage('bot', botErrorMsg.text);
            return;
        }

        // Active streaming placeholders & progressive typing variables
        const activeMessageId = 'stream-' + Date.now();
        let hasCreatedBubble = false;
        let targetContent = '';
        let displayedContent = '';
        let typingTimer = null;
        let isStreamDone = false;
        let sources = [];
        let modelUsed = 'primary';

        const typeNext = () => {
            if (displayedContent.length < targetContent.length) {
                const remaining = targetContent.length - displayedContent.length;
                let step = 1;
                // Catch up dynamically if we are far behind due to large buffered chunks
                if (remaining > 300) step = 15;
                else if (remaining > 100) step = 8;
                else if (remaining > 30) step = 4;
                else if (remaining > 10) step = 2;

                displayedContent += targetContent.substr(displayedContent.length, step);
                updateStreamingBubble(activeMessageId, displayedContent);
                typingTimer = setTimeout(typeNext, 15);
            } else {
                typingTimer = null;
                if (isStreamDone) {
                    finalizeStreamingMessage(activeMessageId, targetContent, sources);
                    const botMsg = { 
                        sender: 'bot', 
                        text: targetContent, 
                        sources: sources, 
                        timestamp: new Date().toISOString()
                    };
                    state.localHistory[state.activeWorkspaceId].push(botMsg);
                }
            }
        };

        const processToken = (token) => {
            targetContent += token;
            if (!hasCreatedBubble) {
                const typingEl = document.getElementById(typingId);
                if (typingEl) typingEl.remove();
                appendStreamingPlaceholder(activeMessageId);
                hasCreatedBubble = true;
            }
            if (!typingTimer) {
                typeNext();
            }
        };

        // Read the streaming response body
        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // keep partial last line in buffer

            for (const line of lines) {
                const cleanLine = line.trim();
                if (!cleanLine.startsWith('data:')) continue;

                try {
                    const payloadStr = cleanLine.substring(5).trim();
                    const data = JSON.parse(payloadStr);

                    if (data.type === 'token') {
                        processToken(data.content);
                    } else if (data.type === 'done') {
                        sources = data.sources || [];
                        modelUsed = data.model_used || 'primary';
                    }
                } catch (e) {
                    console.error('Failed to parse SSE line:', cleanLine, e);
                }
            }
        }

        // Process remaining buffer
        if (buffer && buffer.trim().startsWith('data:')) {
            try {
                const payloadStr = buffer.trim().substring(5).trim();
                const data = JSON.parse(payloadStr);
                if (data.type === 'token') {
                    processToken(data.content);
                } else if (data.type === 'done') {
                    sources = data.sources || [];
                    modelUsed = data.model_used || 'primary';
                }
            } catch (e) {
                console.warn('Final buffer parse failed:', e);
            }
        }

        isStreamDone = true;
        
        // If typing has already caught up, finalize it immediately
        if (displayedContent.length === targetContent.length) {
            const typingEl = document.getElementById(typingId);
            if (typingEl) typingEl.remove();
            
            // Ensure bubble is created even if no tokens were yielded
            if (!hasCreatedBubble) {
                appendStreamingPlaceholder(activeMessageId);
                hasCreatedBubble = true;
            }
            
            finalizeStreamingMessage(activeMessageId, targetContent, sources);
            
            const botMsg = { 
                sender: 'bot', 
                text: targetContent, 
                sources: sources, 
                timestamp: new Date().toISOString()
            };
            state.localHistory[state.activeWorkspaceId].push(botMsg);
        }

    } catch (err) {
        console.error('Chat endpoint failed:', err);
        const typingEl = document.getElementById(typingId);
        if (typingEl) typingEl.remove();

        const botErrorMsg = {
            sender: 'bot',
            text: '⚠️ Network connection lost or interrupted.',
            sources: [],
            timestamp: new Date().toISOString()
        };
        state.localHistory[state.activeWorkspaceId].push(botErrorMsg);
        appendMessage('bot', botErrorMsg.text);
    } finally {
        state.isSending = false;
        dom.btnSend.disabled = dom.chatInput.value.trim().length === 0;
    }
}

// ----------------------------------------------------
// UI RENDERING ENGINE
// ----------------------------------------------------

function renderWorkspaces() {
    dom.workspaceList.innerHTML = '';
    state.workspaces.forEach(chat => {
        const item = document.createElement('div');
        item.className = `workspace-item ${chat.chat_id === state.activeWorkspaceId ? 'active' : ''}`;
        item.dataset.id = chat.chat_id;
        item.addEventListener('click', () => selectWorkspace(chat.chat_id));

        const content = document.createElement('div');
        content.className = 'workspace-item-content';

        // Custom Chat Icon
        content.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path stroke-linecap="round" stroke-linejoin="round" d="M20.25 8.511c.083.298.125.607.125.921 0 2.253-1.859 4.095-4.148 4.095-1.12 0-2.133-.448-2.88-1.168L9.75 14.25m6-5.739L9.75 11.25m0-4.5h.008v.008H9.75V6.75zM3.75 13.5h.008v.008H3.75V13.5zm3 3h.008v.008H6.75v-.008zm1.5 1.5h.008v.008H8.25v-.008zm3 1.5h.008v.008h-.008v-.008zm-9-6a3 3 0 013-3h12.75a3 3 0 013 3v5.25a3 3 0 01-3 3H5.25a3 3 0 01-3-3V10.5z" />
            </svg>
            <span class="workspace-title">${escapeHTML(chat.title)}</span>
        `;

        item.appendChild(content);

        // Workspace action buttons
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'workspace-actions';

        // Rename button
        const renameBtn = document.createElement('button');
        renameBtn.className = 'btn-rename';
        renameBtn.title = 'Rename Workspace';
        renameBtn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path stroke-linecap="round" stroke-linejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L6.832 19.82a4.5 4.5 0 01-1.897 1.13l-2.685.8.8-2.685a4.5 4.5 0 011.13-1.897L16.863 4.487zm0 0L19.5 7.125" />
            </svg>
        `;
        renameBtn.addEventListener('click', (e) => renameWorkspace(chat.chat_id, chat.title, e));
        actionsDiv.appendChild(renameBtn);

        // Delete button
        const delBtn = document.createElement('button');
        delBtn.className = 'btn-delete';
        delBtn.title = 'Delete Workspace';
        delBtn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path stroke-linecap="round" stroke-linejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
            </svg>
        `;
        delBtn.addEventListener('click', (e) => deleteWorkspace(chat.chat_id, e));
        actionsDiv.appendChild(delBtn);

        item.appendChild(actionsDiv);
        dom.workspaceList.appendChild(item);
    });
}

function renderDocuments() {
    dom.documentList.innerHTML = '';
    
    if (state.documents.length === 0) {
        dom.documentList.innerHTML = `
            <div class="empty-docs-state">
                <p>No documents uploaded to this workspace yet.</p>
            </div>
        `;
        return;
    }

    state.documents.forEach(doc => {
        const card = document.createElement('div');
        card.className = 'doc-card';

        const isProcessing = doc.status.startsWith('Processing') || doc.status.includes('Uploading');
        const statusBadgeClass = isProcessing ? 'processing' : 'indexed';
        const displayStatus = doc.status.replace('Processing: ', '');

        card.innerHTML = `
            <div class="doc-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                </svg>
            </div>
            <div class="doc-info">
                <div class="doc-name" title="${escapeHTML(doc.filename)}">${escapeHTML(doc.filename)}</div>
                <div class="doc-stats">
                    <span>${doc.char_count > 0 ? formatNumber(doc.char_count) + ' chars' : '0 chars'}</span>
                    <span>•</span>
                    <span>${doc.chunk_count > 0 ? doc.chunk_count + ' chunks' : '0 chunks'}</span>
                </div>
                <span class="doc-status-badge ${statusBadgeClass}">${escapeHTML(displayStatus)}</span>
            </div>
        `;

        const delBtn = document.createElement('button');
        delBtn.className = 'doc-delete-btn';
        delBtn.title = 'Delete Document';
        delBtn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path stroke-linecap="round" stroke-linejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
            </svg>
        `;
        delBtn.addEventListener('click', () => deleteDocument(doc.id));
        card.appendChild(delBtn);

        dom.documentList.appendChild(card);
    });
}

function renderMessages() {
    dom.chatMessages.innerHTML = '';
    
    const history = state.localHistory[state.activeWorkspaceId] || [];
    if (history.length === 0) {
        dom.chatMessages.innerHTML = `
            <div class="welcome-message-card">
                <div class="welcome-icon">💬</div>
                <h3>Welcome to ${escapeHTML(state.activeWorkspaceTitle)}</h3>
                <p>Ask a question about the document context linked to this workspace. The assistant uses semantic embeddings to retrieve grounded answers from your files.</p>
            </div>
        `;
        return;
    }

    history.forEach(msg => {
        appendMessage(msg.sender, msg.text, msg.sources, msg.timestamp);
    });
}

function appendMessage(sender, text, sources = [], timestamp = new Date().toISOString()) {
    const row = document.createElement('div');
    row.className = `message-row ${sender}`;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    // Parse simple markdown/text styles safely
    const formattedText = parseMessageText(text);

    bubble.innerHTML = `
        <div class="message-content">${formattedText}</div>
    `;

    // Render Sources section for bot responses if available
    if (sender === 'bot' && sources && sources.length > 0) {
        const sourcesSection = document.createElement('div');
        sourcesSection.className = 'sources-toggle';

        const toggleBtn = document.createElement('button');
        toggleBtn.className = 'btn-sources';
        toggleBtn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
            </svg>
            <span>View Source Passages (${sources.length})</span>
        `;

        const sourcesContainer = document.createElement('div');
        sourcesContainer.className = 'sources-container';
        sourcesContainer.style.display = 'none';

        sources.forEach(src => {
            const card = document.createElement('div');
            card.className = 'source-card';
            card.innerHTML = `
                <div class="source-card-header">
                    <span class="source-title" title="${escapeHTML(src.source)}">${escapeHTML(src.source)}</span>
                    <span class="source-score">${Math.round(src.score)}% match</span>
                </div>
                <div class="source-content">${escapeHTML(src.content)}</div>
            `;
            sourcesContainer.appendChild(card);
        });

        toggleBtn.addEventListener('click', () => {
            const isClosed = sourcesContainer.style.display === 'none';
            sourcesContainer.style.display = isClosed ? 'flex' : 'none';
            toggleBtn.className = `btn-sources ${isClosed ? 'open' : ''}`;
        });

        sourcesSection.appendChild(toggleBtn);
        sourcesSection.appendChild(sourcesContainer);
        bubble.appendChild(sourcesSection);
    }

    // Add metadata/timestamp
    const meta = document.createElement('div');
    meta.className = 'message-meta';
    const timeStr = new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    meta.innerHTML = `
        <span>${sender === 'user' ? 'You' : 'Assistant'}</span>
        <span>•</span>
        <span>${timeStr}</span>
    `;
    bubble.appendChild(meta);

    row.appendChild(bubble);
    dom.chatMessages.appendChild(row);
    scrollToBottom();
}

function appendTypingPlaceholder(id) {
    const row = document.createElement('div');
    row.className = 'message-row bot';
    row.id = id;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    bubble.innerHTML = `
        <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
        </div>
        <div class="message-meta">
            <span>Assistant is thinking...</span>
        </div>
    `;

    row.appendChild(bubble);
    dom.chatMessages.appendChild(row);
    scrollToBottom();
}

// ----------------------------------------------------
// HELPER UTILITIES
// ----------------------------------------------------

function parseMessageText(text) {
    if (!text) return '';
    
    const lines = text.split('\n');
    let html = '';
    
    let inCodeBlock = false;
    let codeContent = [];
    let codeLang = '';
    
    let inList = false;
    let listType = ''; // 'ul' or 'ol'
    
    const inlineParse = (str) => {
        // Bold (**text**)
        str = str.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Italic (*text*)
        str = str.replace(/\*(.*?)\*/g, '<em>$1</em>');
        // Inline code (`code`)
        str = str.replace(/`(.*?)`/g, '<code class="inline-code">$1</code>');
        return str;
    };
    
    const closeList = () => {
        if (inList) {
            html += `</${listType}>`;
            inList = false;
            listType = '';
        }
    };
    
    for (let i = 0; i < lines.length; i++) {
        let line = lines[i];
        
        // Handle fenced code blocks
        if (line.trim().startsWith('```')) {
            if (inCodeBlock) {
                // Close code block
                inCodeBlock = false;
                const codeEscaped = escapeHTML(codeContent.join('\n'));
                const blockId = 'code-' + Math.random().toString(36).substr(2, 9);
                html += `
                    <div class="code-block-wrapper">
                        <div class="code-block-header">
                            <span class="code-block-lang">${escapeHTML(codeLang || 'code')}</span>
                            <button class="btn-copy-code" onclick="copyToClipboard('${blockId}')">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:14px;height:14px;">
                                    <path stroke-linecap="round" stroke-linejoin="round" d="M8.25 7.5V6.108c0-1.135.845-2.098 1.976-2.192.373-.03.748-.057 1.123-.08M15.75 18H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08M15.75 18.75v-1.875a3.375 3.375 0 00-3.375-3.375h-1.5a1.125 1.125 0 01-1.125-1.125v-1.5a3.375 3.375 0 00-3.375-3.375H3.75m3.75 9H3.75m12 0h.008v.008h-.008v-.008z" />
                                </svg>
                                <span>Copy</span>
                            </button>
                        </div>
                        <pre><code id="${blockId}">${codeEscaped}</code></pre>
                    </div>
                `;
                codeContent = [];
                codeLang = '';
            } else {
                // Open code block
                closeList();
                inCodeBlock = true;
                codeLang = line.trim().substring(3).trim();
            }
            continue;
        }
        
        if (inCodeBlock) {
            codeContent.push(line);
            continue;
        }
        
        // Handle Horizontal Rules
        if (line.trim() === '---' || line.trim() === '***') {
            closeList();
            html += '<hr>';
            continue;
        }
        
        // Handle Headings (H1 to H4)
        const headingMatch = line.match(/^(#{1,4})\s+(.*)$/);
        if (headingMatch) {
            closeList();
            const level = headingMatch[1].length;
            const content = inlineParse(escapeHTML(headingMatch[2]));
            html += `<h${level}>${content}</h${level}>`;
            continue;
        }
        
        // Handle Unordered Lists (- or *)
        const ulMatch = line.match(/^(\s*)([-*])\s+(.*)$/);
        if (ulMatch) {
            const content = inlineParse(escapeHTML(ulMatch[3]));
            if (!inList || listType !== 'ul') {
                closeList();
                html += '<ul>';
                inList = true;
                listType = 'ul';
            }
            html += `<li>${content}</li>`;
            continue;
        }
        
        // Handle Ordered Lists (1. item)
        const olMatch = line.match(/^(\s*)(\d+)\.\s+(.*)$/);
        if (olMatch) {
            const content = inlineParse(escapeHTML(olMatch[3]));
            if (!inList || listType !== 'ol') {
                closeList();
                html += '<ol>';
                inList = true;
                listType = 'ol';
            }
            html += `<li>${content}</li>`;
            continue;
        }
        
        // Handle empty lines (paragraph separations)
        if (line.trim() === '') {
            closeList();
            continue;
        }
        
        // Standard body line
        closeList();
        const inlineContent = inlineParse(escapeHTML(line));
        html += `<p>${inlineContent}</p>`;
    }
    
    closeList();
    return html;
}

function copyToClipboard(elementId) {
    const el = document.getElementById(elementId);
    if (!el) return;
    
    const text = el.innerText || el.textContent;
    
    const onSuccess = () => {
        const btn = el.closest('.code-block-wrapper').querySelector('.btn-copy-code');
        if (btn) {
            const originalHTML = btn.innerHTML;
            btn.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="width:14px;height:14px;color:var(--color-success);">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                </svg>
                <span style="color:var(--color-success);">Copied!</span>
            `;
            setTimeout(() => {
                btn.innerHTML = originalHTML;
            }, 2000);
        }
    };

    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(onSuccess).catch(err => {
            console.warn('navigator.clipboard failed, trying fallback:', err);
            fallbackCopyToClipboard(text, onSuccess);
        });
    } else {
        fallbackCopyToClipboard(text, onSuccess);
    }
}

function fallbackCopyToClipboard(text, onSuccess) {
    try {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        // Prevent scrolling to bottom
        textarea.style.top = '0';
        textarea.style.left = '0';
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        const successful = document.execCommand('copy');
        document.body.removeChild(textarea);
        if (successful) {
            onSuccess();
        } else {
            console.error('Fallback copy command was unsuccessful');
        }
    } catch (err) {
        console.error('Fallback copy failed:', err);
    }
}

function appendStreamingPlaceholder(id) {
    const row = document.createElement('div');
    row.className = 'message-row bot';
    row.id = id;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble streaming';

    bubble.innerHTML = `
        <div class="message-content"><span class="streaming-cursor"></span></div>
    `;

    row.appendChild(bubble);
    dom.chatMessages.appendChild(row);
    scrollToBottom();
    return bubble;
}

function updateStreamingBubble(id, content) {
    const row = document.getElementById(id);
    if (!row) return;
    const contentDiv = row.querySelector('.message-content');
    if (!contentDiv) return;
    
    const formatted = parseMessageText(content);
    contentDiv.innerHTML = `${formatted}<span class="streaming-cursor"></span>`;
    scrollToBottom();
}

function finalizeStreamingMessage(id, content, sources = []) {
    const row = document.getElementById(id);
    if (!row) return;
    
    const bubble = row.querySelector('.message-bubble');
    if (bubble) {
        bubble.classList.remove('streaming');
    }
    
    const contentDiv = row.querySelector('.message-content');
    if (contentDiv) {
        const formatted = parseMessageText(content);
        contentDiv.innerHTML = formatted;
    }
    
    if (sources && sources.length > 0 && bubble) {
        const sourcesSection = document.createElement('div');
        sourcesSection.className = 'sources-toggle';

        const toggleBtn = document.createElement('button');
        toggleBtn.className = 'btn-sources';
        toggleBtn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
            </svg>
            <span>View Source Passages (${sources.length})</span>
        `;

        const sourcesContainer = document.createElement('div');
        sourcesContainer.className = 'sources-container';
        sourcesContainer.style.display = 'none';

        sources.forEach(src => {
            const card = document.createElement('div');
            card.className = 'source-card';
            card.innerHTML = `
                <div class="source-card-header">
                    <span class="source-title" title="${escapeHTML(src.source)}">${escapeHTML(src.source)}</span>
                    <span class="source-score">${Math.round(src.score)}% match</span>
                </div>
                <div class="source-content">${escapeHTML(src.content)}</div>
            `;
            sourcesContainer.appendChild(card);
        });

        toggleBtn.addEventListener('click', () => {
            const isClosed = sourcesContainer.style.display === 'none';
            sourcesContainer.style.display = isClosed ? 'flex' : 'none';
            toggleBtn.className = `btn-sources ${isClosed ? 'open' : ''}`;
        });

        sourcesSection.appendChild(toggleBtn);
        sourcesSection.appendChild(sourcesContainer);
        bubble.appendChild(sourcesSection);
    }
    
    if (bubble) {
        const meta = document.createElement('div');
        meta.className = 'message-meta';
        const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        meta.innerHTML = `
            <span>Assistant</span>
            <span>•</span>
            <span>${timeStr}</span>
        `;
        bubble.appendChild(meta);
    }
    scrollToBottom();
}

function escapeHTML(str) {
    if (!str) return '';
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function formatNumber(num) {
    return new Intl.NumberFormat().format(num);
}

function scrollToBottom() {
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
}
