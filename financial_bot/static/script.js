// static/script.js
document.addEventListener('DOMContentLoaded', () => {
    // Main Chat Elements
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const chatContainer = document.getElementById('chat-container');
    const loadingIndicator = document.getElementById('loading');
    const converter = new showdown.Converter();

    // Portfolio Analysis Elements
    const portfolioUploadBtn = document.getElementById('upload-btn');
    const portfolioFileInput = document.getElementById('file-input');

    // User Library Elements
    const uploadDocBtn = document.getElementById('upload-doc-btn');
    const userFileInput = document.getElementById('user-file-input');
    const userLibraryList = document.getElementById('user-library-list');
    const searchScopeToggle = document.getElementById('search-scope-toggle');

    // Engagement Feature Buttons
    const scamQuizBtn = document.getElementById('scam-quiz-btn');
    const sipCalculatorBtn = document.getElementById('sip-calculator-btn');
    const mythBusterBtn = document.getElementById('myth-buster-btn');

    // Modal Elements
    const scamModal = document.getElementById('scam-modal');
    const closeScamModalBtn = document.getElementById('close-scam-modal');
    const scamQuestionEl = document.getElementById('scam-question');
    const scamChoiceBtn = document.getElementById('scam-choice-btn');
    const legitChoiceBtn = document.getElementById('legit-choice-btn');
    const scamFeedbackEl = document.getElementById('scam-feedback');
    const nextScamBtn = document.getElementById('next-scam-btn');
    let currentScamQuestion = null;

    const sipModal = document.getElementById('sip-modal');
    const closeSipModalBtn = document.getElementById('close-sip-modal');
    const calculateSipBtn = document.getElementById('calculate-sip-btn');
    const sipResultEl = document.getElementById('sip-result');
    const sipGoalInput = document.getElementById('sip-goal');
    const sipAmountInput = document.getElementById('sip-amount');
    const sipYearsInput = document.getElementById('sip-years');
    let sipChart = null;

    const mythModal = document.getElementById('myth-modal');
    const closeMythModalBtn = document.getElementById('close-myth-modal');
    const mythStatementEl = document.getElementById('myth-statement');
    const mythChoiceBtn = document.getElementById('myth-choice-btn');
    const factChoiceBtn = document.getElementById('fact-choice-btn');
    const mythFeedbackEl = document.getElementById('myth-feedback');
    const nextMythBtn = document.getElementById('next-myth-btn');
    let currentMyth = null;

    // ---- DASHBOARD BUTTON: create if missing and wire navigation ----
    const ensureDashboardButton = () => {
        // Add a sidebar Dashboard button if missing
        try {
            const navContainer = (scamQuizBtn && scamQuizBtn.parentElement) || document.querySelector('nav');
            if (navContainer && !document.getElementById('dashboard-btn')) {
                const dashboardBtn = document.createElement('button');
                dashboardBtn.id = 'dashboard-btn';
                dashboardBtn.className = 'flex items-center gap-3 py-2 px-3 rounded-lg hover:bg-gray-800 transition';
                dashboardBtn.innerHTML = '📊 <span class="hidden md:inline">Dashboard</span>';
                // put at end of nav
                navContainer.appendChild(dashboardBtn);
                dashboardBtn.addEventListener('click', () => {
                    // navigate to dashboard
                    window.location.href = '/dashboard';
                });
            }

            // Add top bar link handler if present
            const topLink = document.getElementById('dashboard-top-link') || document.querySelector('a[href="/dashboard"]');
            if (topLink) {
                topLink.addEventListener('click', (e) => {
                    // allow default navigation; but keep this to ensure same behavior across browsers
                    // if you want SPA behavior, replace this with fetch + render.
                });
            }
        } catch (e) {
            console.warn('ensureDashboardButton error:', e);
        }
    };
    ensureDashboardButton();

    // Small helper to add messages
    const addMessage = (sender, message, isHtml = false) => {
        if (!chatContainer) return;
        const messageWrapper = document.createElement('div');
        messageWrapper.classList.add('flex', 'mb-2');

        const messageDiv = document.createElement('div');
        messageDiv.classList.add('p-3', 'rounded-2xl', 'max-w-md', 'shadow');

        if (sender === 'user') {
            messageWrapper.classList.add('justify-end');
            messageDiv.classList.add('bg-blue-700', 'text-white');
            messageDiv.innerHTML = `<p>${message}</p>`;
        } else {
            messageWrapper.classList.add('justify-start');
            messageDiv.classList.add('bg-indigo-500', 'text-white');
            messageDiv.innerHTML = isHtml ? message : converter.makeHtml(message || '');
        }

        messageWrapper.appendChild(messageDiv);
        chatContainer.appendChild(messageWrapper);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    };

    // SEBI circulars loader
    async function loadSebicirculars() {
        try {
            const res = await fetch('/sebi/circulars');
            const json = await res.json();
            const container = document.querySelector('.news-section');
            if (!container) return;
            if (!json.circulars?.length) {
                container.innerHTML = '<p>No SEBI circulars found.</p>';
                return;
            }
            container.innerHTML = '';
            json.circulars.forEach(c => {
                const div = document.createElement('div');
                div.className = 'p-3 bg-gray-50 rounded-lg shadow-sm hover:shadow-md transition';
                div.innerHTML = `<a href="${c.url}" target="_blank" class="text-indigo-600 hover:underline text-sm flex justify-between">
                                    <span>${c.date}</span><span>${c.title}</span>
                                 </a>`;
                container.appendChild(div);
            });
        } catch (e) {
            console.error("Failed to load SEBI circulars:", e);
        }
    }
    loadSebicirculars();
    setInterval(loadSebicirculars, 24 * 60 * 60 * 1000);

    // Live market updater
    async function updateLiveMarket() {
        try {
            const response = await fetch('/market/live');
            const data = await response.json();
            if (data.error) return console.error(data.error);

            const updateElem = (id, info) => {
                const el = document.getElementById(id);
                if (!el) return;
                if (!info || info.price === null || info.price === undefined) {
                    el.innerHTML = `--`;
                    return;
                }
                const arrow = info.change >= 0 ? '🔺' : '🔻';
                const color = info.change >= 0 ? 'text-green-600' : 'text-red-600';
                el.innerHTML = `<span class="${color} font-bold">₹${info.price} ${arrow} (${info.pct_change}%)</span>`;
            };

            updateElem('nifty-price', data['NIFTY 50']);
            updateElem('sensex-price', data['SENSEX']);
            updateElem('cdsl-price', data['CDSL']);
            updateElem('KFINTECH-price', data['KFINTECH']);
        } catch (err) {
            console.error("Failed to fetch market data:", err);
        }
    }
    updateLiveMarket();
    setInterval(updateLiveMarket, 180000);

    // Ask handler
    const handleSend = async () => {
        if (!userInput) return;
        const question = userInput.value.trim();
        if (!question) return;
        addMessage('user', question);
        userInput.value = '';
        if (loadingIndicator) loadingIndicator.classList.remove('hidden');
        const scope = searchScopeToggle && searchScopeToggle.checked ? 'user_only' : 'all';
        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question, scope: scope }),
            });
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(errorText || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            addMessage('bot', data.answer || data.error || 'No answer.');
        } catch (error) {
            addMessage('bot', `Sorry, an error occurred: ${error.message}`);
        } finally {
            if (loadingIndicator) loadingIndicator.classList.add('hidden');
        }
    };

    // Portfolio analysis (file upload)
    const handlePortfolioUpload = async () => {
        const file = portfolioFileInput && portfolioFileInput.files[0];
        if (!file) {
            addMessage('bot', 'Please select a portfolio file first.');
            return;
        }

        addMessage('user', `Analyzing portfolio: ${file.name}`);
        if (loadingIndicator) loadingIndicator.classList.remove('hidden');

        const formData = new FormData();
        formData.append('portfolioFile', file);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json().catch(()=>({}));
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            const analysisWindow = window.open('', '_blank');
            if (analysisWindow) {
                analysisWindow.document.write(createPortfolioPage(data.analysis_markdown, data.chart_data));
                analysisWindow.document.close();
            } else {
                addMessage('bot', 'Could not open analysis window. Please disable your pop-up blocker.');
            }

        } catch (error) {
            console.error('Error:', error);
            addMessage('bot', `Sorry, an error occurred during analysis: ${error.message}`);
        } finally {
            if (loadingIndicator) loadingIndicator.classList.add('hidden');
            if (portfolioFileInput) portfolioFileInput.value = '';
        }
    };

    // Create portfolio analysis page
    const createPortfolioPage = (analysisMarkdown, chartData) => {
        const analysisHtml = converter.makeHtml(analysisMarkdown || '');
        const chartDataJson = JSON.stringify(chartData || {});
        return `
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Portfolio Analysis - SEBI Saathi</title>
                <script src="https://cdn.tailwindcss.com"></script>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body { font-family: 'Inter', sans-serif; }
                    .analysis-section { background-color: #f9fafb; border: 1px solid #e5e7eb; padding: 1.5rem; border-radius: 0.75rem; }
                </style>
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
            </head>
            <body class="bg-gray-100">
                <div class="container mx-auto p-8">
                    <div class="w-full max-w-4xl mx-auto bg-white rounded-2xl shadow-2xl p-6">
                        <div class="border-b pb-4 mb-6">
                            <h1 class="text-3xl font-bold text-gray-800 text-center">SEBI Saathi 🇮🇳</h1>
                            <p class="text-center text-gray-500">Portfolio Health Check</p>
                        </div>
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div class="analysis-section">
                                <h2 class="text-2xl font-semibold text-gray-700 mb-4">Sector Allocation</h2>
                                <canvas id="portfolioChart"></canvas>
                            </div>
                            <div class="analysis-section">
                                <h2 class="text-2xl font-semibold text-gray-700 mb-4">AI-Powered Analysis</h2>
                                <div class="text-gray-600 space-y-4">${analysisHtml}</div>
                            </div>
                        </div>
                    </div>
                </div>
                <script>
                    document.addEventListener('DOMContentLoaded', () => {
                        const chartData = ${chartDataJson};
                        if (chartData && Object.keys(chartData).length > 0) {
                            const ctx = document.getElementById('portfolioChart').getContext('2d');
                            new Chart(ctx, {
                                type: 'pie',
                                data: {
                                    labels: Object.keys(chartData),
                                    datasets: [{
                                        data: Object.values(chartData),
                                        backgroundColor: ['#4F46E5', '#7C3AED', '#EC4899', '#F59E0B', '#10B981', '#3B82F6'],
                                    }]
                                },
                                options: { responsive: true, plugins: { legend: { position: 'top' } } }
                            });
                        }
                    });
                </script>
            </body>
            </html>
        `;
    };

    // User library helpers
    const updateUserLibrary = (files) => {
        if (!userLibraryList) return;
        userLibraryList.innerHTML = '';
        if (files && files.length > 0) {
            const list = document.createElement('ul');
            list.className = 'space-y-2';
            files.forEach(file => {
                const listItem = document.createElement('li');
                listItem.className = 'p-2 bg-gray-700 rounded text-sm flex items-center justify-between';
                const fileNameSpan = document.createElement('span');
                fileNameSpan.className = 'truncate text-white';
                fileNameSpan.textContent = file;
                const deleteBtn = document.createElement('button');
                deleteBtn.className = 'ml-2 text-red-200 hover:text-red-400';
                deleteBtn.innerHTML = 'Delete';
                deleteBtn.title = 'Delete file';
                deleteBtn.onclick = () => handleDeleteFile(file);
                listItem.appendChild(fileNameSpan);
                listItem.appendChild(deleteBtn);
                list.appendChild(listItem);
            });
            userLibraryList.appendChild(list);
        } else {
            userLibraryList.innerHTML = '<p class="text-gray-400 text-sm">Your library is empty.</p>';
        }
    };

    const loadUserLibrary = async () => {
        try {
            const response = await fetch('/get_user_library');
            const data = await response.json();
            updateUserLibrary(data.user_library);
        } catch (e) { console.error("Could not load user library", e); }
    };

    const handleDocUpload = async () => {
        const file = userFileInput && userFileInput.files[0];
        if (!file) return;
        addMessage('bot', `Processing <strong>${file.name}</strong>...`, true);
        if (loadingIndicator) loadingIndicator.classList.remove('hidden');
        const formData = new FormData();
        formData.append('userFile', file);
        try {
            const response = await fetch('/upload_and_ingest', { method: 'POST', body: formData });
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(errorText || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            if (data.success) {
                addMessage('bot', `<strong>${file.name}</strong> added to your library.`, true);
                updateUserLibrary(data.user_library);
            } else { throw new Error(data.error); }
        } catch (error) {
            addMessage('bot', `Error processing file: ${error.message}`);
        } finally {
            if (loadingIndicator) loadingIndicator.classList.add('hidden');
            if (userFileInput) userFileInput.value = '';
        }
    };

    const handleDeleteFile = async (filename) => {
        if (!confirm(`Are you sure you want to delete "${filename}"?`)) return;
        addMessage('bot', `Deleting <strong>${filename}</strong>...`, true);
        try {
            const response = await fetch('/delete_user_file', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: filename }),
            });
            const data = await response.json();
            if (data.success) {
                addMessage('bot', `<strong>${filename}</strong> deleted.`, true);
                updateUserLibrary(data.user_library);
            } else { throw new Error(data.error); }
        } catch (error) {
            addMessage('bot', `Error deleting file: ${error.message}`);
        }
    };

    // Scam / SIP / Myth feature handlers
    const loadScamQuestion = async () => {
        if (!scamQuestionEl) return;
        scamFeedbackEl && (scamFeedbackEl.innerHTML = '');
        scamQuestionEl.textContent = 'Loading...';
        const response = await fetch('/quiz/next_question');
        currentScamQuestion = await response.json();
        scamQuestionEl.textContent = currentScamQuestion.message;
    };
    const checkScamAnswer = (userChoice) => {
        if (!currentScamQuestion) return;
        const isCorrect = userChoice.toLowerCase() === currentScamQuestion.type;
        scamFeedbackEl.innerHTML = `<p class="font-bold ${isCorrect ? 'text-green-600' : 'text-red-600'}">${isCorrect ? 'Correct!' : 'Incorrect.'} It was ${currentScamQuestion.type}.</p><p class="mt-2">${currentScamQuestion.explanation}</p>`;
    };

    const calculateSip = async () => {
        const goal = sipGoalInput?.value, amount = sipAmountInput?.value, years = sipYearsInput?.value;
        if (!goal || !amount || !years) { sipResultEl.textContent = 'Please fill all fields.'; return; }
        const response = await fetch('/calculate_sip', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ amount, years }),
        });
        const data = await response.json();
        if (data.error) { sipResultEl.textContent = data.error; return; }
        sipResultEl.innerHTML = `To reach <strong>₹${parseInt(amount).toLocaleString('en-IN')}</strong> for your <strong>${goal}</strong>, you need a monthly SIP of <strong>₹${data.monthly_sip.toLocaleString('en-IN')}</strong>.`;
        if (sipChart) sipChart.destroy();
        const ctx = document.getElementById('sipChart')?.getContext('2d');
        if (!ctx) return;
        sipChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.growth_data.map(d => `Year ${d.year}`),
                datasets: [{ label: 'Invested', data: data.growth_data.map(d => d.invested) }, { label: 'Value', data: data.growth_data.map(d => d.value) }]
            },
            options: { scales: { y: { beginAtZero: true } } }
        });
    };

    const loadMyth = async () => {
        if (!mythStatementEl) return;
        mythFeedbackEl && (mythFeedbackEl.innerHTML = '');
        mythStatementEl.textContent = 'Loading...';
        const response = await fetch('/get_myth');
        currentMyth = await response.json();
        mythStatementEl.textContent = currentMyth.statement;
    };
    const checkMythAnswer = (userChoice) => {
        if (!currentMyth) return;
        const isCorrect = userChoice === currentMyth.type;
        mythFeedbackEl.innerHTML = `<p class="font-bold ${isCorrect ? 'Correct!' : 'Not quite!'} It's a ${currentMyth.type}.</p><p class="mt-2">${currentMyth.explanation}</p>`;
        mythFeedbackEl.classList.add(isCorrect ? 'bg-green-100' : 'bg-red-100');
    };

    // --- Final Event Listeners ---
    addMessage('bot', 'Welcome to SEBI Saathi!');
    loadUserLibrary();

    sendBtn && sendBtn.addEventListener('click', handleSend);
    userInput && userInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') handleSend(); });
    uploadDocBtn && uploadDocBtn.addEventListener('click', () => userFileInput && userFileInput.click());
    userFileInput && userFileInput.addEventListener('change', handleDocUpload);
    portfolioUploadBtn && portfolioUploadBtn.addEventListener('click', () => portfolioFileInput && portfolioFileInput.click());
    portfolioFileInput && portfolioFileInput.addEventListener('change', handlePortfolioUpload);
    scamQuizBtn && scamQuizBtn.addEventListener('click', () => { scamModal && scamModal.classList.add('active'); loadScamQuestion(); });
    closeScamModalBtn && closeScamModalBtn.addEventListener('click', () => scamModal && scamModal.classList.remove('active'));
    nextScamBtn && nextScamBtn.addEventListener('click', loadScamQuestion);
    scamChoiceBtn && scamChoiceBtn.addEventListener('click', () => checkScamAnswer('scam'));
    legitChoiceBtn && legitChoiceBtn.addEventListener('click', () => checkScamAnswer('legit'));
    sipCalculatorBtn && sipCalculatorBtn.addEventListener('click', () => sipModal && sipModal.classList.add('active'));
    closeSipModalBtn && closeSipModalBtn.addEventListener('click', () => sipModal && sipModal.classList.remove('active'));
    calculateSipBtn && calculateSipBtn.addEventListener('click', calculateSip);
    mythBusterBtn && mythBusterBtn.addEventListener('click', () => { mythModal && mythModal.classList.add('active'); loadMyth(); });
    closeMythModalBtn && closeMythModalBtn.addEventListener('click', () => mythModal && mythModal.classList.remove('active'));
    nextMythBtn && nextMythBtn.addEventListener('click', loadMyth);
    mythChoiceBtn && mythChoiceBtn.addEventListener('click', () => checkMythAnswer('Myth'));
    factChoiceBtn && factChoiceBtn.addEventListener('click', () => checkMythAnswer('Fact'));

    // Logout button handler (if present)
    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', async () => {
            try {
                const resp = await fetch('/auth/logout', { method: 'POST' });
                if (resp.ok) {
                    addMessage('bot', 'Logged out. Redirecting to login...');
                    setTimeout(() => window.location.href = '/login', 300);
                } else {
                    let data = {};
                    try { data = await resp.json(); } catch(e) {}
                    addMessage('bot', `Logout failed: ${data.error || resp.statusText}`);
                }
            } catch (err) {
                console.error('Logout error', err);
                addMessage('bot', 'Logout failed (network error).');
            }
        });
    }

    // Sources UI
    const ensureSourcesUI = () => {
        const buttonsBar = scamQuizBtn ? scamQuizBtn.parentElement : null;
        if (buttonsBar && !document.getElementById('sources-btn')) {
            const sourcesBtn = document.createElement('button');
            sourcesBtn.id = 'sources-btn';
            sourcesBtn.className = 'bg-indigo-500 text-white font-semibold py-2 px-4 rounded-lg hover:bg-indigo-600 text-sm';
            sourcesBtn.textContent = 'Sources';
            buttonsBar.appendChild(sourcesBtn);
        }
        if (!document.getElementById('sources-modal')) {
            const modal = document.createElement('div');
            modal.id = 'sources-modal';
            modal.className = 'modal fixed inset-0 bg-gray-800 bg-opacity-75 items-center justify-center';
            modal.innerHTML = `
                <div class="bg-white rounded-lg p-8 max-w-xl w-full">
                    <div class="flex items-center justify-between mb-4">
                        <h2 class="text-2xl font-bold">Knowledge Sources</h2>
                        <button id="close-sources-modal" class="text-gray-600 hover:text-gray-900">Close</button>
                    </div>
                    <p class="text-sm text-gray-500 mb-3">These files (PDF/TXT/CSV) are indexed under data/rag_sources and used to answer questions.</p>
                    <div id="sources-list" class="space-y-2 max-h-80 overflow-y-auto"></div>
                </div>
            `;
            document.body.appendChild(modal);
        }
    };
    ensureSourcesUI();

    const sourcesBtn = document.getElementById('sources-btn');
    const sourcesModal = document.getElementById('sources-modal');
    const closeSourcesModalBtn = document.getElementById('close-sources-modal');
    const sourcesListEl = document.getElementById('sources-list');

    const loadSources = async () => {
        if (!sourcesListEl) return;
        sourcesListEl.innerHTML = '<div class="text-gray-500">Loading…</div>';
        try {
            const resp = await fetch('/sources');
            const data = await resp.json();
            if (!data.sources || data.sources.length === 0) {
                sourcesListEl.innerHTML = '<div class="text-gray-500">No PDF sources found.</div>';
                return;
            }
            sourcesListEl.innerHTML = '';
            data.sources.forEach(src => {
                const row = document.createElement('div');
                row.className = 'flex items-center justify-between p-2 border rounded';
                const name = document.createElement('div');
                name.className = 'text-sm truncate pr-3';
                name.textContent = src.name;
                const ext = (src.name.split('.').pop() || '').toLowerCase();
                const link = document.createElement('a');
                link.href = src.url;
                link.target = '_blank';
                link.rel = 'noopener noreferrer';
                link.className = 'text-indigo-600 hover:underline text-sm flex-shrink-0';
                link.textContent = ext === 'pdf' ? 'Open' : 'View';
                row.appendChild(name);
                row.appendChild(link);
                sourcesListEl.appendChild(row);
            });
        } catch (e) {
            sourcesListEl.innerHTML = '<div class="text-red-600">Failed to load sources.</div>';
        }
    };

    if (sourcesBtn && sourcesModal && closeSourcesModalBtn) {
        sourcesBtn.addEventListener('click', () => {
            sourcesModal.classList.add('active');
            loadSources();
        });
        closeSourcesModalBtn.addEventListener('click', () => sourcesModal.classList.remove('active'));
    }

    // ensure dashboard button exists and works (idempotent)
    ensureDashboardButton();

    // done
});
