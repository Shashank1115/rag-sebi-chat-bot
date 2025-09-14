// static/script.js
document.addEventListener('DOMContentLoaded', () => {
    // converter for markdown -> html
    const converter = new showdown.Converter();

    // --- Main elements ---
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const chatContainer = document.getElementById('chat-container');
    const loadingIndicator = document.getElementById('loading');

    // portfolio analysis elements
    const portfolioUploadBtn = document.getElementById('upload-btn');
    const portfolioFileInput = document.getElementById('file-input');

    // user library
    const uploadDocBtn = document.getElementById('upload-doc-btn');
    const userFileInput = document.getElementById('user-file-input');
    const userLibraryList = document.getElementById('user-library-list');
    const searchScopeToggle = document.getElementById('search-scope-toggle');

    // engagement buttons
    const scamQuizBtn = document.getElementById('scam-quiz-btn');
    const sipCalculatorBtn = document.getElementById('sip-calculator-btn');
    const mythBusterBtn = document.getElementById('myth-buster-btn');

    // modals & related
    const scamModal = document.getElementById('scam-modal');
    const closeScamModalBtn = document.getElementById('close-scam-modal');
    const scamQuestionEl = document.getElementById('scam-question');
    const scamChoiceBtn = document.getElementById('scam-choice-btn');
    const legitChoiceBtn = document.getElementById('legit-choice-btn');
    const scamFeedbackEl = document.getElementById('scam-feedback');
    const nextScamBtn = document.getElementById('next-scam-btn');

    const sipModal = document.getElementById('sip-modal');
    const closeSipModalBtn = document.getElementById('close-sip-modal');
    const calculateSipBtn = document.getElementById('calculate-sip-btn');
    const sipResultEl = document.getElementById('sip-result');
    const sipGoalInput = document.getElementById('sip-goal');
    const sipAmountInput = document.getElementById('sip-amount');
    const sipYearsInput = document.getElementById('sip-years');

    const mythModal = document.getElementById('myth-modal');
    const closeMythModalBtn = document.getElementById('close-myth-modal');
    const mythStatementEl = document.getElementById('myth-statement');
    const mythChoiceBtn = document.getElementById('myth-choice-btn');
    const factChoiceBtn = document.getElementById('fact-choice-btn');
    const mythFeedbackEl = document.getElementById('myth-feedback');
    const nextMythBtn = document.getElementById('next-myth-btn');

    // IPO modal elements (new)
    const ipoBtn = document.getElementById('ipo-btn');
    const ipoModal = document.getElementById('ipo-modal');
    const closeIpoModalBtn = document.getElementById('close-ipo-modal');
    const ipoForm = document.getElementById('ipo-form');
    const ipoFileInput = document.getElementById('ipo-file');
    const ipoContentTextarea = document.getElementById('ipo-content');
    const ipoReportDiv = document.getElementById('ipo-report');

    // sources
    const sourcesBtn = document.getElementById('sources-btn');
    const sourcesModal = document.getElementById('sources-modal');
    const closeSourcesModalBtn = document.getElementById('close-sources-modal');
    const sourcesListEl = document.getElementById('sources-list');

    // other UI
    const logoutBtn = document.getElementById('logout-btn');

    // helper: add chat message
    const addMessage = (sender, message, isHtml = false) => {
        if (!chatContainer) return;
        const wrapper = document.createElement('div');
        wrapper.className = 'flex mb-2';
        const bubble = document.createElement('div');
        bubble.className = 'p-3 rounded-2xl max-w-md shadow';

        if (sender === 'user') {
            wrapper.classList.add('justify-end');
            bubble.classList.add('bg-blue-700', 'text-white');
            bubble.innerHTML = `<p>${escapeHtml(message)}</p>`;
        } else {
            wrapper.classList.add('justify-start');
            bubble.classList.add('bg-indigo-500', 'text-white');
            bubble.innerHTML = isHtml ? message : converter.makeHtml(message || '');
        }

        wrapper.appendChild(bubble);
        chatContainer.appendChild(wrapper);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    };

    function escapeHtml(s) {
        if (!s) return '';
        return s.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    }

    // ------------------------------
    // SEBI circulars loader
    // ------------------------------
    async function loadSebicirculars() {
        try {
            const res = await fetch('/sebi/circulars');
            const json = await res.json();
            const container = document.querySelector('.news-section');
            if (!container) return;
            container.innerHTML = '';
            if (!json.circulars || json.circulars.length === 0) {
                container.innerHTML = '<p class="text-gray-500 text-sm">No SEBI circulars found.</p>';
                return;
            }
            json.circulars.forEach(c => {
                const div = document.createElement('div');
                div.className = 'p-3 bg-gray-50 rounded-lg shadow-sm hover:shadow-md transition';
                div.innerHTML = `<a href="${c.url}" target="_blank" class="text-indigo-600 hover:underline text-sm flex justify-between">
                                    <span>${c.date}</span><span>${c.title}</span>
                                 </a>`;
                container.appendChild(div);
            });
        } catch (e) {
            console.error('Failed to load SEBI circulars', e);
        }
    }
    loadSebicirculars();
    setInterval(loadSebicirculars, 24 * 60 * 60 * 1000);

    // ------------------------------
    // Live market updater
    // ------------------------------
    async function updateLiveMarket() {
        try {
            const resp = await fetch('/market/live');
            const data = await resp.json();
            if (!data) return;
            const updateElem = (id, info) => {
                const el = document.getElementById(id);
                if (!el) return;
                if (!info || info.price === null || info.price === undefined) {
                    el.innerHTML = `<div class="p-3 bg-gray-50 rounded">${id.replace('-', ' ')} — --</div>`;
                    return;
                }
                const arrow = info.change >= 0 ? '🔺' : '🔻';
                const colorClass = info.change >= 0 ? 'text-green-600' : 'text-red-600';
                el.innerHTML = `<div class="p-3 bg-gray-50 rounded flex justify-between"><span>${id.replace('-', ' ').toUpperCase()}</span><span class="${colorClass} font-bold">₹${info.price} ${arrow} (${info.pct_change}%)</span></div>`;
            };
            updateElem('nifty-price', data['NIFTY 50']);
            updateElem('sensex-price', data['SENSEX']);
            updateElem('cdsl-price', data['CDSL']);
            updateElem('KFINTECH-price', data['KFINTECH']);
        } catch (e) {
            console.error('Failed to update market', e);
        }
    }
    updateLiveMarket();
    setInterval(updateLiveMarket, 3 * 60 * 1000); // every 3 minutes

    // ------------------------------
    // ASK / Chat
    // ------------------------------
    const handleSend = async () => {
        if (!userInput) return;
        const question = userInput.value.trim();
        if (!question) return;
        addMessage('user', question);
        userInput.value = '';
        if (loadingIndicator) loadingIndicator.classList.remove('hidden');

        const scope = (searchScopeToggle && searchScopeToggle.checked) ? 'user_only' : 'all';
        try {
            const res = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, scope })
            });
            if (!res.ok) {
                const text = await res.text();
                throw new Error(text || `${res.status}`);
            }
            const j = await res.json();
            addMessage('bot', j.answer || j.error || 'No answer.', false);
        } catch (err) {
            console.error('ask error', err);
            addMessage('bot', `Sorry, an error occurred: ${err.message}`);
        } finally {
            if (loadingIndicator) loadingIndicator.classList.add('hidden');
        }
    };

    // ------------------------------
    // Portfolio upload
    // ------------------------------
    const handlePortfolioUpload = async () => {
        const file = portfolioFileInput && portfolioFileInput.files[0];
        if (!file) {
            addMessage('bot', 'Please select a portfolio file first.');
            return;
        }
        addMessage('user', `Analyzing portfolio: ${file.name}`);
        if (loadingIndicator) loadingIndicator.classList.remove('hidden');
        const fd = new FormData();
        fd.append('portfolioFile', file);
        try {
            const res = await fetch('/analyze', { method: 'POST', body: fd });
            if (!res.ok) {
                const err = await res.json().catch(()=>null);
                throw new Error(err?.error || `HTTP ${res.status}`);
            }
            const data = await res.json();
            const win = window.open('', '_blank');
            if (!win) {
                addMessage('bot', 'Could not open analysis window. Please allow popups.');
                return;
            }
            const html = createPortfolioPage(data.analysis_markdown, data.chart_data);
            win.document.open();
            win.document.write(html);
            win.document.close();
        } catch (e) {
            console.error('Portfolio analyze error', e);
            addMessage('bot', `Portfolio analysis failed: ${e.message}`);
        } finally {
            if (loadingIndicator) loadingIndicator.classList.add('hidden');
            if (portfolioFileInput) portfolioFileInput.value = '';
        }
    };

    function createPortfolioPage(md, chartData) {
        const analysisHtml = converter.makeHtml(md || '');
        const chartJson = JSON.stringify(chartData || {});
        return `
            <!doctype html>
            <html>
            <head>
                <meta charset="utf-8"/>
                <meta name="viewport" content="width=device-width,initial-scale=1"/>
                <title>Portfolio Analysis</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <link href="https://cdn.tailwindcss.com" rel="stylesheet">
            </head>
            <body class="p-6">
                <div class="max-w-4xl mx-auto">
                    <h1 class="text-2xl font-bold mb-2">Portfolio Health Check</h1>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div><canvas id="portfolioChart"></canvas></div>
                        <div class="prose">${analysisHtml}</div>
                    </div>
                </div>
                <script>
                    const chartData = ${chartJson};
                    if (chartData && Object.keys(chartData).length) {
                        const ctx = document.getElementById('portfolioChart').getContext('2d');
                        new Chart(ctx, {
                            type: 'pie',
                            data: {
                                labels: Object.keys(chartData),
                                datasets: [{ data: Object.values(chartData) }]
                            }
                        });
                    }
                </script>
            </body>
            </html>
        `;
    }

    // ------------------------------
    // User library
    // ------------------------------
    const updateUserLibrary = (files) => {
        if (!userLibraryList) return;
        userLibraryList.innerHTML = '';
        if (!files || files.length === 0) {
            userLibraryList.innerHTML = '<div class="text-xs text-gray-400">Your library is empty.</div>';
            return;
        }
        const ul = document.createElement('ul');
        ul.className = 'space-y-2 text-xs';
        files.forEach(f => {
            const li = document.createElement('li');
            li.className = 'flex justify-between items-center';
            const name = document.createElement('span');
            name.className = 'truncate';
            name.textContent = f;
            const del = document.createElement('button');
            del.className = 'text-red-400 text-xs';
            del.textContent = 'Delete';
            del.onclick = () => handleDeleteFile(f);
            li.appendChild(name);
            li.appendChild(del);
            ul.appendChild(li);
        });
        userLibraryList.appendChild(ul);
    };

    const loadUserLibrary = async () => {
        try {
            const res = await fetch('/get_user_library');
            const j = await res.json();
            updateUserLibrary(j.user_library || []);
        } catch (e) {
            console.error('Could not load user library', e);
        }
    };

    const handleDocUpload = async () => {
        const f = userFileInput && userFileInput.files[0];
        if (!f) return;
        addMessage('user', `Uploading ${f.name}`);
        if (loadingIndicator) loadingIndicator.classList.remove('hidden');
        const fd = new FormData();
        fd.append('userFile', f);
        try {
            const res = await fetch('/upload_and_ingest', { method: 'POST', body: fd });
            if (!res.ok) {
                const t = await res.text();
                throw new Error(t || res.status);
            }
            const j = await res.json();
            if (j.success) {
                addMessage('bot', `<strong>${f.name}</strong> added to library.`, true);
                updateUserLibrary(j.user_library || []);
            } else throw new Error(j.error || 'Upload failed');
        } catch (e) {
            console.error('upload error', e);
            addMessage('bot', `Upload failed: ${e.message}`);
        } finally {
            if (loadingIndicator) loadingIndicator.classList.add('hidden');
            if (userFileInput) userFileInput.value = '';
        }
    };

    const handleDeleteFile = async (filename) => {
        if (!confirm(`Delete ${filename}?`)) return;
        addMessage('bot', `Deleting ${filename}...`, true);
        try {
            const res = await fetch('/delete_user_file', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename })
            });
            const j = await res.json();
            if (j.success) {
                addMessage('bot', `<strong>${filename}</strong> deleted.`, true);
                updateUserLibrary(j.user_library || []);
            } else throw new Error(j.error || 'Delete failed');
        } catch (e) {
            console.error('delete error', e);
            addMessage('bot', `Delete failed: ${e.message}`);
        }
    };

    // ------------------------------
    // Scam / SIP / Myth handlers
    // ------------------------------
    let currentScamQuestion = null;
    async function loadScamQuestion() {
        if (!scamQuestionEl) return;
        scamFeedbackEl && (scamFeedbackEl.innerHTML = '');
        scamQuestionEl.textContent = 'Loading...';
        try {
            const res = await fetch('/quiz/next_question');
            currentScamQuestion = await res.json();
            scamQuestionEl.textContent = currentScamQuestion.message;
        } catch (e) {
            console.error('scam load error', e);
            scamQuestionEl.textContent = 'Failed to load question.';
        }
    }
    function checkScamAnswer(choice) {
        if (!currentScamQuestion) return;
        const isCorrect = choice.toLowerCase() === currentScamQuestion.type;
        scamFeedbackEl.innerHTML = `<p class="font-bold ${isCorrect ? 'text-green-600' : 'text-red-600'}">${isCorrect ? 'Correct!' : 'Incorrect.'} It was ${currentScamQuestion.type}.</p><p class="mt-2">${currentScamQuestion.explanation}</p>`;
    }

    async function calculateSip() {
        const goal = sipGoalInput?.value, amount = sipAmountInput?.value, years = sipYearsInput?.value;
        if (!goal || !amount || !years) { sipResultEl.textContent = 'Please fill fields.'; return; }
        try {
            const res = await fetch('/calculate_sip', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ amount, years })
            });
            const j = await res.json();
            if (j.error) {
                sipResultEl.textContent = j.error;
                return;
            }
            sipResultEl.innerHTML = `Monthly SIP: <strong>₹${j.monthly_sip.toLocaleString('en-IN')}</strong>`;
            // optional: draw chart if canvas present
            const ctx = document.getElementById('sipChart')?.getContext?.('2d');
            if (ctx && j.growth_data) {
                if (window._sipChart) window._sipChart.destroy();
                window._sipChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: j.growth_data.map(d => `Y${d.year}`),
                        datasets: [{ label: 'Invested', data: j.growth_data.map(d => d.invested) }, { label: 'Value', data: j.growth_data.map(d => d.value) }]
                    },
                    options: { scales: { y: { beginAtZero: true } } }
                });
            }
        } catch (e) {
            console.error('sip error', e);
            sipResultEl.textContent = 'Calculation failed.';
        }
    }

    let currentMyth = null;
    async function loadMyth() {
        if (!mythStatementEl) return;
        mythFeedbackEl && (mythFeedbackEl.innerHTML = '');
        mythStatementEl.textContent = 'Loading...';
        try {
            const res = await fetch('/get_myth');
            currentMyth = await res.json();
            mythStatementEl.textContent = currentMyth.statement;
        } catch (e) {
            console.error('myth load', e);
            mythStatementEl.textContent = 'Failed to load.';
        }
    }
    function checkMythAnswer(choice) {
        if (!currentMyth) return;
        const isCorrect = choice === currentMyth.type;
        mythFeedbackEl.innerHTML = `<p class="font-bold ${isCorrect ? 'text-green-600' : 'text-red-600'}">${isCorrect ? 'Correct!' : 'Not quite.'} It's a ${currentMyth.type}.</p><p class="mt-2">${currentMyth.explanation}</p>`;
    }

    // ------------------------------
    // Sources UI logic (keeps same behavior)
    // ------------------------------
    const ensureSourcesUI = () => {
        const buttonsBar = scamQuizBtn ? scamQuizBtn.parentElement : null;
        if (buttonsBar && !document.getElementById('sources-btn')) {
            const btn = document.createElement('button');
            btn.id = 'sources-btn';
            btn.className = 'bg-indigo-500 text-white font-semibold py-2 px-4 rounded-lg hover:bg-indigo-600 text-sm';
            btn.textContent = 'Sources';
            buttonsBar.appendChild(btn);
        }
        if (!document.getElementById('sources-modal')) {
            const modal = document.createElement('div');
            modal.id = 'sources-modal';
            modal.className = 'modal hidden fixed inset-0 bg-gray-800 bg-opacity-75 items-center justify-center';
            modal.innerHTML = `
                <div class="bg-white rounded-lg p-8 max-w-xl w-full">
                    <div class="flex items-center justify-between mb-4">
                        <h2 class="text-2xl font-bold">Knowledge Sources</h2>
                        <button id="close-sources-modal" class="text-gray-600 hover:text-gray-900">Close</button>
                    </div>
                    <p class="text-sm text-gray-500 mb-3">Files indexed in data/rag_sources used for answers.</p>
                    <div id="sources-list" class="space-y-2 max-h-80 overflow-y-auto"></div>
                </div>
            `;
            document.body.appendChild(modal);
        }
    };
    ensureSourcesUI();

    async function loadSources() {
        if (!sourcesListEl) return;
        sourcesListEl.innerHTML = '<div class="text-gray-500">Loading…</div>';
        try {
            const res = await fetch('/sources');
            const j = await res.json();
            sourcesListEl.innerHTML = '';
            if (!j.sources || j.sources.length === 0) {
                sourcesListEl.innerHTML = '<div class="text-gray-500">No PDF sources found.</div>';
                return;
            }
            j.sources.forEach(src => {
                const row = document.createElement('div');
                row.className = 'flex items-center justify-between p-2 border rounded';
                const name = document.createElement('div');
                name.className = 'text-sm truncate pr-3';
                name.textContent = src.name;
                const link = document.createElement('a');
                link.href = src.url;
                link.target = '_blank';
                link.rel = 'noopener noreferrer';
                link.className = 'text-indigo-600 hover:underline text-sm flex-shrink-0';
                link.textContent = (src.name.split('.').pop() || '').toLowerCase() === 'pdf' ? 'Open' : 'View';
                row.appendChild(name);
                row.appendChild(link);
                sourcesListEl.appendChild(row);
            });
        } catch (e) {
            console.error('sources load error', e);
            sourcesListEl.innerHTML = '<div class="text-red-600">Failed to load sources.</div>';
        }
    }

    // ------------------------------
    // IPO Analyzer handlers (NEW)
    // ------------------------------
    // open modal
    ipoBtn && ipoBtn.addEventListener('click', (e) => {
        if (!ipoModal) return;
        ipoModal.classList.remove('hidden');
        ipoReportDiv && (ipoReportDiv.innerHTML = '');
        ipoContentTextarea && (ipoContentTextarea.value = '');
        ipoFileInput && (ipoFileInput.value = '');
    });

    // close modal
    closeIpoModalBtn && closeIpoModalBtn.addEventListener('click', () => {
        ipoModal && ipoModal.classList.add('hidden');
    });

    // submit form
    ipoForm && ipoForm.addEventListener('submit', async (ev) => {
        ev.preventDefault();
        ipoReportDiv && (ipoReportDiv.innerHTML = '<div class="text-sm text-gray-500">Analyzing IPO — please wait...</div>');
        const file = ipoFileInput && ipoFileInput.files && ipoFileInput.files[0];
        const pasted = ipoContentTextarea && ipoContentTextarea.value && ipoContentTextarea.value.trim();

        try {
            let res;
            if (file) {
                const fd = new FormData();
                fd.append('ipoFile', file);
                res = await fetch('/ipo/analyze', { method: 'POST', body: fd });
            } else if (pasted) {
                // prefer JSON when sending text only
                res = await fetch('/ipo/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content: pasted })
                });
            } else {
                ipoReportDiv.innerHTML = '<div class="text-sm text-red-500">Please provide a PDF or paste IPO content.</div>';
                return;
            }

            if (!res.ok) {
                const txt = await res.text().catch(()=>null);
                throw new Error(txt || `HTTP ${res.status}`);
            }
            const j = await res.json();
            if (j.error) {
                throw new Error(j.error);
            }
            const md = j.ipo_report_md || j.ipo_report || 'No report returned.';
            const html = converter.makeHtml(md || '');
            ipoReportDiv.innerHTML = html;
            // optionally, scroll to results
            ipoReportDiv.scrollIntoView({ behavior: 'smooth' });
        } catch (err) {
            console.error('IPO analyze error', err);
            ipoReportDiv.innerHTML = `<div class="text-sm text-red-600">Analysis failed: ${escapeHtml(err.message)}</div>`;
        }
    });

    // close modal clicking outside (optional)
    ipoModal && ipoModal.addEventListener('click', (e) => {
        if (e.target === ipoModal) ipoModal.classList.add('hidden');
    });

    // ------------------------------
    // Final event wiring
    // ------------------------------
    // basic chat bindings
    sendBtn && sendBtn.addEventListener('click', handleSend);
    userInput && userInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') handleSend(); });

    // portfolio/upload bindings
    portfolioUploadBtn && portfolioUploadBtn.addEventListener('click', () => portfolioFileInput && portfolioFileInput.click());
    portfolioFileInput && portfolioFileInput.addEventListener('change', handlePortfolioUpload);

    // user library upload
    uploadDocBtn && uploadDocBtn.addEventListener('click', () => userFileInput && userFileInput.click());
    userFileInput && userFileInput.addEventListener('change', handleDocUpload);

    // scam/sip/myth buttons
    scamQuizBtn && scamQuizBtn.addEventListener('click', () => { scamModal && scamModal.classList.remove('hidden'); loadScamQuestion(); });
    closeScamModalBtn && closeScamModalBtn.addEventListener('click', () => scamModal && scamModal.classList.add('hidden'));
    nextScamBtn && nextScamBtn.addEventListener('click', loadScamQuestion);
    scamChoiceBtn && scamChoiceBtn.addEventListener('click', () => checkScamAnswer('scam'));
    legitChoiceBtn && legitChoiceBtn.addEventListener('click', () => checkScamAnswer('legit'));

    sipCalculatorBtn && sipCalculatorBtn.addEventListener('click', () => sipModal && sipModal.classList.remove('hidden'));
    closeSipModalBtn && closeSipModalBtn.addEventListener('click', () => sipModal && sipModal.classList.add('hidden'));
    calculateSipBtn && calculateSipBtn.addEventListener('click', calculateSip);

    mythBusterBtn && mythBusterBtn.addEventListener('click', () => { mythModal && mythModal.classList.remove('hidden'); loadMyth(); });
    closeMythModalBtn && closeMythModalBtn.addEventListener('click', () => mythModal && mythModal.classList.add('hidden'));
    nextMythBtn && nextMythBtn.addEventListener('click', loadMyth);
    mythChoiceBtn && mythChoiceBtn.addEventListener('click', () => checkMythAnswer('Myth'));
    factChoiceBtn && factChoiceBtn.addEventListener('click', () => checkMythAnswer('Fact'));

    // sources
    if (sourcesBtn && sourcesModal && closeSourcesModalBtn) {
        sourcesBtn.addEventListener('click', () => {
            sourcesModal.classList.remove('hidden');
            loadSources();
        });
        closeSourcesModalBtn.addEventListener('click', () => sourcesModal.classList.add('hidden'));
    }

    // logout
    logoutBtn && logoutBtn.addEventListener('click', async () => {
        try {
            const res = await fetch('/auth/logout', { method: 'POST' });
            const j = await res.json().catch(()=>null);
            if (j && j.success) {
                window.location.href = '/login';
            } else {
                addMessage('bot', 'Logout failed.');
            }
        } catch (e) {
            console.error('logout', e);
            addMessage('bot', 'Logout failed.');
        }
    });

    // small welcome message and load library
    addMessage('bot', 'Welcome to SEBI Saathi!');
    loadUserLibrary();

    // ensure dashboard button exists (idempotent)
    (function ensureDashboardButton(){
        try {
            const nav = document.querySelector('nav') || document.querySelector('aside');
            if (!nav) return;
            if (!document.getElementById('dashboard-btn')) {
                // if you want programmatic add; your template already has Dashboard link so this is optional
            }
        } catch (e) { /* no-op */ }
    })();
});
