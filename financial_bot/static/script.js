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

    // portfolio history canvas (optional)
    const portfolioLineCanvas = document.getElementById('portfolioLineChart');

    // sources
    const sourcesModal = document.getElementById('sources-modal');
    const sourcesListEl = document.getElementById('sources-list');
    const sourcesBtnTools = document.getElementById('sources-btn-tools');
    const sourcesBtnNav = document.getElementById('sources-btn-nav');
    const viewSourcesLink = document.getElementById('view-sources-link');
    const closeSourcesModalBtn = document.getElementById('close-sources-modal');

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
    // Utility: ensure Chart.js is loaded (returns a Promise)
    // ------------------------------
    function ensureChartLoaded(timeoutMs = 8000) {
        return new Promise((resolve, reject) => {
            if (window.Chart) return resolve(window.Chart);
            // if already injecting, attach to its load/error events
            const existing = document.querySelector('script[data-chartjs-loader="true"]');
            if (existing) {
                existing.addEventListener('load', () => resolve(window.Chart));
                existing.addEventListener('error', () => reject(new Error('Failed to load Chart.js')));
                return;
            }
            const s = document.createElement('script');
            s.src = 'https://cdn.jsdelivr.net/npm/chart.js';
            s.async = true;
            s.setAttribute('data-chartjs-loader', 'true');
            s.onload = () => resolve(window.Chart);
            s.onerror = () => reject(new Error('Failed to load Chart.js'));
            document.head.appendChild(s);
            // timeout guard
            setTimeout(() => {
                if (window.Chart) return resolve(window.Chart);
                reject(new Error('Chart.js load timed out'));
            }, timeoutMs);
        });
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
 // Portfolio upload -> POST /analyze expects form field name "portfolioFile"
const handlePortfolioUpload = async () => {
  const file = portfolioFileInput && portfolioFileInput.files && portfolioFileInput.files[0];
  if (!file) {
    addMessage('bot', 'Please select a portfolio file first.');
    return;
  }

  addMessage('user', `Analyzing portfolio: ${file.name}`);
  if (loadingIndicator) loadingIndicator.classList.remove('hidden');

  const fd = new FormData();
  fd.append('portfolioFile', file); // IMPORTANT: backend expects 'portfolioFile'

  try {
    const res = await fetch('/analyze', {
      method: 'POST',
      body: fd,
      credentials: 'same-origin' // send session cookie
    });

    // If backend redirected to login (302) the browser may have followed it.
    // Check for non-JSON / unauthorized responses
    if (res.status === 401 || (res.redirected && res.url.includes('/login'))) {
      addMessage('bot', 'You must be logged in to upload portfolios. Redirecting to login...');
      setTimeout(() => window.location.href = '/login', 800);
      return;
    }

    const text = await res.text();
    let j = {};
    try { j = text ? JSON.parse(text) : {}; } catch (e) {
      // Some failures return HTML (login page), show useful message
      throw new Error('Server returned unexpected response (not JSON). You may need to login.');
    }

    if (j.error) throw new Error(j.error);

    // open a new window with formatted analysis (server returns analysis_markdown & chart_data)
    const html = createPortfolioPage(j.analysis_markdown || '', j.chart_data || {}, j.holdings || []);
    const win = window.open('', '_blank');
    if (!win) {
      addMessage('bot', 'Could not open analysis window. Please allow popups.');
      return;
    }
    win.document.open();
    win.document.write(html);
    win.document.close();

    addMessage('bot', `<strong>Portfolio analyzed:</strong> ${file.name}`, true);
    // optionally refresh dashboard
    if (typeof loadDashboard === 'function') loadDashboard();
  } catch (err) {
    console.error('Portfolio analyze error', err);
    addMessage('bot', `Portfolio analysis failed: ${err.message || err}`, true);
  } finally {
    if (loadingIndicator) loadingIndicator.classList.add('hidden');
    if (portfolioFileInput) portfolioFileInput.value = '';
  }
};


    function createPortfolioPage(md, chartData = {}, holdings = []) {
      // (same function body as before)
      // We'll reuse the existing function you had for portfolio page creation.
      // To keep script concise here, call the original implementation (you already have it above in your project).
      // But for safety, if not present, fallback to a minimal page.
      try {
        // If the large createPortfolioPage implementation exists in global scope use that
        if (typeof window.__createPortfolioPageImpl === 'function') {
          return window.__createPortfolioPageImpl(md, chartData, holdings);
        }
      } catch (e) {
        console.warn('createPortfolioPage helper not found, using simple fallback', e);
      }
      // Simple fallback:
      const analysisHtml = converter.makeHtml(md || '');
      return `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Portfolio Analysis</title><script src="https://cdn.jsdelivr.net/npm/chart.js"></script></head><body><div>${analysisHtml}</div><script>console.log('fallback page');</script></body></html>`;
    }

    // If you included the full createPortfolioPage implementation in the HTML global,
    // expose it for the function above to call:
    if (window && !window.__createPortfolioPageImpl) {
      // If you have the big implementation embedded inline in index.html already, skip.
      // Otherwise we won't overwrite anything here.
    }

    // ------------------------------
    // Portfolio history (6 months) & chart rendering
    // ------------------------------
    window._portfolioChart = window._portfolioChart || null;
    async function loadPortfolioHistory() {
      if (!portfolioLineCanvas) return;
      try {
        const res = await fetch('/portfolio/history', { credentials: 'same-origin' });
        if (!res.ok) {
          // silently ignore if endpoint not available
          return;
        }
        const j = await res.json();
        if (!j.history || !Array.isArray(j.history) || j.history.length === 0) return;

        // ensure Chart.js
        try {
          await ensureChartLoaded();
        } catch (e) {
          console.warn('Chart.js failed to load for portfolio history', e);
          return;
        }

        const labels = j.history.map(h => h.date);
        const data = j.history.map(h => h.value);

        // destroy existing
        if (window._portfolioChart) {
          try { window._portfolioChart.destroy(); } catch(e){/*no-op*/ }
          window._portfolioChart = null;
        }

        const ctx = portfolioLineCanvas.getContext('2d');
        window._portfolioChart = new Chart(ctx, {
          type: 'line',
          data: {
            labels,
            datasets: [{
              label: 'Portfolio Value',
              data,
              tension: 0.25,
              fill: true,
              backgroundColor: 'rgba(59,130,246,0.08)',
              borderColor: '#3b82f6',
              pointRadius: 2,
              borderWidth: 2
            }]
          },
          options: {
            plugins: { legend: { display: false } },
            scales: {
              x: { display: true },
              y: { display: true, beginAtZero: false }
            },
            maintainAspectRatio: false,
            responsive: true
          }
        });
      } catch (err) {
        console.error('Failed to load portfolio history', err);
      }
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
            const actions = document.createElement('div');
            actions.className = 'flex gap-2 items-center';
            const openLink = document.createElement('a');
            openLink.href = '/source_view/' + encodeURIComponent(f);
            openLink.target = '_blank';
            openLink.rel = 'noopener noreferrer';
            openLink.className = 'text-indigo-600 text-xs hover:underline';
            openLink.textContent = 'Open';
            const del = document.createElement('button');
            del.className = 'text-red-400 text-xs';
            del.textContent = 'Delete';
            del.onclick = () => handleDeleteFile(f);
            actions.appendChild(openLink);
            actions.appendChild(del);
            li.appendChild(name);
            li.appendChild(actions);
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
// User library upload -> POST /upload_and_ingest expects form field name "userFile"
const handleDocUpload = async () => {
  const f = userFileInput && userFileInput.files && userFileInput.files[0];
  if (!f) return;

  addMessage('user', `Uploading ${f.name}`);
  if (loadingIndicator) loadingIndicator.classList.remove('hidden');

  const fd = new FormData();
  fd.append('userFile', f); // IMPORTANT: backend expects 'userFile'

  try {
    const res = await fetch('/upload_and_ingest', {
      method: 'POST',
      body: fd,
      credentials: 'same-origin' // ensure session cookie is sent
    });

    // handle redirect to login or unauthorized
    if (res.status === 401 || (res.redirected && res.url.includes('/login'))) {
      addMessage('bot', 'You must be logged in to upload documents. Redirecting to login...');
      setTimeout(() => window.location.href = '/login', 800);
      return;
    }

    const text = await res.text();
    let j = {};
    try { j = text ? JSON.parse(text) : {}; } catch (e) {
      throw new Error('Server returned unexpected response (not JSON). You may need to login.');
    }

    if (!res.ok) {
      throw new Error(j.error || `HTTP ${res.status}`);
    }

    if (j.success) {
      addMessage('bot', `<strong>${f.name}</strong> added to library.`, true);
      updateUserLibrary(j.user_library || []);
      // maybe refresh vectorstore or other UI bits
    } else {
      throw new Error(j.error || 'Upload failed');
    }
  } catch (e) {
    console.error('upload error', e);
    addMessage('bot', `Upload failed: ${e.message || e}`, true);
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

    // ------------------------------
    // calculateSip (improved + Chart guard)
    // ------------------------------
    async function calculateSip() {
        const goal = sipGoalInput?.value, amount = sipAmountInput?.value, years = sipYearsInput?.value;
        if (!goal || !amount || !years) { sipResultEl.textContent = 'Please fill fields.'; return; }

        // simple validation
        const amountNum = Number(amount);
        const yearsNum = Number(years);
        if (!isFinite(amountNum) || amountNum <= 0) { sipResultEl.textContent = 'Please enter a valid amount.'; return; }
        if (!isFinite(yearsNum) || yearsNum <= 0) { sipResultEl.textContent = 'Please enter valid years.'; return; }

        if (loadingIndicator) loadingIndicator.classList.remove('hidden');
        try {
            const res = await fetch('/calculate_sip', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ amount: amountNum, years: yearsNum })
            });

            // tolerate non-json responses
            const text = await res.text();
            let j;
            try { j = text ? JSON.parse(text) : {}; } catch (e) { throw new Error('Server returned unexpected response'); }

            if (j.error) {
                sipResultEl.textContent = j.error;
                return;
            }
            const monthly = Number(j.monthly_sip);
            if (!isFinite(monthly)) throw new Error('Invalid SIP returned');

            sipResultEl.innerHTML = `Monthly SIP: <strong>₹${monthly.toLocaleString('en-IN')}</strong>`;

            // if growth_data present — show chart
            if (Array.isArray(j.growth_data) && j.growth_data.length) {
                try {
                    await ensureChartLoaded();
                } catch (chartErr) {
                    console.warn('Chart.js not available:', chartErr);
                    sipResultEl.innerHTML += '<div class="text-sm text-gray-500 mt-2">Chart unavailable (failed to load Chart.js).</div>';
                    return;
                }
                const ctx = document.getElementById('sipChart')?.getContext?.('2d');
                if (ctx) {
                    if (window._sipChart) window._sipChart.destroy();
                    window._sipChart = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: j.growth_data.map(d => `Y${d.year}`),
                            datasets: [
                                { label: 'Invested', data: j.growth_data.map(d => Number(d.invested) || 0) },
                                { label: 'Value', data: j.growth_data.map(d => Number(d.value) || 0) }
                            ]
                        },
                        options: { scales: { y: { beginAtZero: true } }, responsive: true }
                    });
                }
            } else {
                sipResultEl.innerHTML += '<div class="text-sm text-gray-500 mt-2">No growth_data to chart.</div>';
            }
        } catch (e) {
            console.error('sip error', e);
            sipResultEl.textContent = `Calculation failed: ${e.message || e}`;
        } finally {
            if (loadingIndicator) loadingIndicator.classList.add('hidden');
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
    // Sources UI logic (viewer links)
    // ------------------------------
    async function loadSources() {
        if (!sourcesListEl) return;
        sourcesListEl.innerHTML = '<div class="text-gray-500">Loading…</div>';
        try {
            const res = await fetch('/sources');
            const text = await res.text();
            let j;
            try { j = text ? JSON.parse(text) : {}; } catch (e) { throw new Error('Unexpected /sources response'); }

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
                const controls = document.createElement('div');
                controls.className = 'flex items-center gap-2';
                const view = document.createElement('a');
                view.href = '/source_view/' + encodeURIComponent(src.name);
                view.target = '_blank';
                view.rel = 'noopener noreferrer';
                view.className = 'text-indigo-600 hover:underline text-sm';
                view.textContent = 'Open';
                const raw = document.createElement('a');
                raw.href = src.url;
                raw.target = '_blank';
                raw.rel = 'noopener noreferrer';
                raw.className = 'text-muted text-xs';
                raw.textContent = 'Raw';
                controls.appendChild(view);
                controls.appendChild(raw);
                row.appendChild(name);
                row.appendChild(controls);
                sourcesListEl.appendChild(row);
            });
        } catch (e) {
            console.error('sources load error', e);
            if (sourcesListEl) sourcesListEl.innerHTML = '<div class="text-red-600">Failed to load sources.</div>';
        }
    }

    // open sources modal and load
    function openSourcesModal() {
        if (!sourcesModal) return;
        sourcesModal.classList.remove('hidden');
        loadSources();
    }

    if (sourcesBtnTools) sourcesBtnTools.addEventListener('click', openSourcesModal);
    if (sourcesBtnNav) sourcesBtnNav.addEventListener('click', openSourcesModal);
    if (viewSourcesLink) viewSourcesLink.addEventListener('click', (e) => { e.preventDefault(); openSourcesModal(); });
    if (closeSourcesModalBtn) closeSourcesModalBtn.addEventListener('click', () => sourcesModal.classList.add('hidden'));
    if (sourcesModal) sourcesModal.addEventListener('click', (ev) => { if (ev.target === sourcesModal) sourcesModal.classList.add('hidden'); });

    // ------------------------------
    // IPO Analyzer handlers (NEW)
    // ------------------------------
    // open modal
    // ------------------------------
// IPO Analyzer handlers (robust, sends credentials and refreshes list)
// ------------------------------
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
      // include credentials so Flask login session is sent
      res = await fetch('/ipo/analyze', { method: 'POST', body: fd, credentials: 'same-origin' });
    } else if (pasted) {
      res = await fetch('/ipo/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ content: pasted })
      });
    } else {
      ipoReportDiv.innerHTML = '<div class="text-sm text-red-500">Please provide a PDF or paste IPO content.</div>';
      return;
    }

    // handle redirect to login (Flask returns 302 -> browser may follow it; detect by non-json or redirected)
    if (res.status === 401 || res.redirected || res.url?.includes('/login')) {
      ipoReportDiv.innerHTML = '<div class="text-sm text-red-600">You must be logged in to analyze & save IPOs. Redirecting to login...</div>';
      // optionally redirect
      setTimeout(()=> { window.location.href = '/login'; }, 900);
      return;
    }

    const text = await res.text();
    let j;
    try { j = text ? JSON.parse(text) : {}; } catch (e) {
      // server returned HTML (likely login page) or malformed JSON
      console.error('Unexpected IPO analyze response (not JSON):', text);
      ipoReportDiv.innerHTML = `<div class="text-sm text-red-600">Analysis failed: Unexpected server response.</div>`;
      return;
    }

    if (j.error) {
      ipoReportDiv.innerHTML = `<div class="text-sm text-red-600">Analysis failed: ${escapeHtml(j.error)}</div>`;
      return;
    }

    const md = j.ipo_report_md || j.ipo_report || 'No report returned.';
    const html = converter.makeHtml(md || '');
    ipoReportDiv.innerHTML = html;

    // show success message if saved (backend returns report_id when saved)
    if (j.report_id) {
      // show user-friendly confirmation and refresh saved reports area
      const confirmEl = document.createElement('div');
      confirmEl.className = 'mt-4 text-sm text-green-600';
      confirmEl.innerHTML = `Report saved (ID: <strong>${j.report_id}</strong>). SEBI score: <strong>${j.sebi_score ?? 'N/A'}</strong>.`;
      ipoReportDiv.prepend(confirmEl);

      // refresh list of saved IPO reports in dashboard
      if (typeof loadIpoReports === 'function') {
        loadIpoReports();
      }
    } else {
      // if backend did not return report_id, still reload as a precaution (or show note)
      ipoReportDiv.innerHTML += '<div class="text-sm text-yellow-600 mt-3">Report processed but not saved (no id returned).</div>';
    }

    // scroll results into view
    ipoReportDiv.scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    console.error('IPO analyze error', err);
    ipoReportDiv.innerHTML = `<div class="text-sm text-red-600">Analysis failed: ${escapeHtml(err.message || err)}</div>`;
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

    // load portfolio history (if canvas exists)
    loadPortfolioHistory();

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
