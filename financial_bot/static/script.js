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
    // const portfolioUploadForm = document.getElementById('upload-form'); // No longer needed for submission

    // User Library Elements
    const uploadDocBtn = document.getElementById('upload-doc-btn');
    const userFileInput = document.getElementById('user-file-input');
    const userLibraryList = document.getElementById('user-library-list');
    const searchScopeToggle = document.getElementById('search-scope-toggle');

    // Engagement Feature Buttons
    const scamQuizBtn = document.getElementById('scam-quiz-btn');
    const sipCalculatorBtn = document.getElementById('sip-calculator-btn');
    const mythBusterBtn = document.getElementById('myth-buster-btn');

    // (All other modal element references are assumed to be here)
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


    const addMessage = (sender, message, isHtml = false) => {
        const messageWrapper = document.createElement('div');
        messageWrapper.classList.add('flex', sender === 'user' ? 'user-msg' : 'bot-msg');
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('p-3', 'rounded-lg', 'max-w-md');
        if (sender === 'user') {
            messageDiv.classList.add('bg-blue-500', 'text-white');
            messageDiv.innerHTML = `<p>${message}</p>`;
        } else {
            messageDiv.classList.add('bg-indigo-500', 'text-white');
            messageDiv.innerHTML = isHtml ? message : converter.makeHtml(message);
        }
        messageWrapper.appendChild(messageDiv);
        chatContainer.appendChild(messageWrapper);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    };

    const handleSend = async () => {
        const question = userInput.value.trim();
        if (!question) return;
        addMessage('user', question);
        userInput.value = '';
        loadingIndicator.classList.remove('hidden');
        const scope = searchScopeToggle.checked ? 'user_only' : 'all';
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
            addMessage('bot', data.answer || data.error);
        } catch (error) {
            addMessage('bot', `Sorry, an error occurred: ${error.message}`);
        } finally {
            loadingIndicator.classList.add('hidden');
        }
    };
    
    // --- THIS IS THE DEFINITIVE FIX for Portfolio Analysis ---
    const handlePortfolioUpload = async () => {
        const file = portfolioFileInput.files[0];
        if (!file) {
            addMessage('bot', 'Please select a portfolio file first.');
            return;
        }

        addMessage('user', `Analyzing portfolio: ${file.name}`);
        loadingIndicator.classList.remove('hidden');

        const formData = new FormData();
        formData.append('portfolioFile', file);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
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
            loadingIndicator.classList.add('hidden');
            portfolioFileInput.value = '';
        }
    };

    // Helper function to dynamically generate the HTML for the new portfolio page
    const createPortfolioPage = (analysisMarkdown, chartData) => {
        const analysisHtml = converter.makeHtml(analysisMarkdown);
        const chartDataJson = JSON.stringify(chartData);

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
                    .analysis-section ul { list-style-type: disc; padding-left: 20px; }
                    .analysis-section li { margin-bottom: 8px; }
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

    // (All other functions like updateUserLibrary, handleDocUpload, engagement features, etc. are assumed to be present and correct)
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
    const updateUserLibrary = (files) => {
        userLibraryList.innerHTML = '';
        if (files && files.length > 0) {
            const list = document.createElement('ul');
            list.className = 'space-y-2';
            files.forEach(file => {
                const listItem = document.createElement('li');
                listItem.className = 'p-2 bg-gray-700 rounded text-sm flex items-center';
                const fileNameSpan = document.createElement('span');
                fileNameSpan.className = 'truncate';
                fileNameSpan.textContent = file;
                const deleteBtn = document.createElement('span');
                deleteBtn.className = 'delete-btn';
                deleteBtn.innerHTML = '&times;';
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
        const file = userFileInput.files[0];
        if (!file) return;
        addMessage('bot', `Processing <strong>${file.name}</strong>...`, true);
        loadingIndicator.classList.remove('hidden');
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
            loadingIndicator.classList.add('hidden');
            userFileInput.value = '';
        }
    };
    const loadScamQuestion = async () => {
        scamFeedbackEl.innerHTML = '';
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
        const goal = sipGoalInput.value, amount = sipAmountInput.value, years = sipYearsInput.value;
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
        const ctx = document.getElementById('sipChart').getContext('2d');
        sipChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.growth_data.map(d => `Year ${d.year}`),
                datasets: [{ label: 'Invested', data: data.growth_data.map(d => d.invested), backgroundColor: 'rgba(59, 130, 246, 0.5)' }, { label: 'Value', data: data.growth_data.map(d => d.value), backgroundColor: 'rgba(22, 163, 74, 0.5)' }]
            },
            options: { scales: { y: { beginAtZero: true } } }
        });
    };
    const loadMyth = async () => {
        mythFeedbackEl.innerHTML = '';
        mythFeedbackEl.className = 'text-center p-4 rounded';
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
    sendBtn.addEventListener('click', handleSend);
    userInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') handleSend(); });
    uploadDocBtn.addEventListener('click', () => userFileInput.click());
    userFileInput.addEventListener('change', handleDocUpload);
    portfolioUploadBtn.addEventListener('click', () => portfolioFileInput.click());
    portfolioFileInput.addEventListener('change', handlePortfolioUpload);
    scamQuizBtn.addEventListener('click', () => { scamModal.classList.add('active'); loadScamQuestion(); });
    closeScamModalBtn.addEventListener('click', () => scamModal.classList.remove('active'));
    nextScamBtn.addEventListener('click', loadScamQuestion);
    scamChoiceBtn.addEventListener('click', () => checkScamAnswer('scam'));
    legitChoiceBtn.addEventListener('click', () => checkScamAnswer('legit'));
    sipCalculatorBtn.addEventListener('click', () => sipModal.classList.add('active'));
    closeSipModalBtn.addEventListener('click', () => sipModal.classList.remove('active'));
    calculateSipBtn.addEventListener('click', calculateSip);
    mythBusterBtn.addEventListener('click', () => { mythModal.classList.add('active'); loadMyth(); });
    closeMythModalBtn.addEventListener('click', () => mythModal.classList.remove('active'));
    nextMythBtn.addEventListener('click', loadMyth);
    mythChoiceBtn.addEventListener('click', () => checkMythAnswer('Myth'));
    factChoiceBtn.addEventListener('click', () => checkMythAnswer('Fact'));
});
