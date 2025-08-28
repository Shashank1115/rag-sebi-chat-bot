document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const chatContainer = document.getElementById('chat-container');
    const loadingIndicator = document.getElementById('loading');
    const uploadBtn = document.getElementById('upload-btn');
    const fileInput = document.getElementById('file-input');
    // The uploadForm is no longer needed for submission
    const converter = new showdown.Converter();

    const addMessage = (sender, message) => {
        const messageWrapper = document.createElement('div');
        messageWrapper.classList.add('flex', sender === 'user' ? 'user-msg' : 'bot-msg');
        
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('p-3', 'rounded-lg', 'max-w-md');
        
        if (sender === 'user') {
            messageDiv.classList.add('bg-blue-500', 'text-white');
            messageDiv.innerHTML = `<p>${message}</p>`;
        } else {
            messageDiv.classList.add('bg-indigo-500', 'text-white');
            messageDiv.innerHTML = converter.makeHtml(message);
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
        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question }),
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            addMessage('bot', data.answer);
        } catch (error) {
            console.error('Error:', error);
            addMessage('bot', 'Sorry, something went wrong. Please check the server logs.');
        } finally {
            loadingIndicator.classList.add('hidden');
        }
    };
    
    // --- THIS IS THE DEFINITIVE FIX for file uploads ---
    const handleFileUpload = async () => {
        const file = fileInput.files[0];
        if (!file) {
            addMessage('bot', 'Please select a file first.');
            return;
        }

        addMessage('user', `Analyzing file: ${file.name}`);
        loadingIndicator.classList.remove('hidden');

        const formData = new FormData();
        formData.append('portfolioFile', file);

        try {
            // Use fetch to send the file to the backend
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Open a new window/tab for the analysis
            const analysisWindow = window.open('', '_blank');
            if (analysisWindow) {
                // Dynamically write the entire HTML for the new page
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
            fileInput.value = ''; // Reset file input
        }
    };

    // Helper function to generate the HTML for the portfolio page
    const createPortfolioPage = (analysisMarkdown, chartData) => {
        const analysisHtml = converter.makeHtml(analysisMarkdown);
        const chartDataJson = JSON.stringify(chartData);

        return `
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Portfolio Analysis - SEBI Saathi</title>
                <script src="https://cdn.tailwindcss.com"></script>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body { font-family: 'Inter', sans-serif; }
                    .analysis-section { background-color: #f9fafb; border: 1px solid #e5e7eb; padding: 1.5rem; border-radius: 0.75rem; margin-bottom: 1.5rem; }
                    .analysis-section ul { list-style-type: disc; padding-left: 20px; }
                    .analysis-section li { margin-bottom: 8px; }
                </style>
                <link rel="preconnect" href="https://fonts.googleapis.com">
                <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
            </head>
            <body class="bg-gray-100">
                <div class="container mx-auto p-4 sm:p-6 lg:p-8">
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

    // Event listeners
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileUpload);
    sendBtn.addEventListener('click', handleSend);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSend();
    });
});
