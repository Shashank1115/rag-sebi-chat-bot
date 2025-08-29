document.addEventListener('DOMContentLoaded', () => {
    // Main Chat Elements
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const chatContainer = document.getElementById('chat-container');
    const loadingIndicator = document.getElementById('loading');
    const uploadBtn = document.getElementById('upload-btn');
    const fileInput = document.getElementById('file-input');
    const uploadForm = document.getElementById('upload-form');
    const converter = new showdown.Converter();

    // Engagement Feature Buttons
    const scamQuizBtn = document.getElementById('scam-quiz-btn');
    const sipCalculatorBtn = document.getElementById('sip-calculator-btn');
    const mythBusterBtn = document.getElementById('myth-buster-btn');

    // Scam Modal Elements
    const scamModal = document.getElementById('scam-modal');
    const closeScamModalBtn = document.getElementById('close-scam-modal');
    const scamQuestionEl = document.getElementById('scam-question');
    const scamChoiceBtn = document.getElementById('scam-choice-btn');
    const legitChoiceBtn = document.getElementById('legit-choice-btn');
    const scamFeedbackEl = document.getElementById('scam-feedback');
    const nextScamBtn = document.getElementById('next-scam-btn');
    let currentScamQuestion = null;

    // SIP Modal Elements
    const sipModal = document.getElementById('sip-modal');
    const closeSipModalBtn = document.getElementById('close-sip-modal');
    const calculateSipBtn = document.getElementById('calculate-sip-btn');
    const sipResultEl = document.getElementById('sip-result');
    const sipGoalInput = document.getElementById('sip-goal');
    const sipAmountInput = document.getElementById('sip-amount');
    const sipYearsInput = document.getElementById('sip-years');
    let sipChart = null;

    // Myth Modal Elements
    const mythModal = document.getElementById('myth-modal');
    const closeMythModalBtn = document.getElementById('close-myth-modal');
    const mythStatementEl = document.getElementById('myth-statement');
    const mythChoiceBtn = document.getElementById('myth-choice-btn');
    const factChoiceBtn = document.getElementById('fact-choice-btn');
    const mythFeedbackEl = document.getElementById('myth-feedback');
    const nextMythBtn = document.getElementById('next-myth-btn');
    let currentMyth = null;

    // --- Core Chat Functions ---
    const addMessage = (sender, message) => { /* ... same as before ... */ };
    const handleSend = async () => { /* ... same as before ... */ };
    const handleFileUpload = () => { /* ... same as before ... */ };

    // --- Scam Simulator Logic ---
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
        scamFeedbackEl.innerHTML = `
            <p class="font-bold ${isCorrect ? 'text-green-600' : 'text-red-600'}">
                ${isCorrect ? 'Correct!' : 'Incorrect.'} The message was ${currentScamQuestion.type}.
            </p>
            <p class="mt-2">${currentScamQuestion.explanation}</p>
        `;
    };

    scamQuizBtn.addEventListener('click', () => {
        scamModal.classList.add('active');
        loadScamQuestion();
    });
    closeScamModalBtn.addEventListener('click', () => scamModal.classList.remove('active'));
    nextScamBtn.addEventListener('click', loadScamQuestion);
    scamChoiceBtn.addEventListener('click', () => checkScamAnswer('scam'));
    legitChoiceBtn.addEventListener('click', () => checkScamAnswer('legit'));

    // --- SIP Planner Logic ---
    const calculateSip = async () => {
        const goal = sipGoalInput.value;
        const amount = sipAmountInput.value;
        const years = sipYearsInput.value;
        if (!goal || !amount || !years) {
            sipResultEl.textContent = 'Please fill all fields.';
            return;
        }
        const response = await fetch('/calculate_sip', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ amount, years }),
        });
        const data = await response.json();
        if (data.error) {
            sipResultEl.textContent = data.error;
            return;
        }
        sipResultEl.innerHTML = `To reach your goal of <span class="font-bold">₹${parseInt(amount).toLocaleString('en-IN')}</span> for your <span class="font-bold">${goal}</span>, you need to invest <span class="font-bold">₹${data.monthly_sip.toLocaleString('en-IN')}</span> per month.`;
        
        if (sipChart) sipChart.destroy();
        const ctx = document.getElementById('sipChart').getContext('2d');
        sipChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.growth_data.map(d => `Year ${d.year}`),
                datasets: [
                    {
                        label: 'Total Invested',
                        data: data.growth_data.map(d => d.invested),
                        backgroundColor: 'rgba(59, 130, 246, 0.5)',
                    },
                    {
                        label: 'Investment Value',
                        data: data.growth_data.map(d => d.value),
                        backgroundColor: 'rgba(22, 163, 74, 0.5)',
                    }
                ]
            },
            options: { scales: { y: { beginAtZero: true } } }
        });
    };
    
    sipCalculatorBtn.addEventListener('click', () => sipModal.classList.add('active'));
    closeSipModalBtn.addEventListener('click', () => sipModal.classList.remove('active'));
    calculateSipBtn.addEventListener('click', calculateSip);

    // --- Myth Buster Logic ---
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
        mythFeedbackEl.innerHTML = `
            <p class="font-bold">${isCorrect ? 'Correct!' : 'Not quite!'} This is a ${currentMyth.type}.</p>
            <p class="mt-2">${currentMyth.explanation}</p>
        `;
        mythFeedbackEl.classList.add(isCorrect ? 'bg-green-100' : 'bg-red-100');
    };

    mythBusterBtn.addEventListener('click', () => {
        mythModal.classList.add('active');
        loadMyth();
    });
    closeMythModalBtn.addEventListener('click', () => mythModal.classList.remove('active'));
    nextMythBtn.addEventListener('click', loadMyth);
    mythChoiceBtn.addEventListener('click', () => checkMythAnswer('Myth'));
    factChoiceBtn.addEventListener('click', () => checkMythAnswer('Fact'));

    // Re-add main chat event listeners
    sendBtn.addEventListener('click', handleSend);
    userInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') handleSend(); });
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => {
        if (fileInput.files[0]) {
            addMessage('user', `Analyzing file: ${fileInput.files[0].name}`);
            uploadForm.submit();
        }
    });

});
