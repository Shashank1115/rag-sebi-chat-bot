document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const chatContainer = document.getElementById('chat-container');
    const loadingIndicator = document.getElementById('loading');
    const uploadBtn = document.getElementById('upload-btn');
    const fileInput = document.getElementById('file-input');
    const uploadForm = document.getElementById('upload-form'); // Get the form
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
        messageWrapper.appendChild(messageWrapper);
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
    
    // --- THIS IS THE FIX for file uploads ---
    uploadBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (file) {
            addMessage('user', `Analyzing file: ${file.name}`);
            // Submit the form to the /analyze endpoint, which will open in a new tab
            uploadForm.submit();
        }
    });

    sendBtn.addEventListener('click', handleSend);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSend();
    });
});
