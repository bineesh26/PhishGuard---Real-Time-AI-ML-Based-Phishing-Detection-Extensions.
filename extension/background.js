chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'scan_url') {
        fetch('http://127.0.0.1:8000/check', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url: request.url })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => sendResponse({ success: true, data: data }))
        .catch(error => {
            console.error('Error scanning URL:', error);
            sendResponse({ success: false, error: error.message });
        });

        // Return true to indicate we will send a response asynchronously
        return true; 
    }
});
