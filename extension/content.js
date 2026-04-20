let currentButton = null;
let currentTooltip = null;

function removeExistingElements() {
    if (currentButton) {
        currentButton.remove();
        currentButton = null;
    }
    if (currentTooltip) {
        currentTooltip.remove();
        currentTooltip = null;
    }
}

document.addEventListener('mouseup', (e) => {
    // If we clicked inside the tooltip or button, do nothing
    if (e.target.closest('.phishguard-floating-btn') || e.target.closest('.phishguard-tooltip')) {
        return;
    }

    // Always clean up previous ones on a new click outside
    let prevTooltipExists = !!currentTooltip;
    removeExistingElements();

    let selection = window.getSelection().toString().trim();
    
    // Support grabbing the actual URL if the user highlighted text inside a hyperlink
    const anchor = e.target.closest('a');
    if (anchor && anchor.href) {
        selection = anchor.href;
    } else if (selection.length >= 4 && selection.includes('.') && !selection.startsWith('http')) {
        // Fallback: If it's a raw domain without protocol, add http:// so the model predicts properly
        selection = 'http://' + selection;
    }
    
    // Only show if there's a valid selection (at least 4 chars long, looks something like a domain or URL)
    // We'll be lenient so users can scan anything suspicious 
    if (selection.length >= 4 && selection.includes('.')) {
        // Delay slightly to prevent flashing
        setTimeout(() => {
            createFloatingButton(selection, e.pageX, e.pageY);
        }, 10);
    }
});

function createFloatingButton(selectedText, x, y) {
    currentButton = document.createElement('button');
    currentButton.className = 'phishguard-floating-btn';
    currentButton.innerText = 'Scan for Phishing';
    
    // Position slightly below and to the right of the cursor
    currentButton.style.left = `${x + 10}px`;
    currentButton.style.top = `${y + 15}px`;

    currentButton.addEventListener('click', async (e) => {
        e.stopPropagation();
        
        // Set loading state
        currentButton.classList.add('loading');
        currentButton.innerText = 'Scanning...';

        try {
            chrome.runtime.sendMessage({ action: 'scan_url', url: selectedText }, (response) => {
                if (chrome.runtime.lastError) {
                    console.error('Runtime error:', chrome.runtime.lastError);
                    showTooltip('Error', 'Could not connect to PhishGuard backend.', 'phishing', x, y);
                    currentButton.remove();
                    return;
                }
                
                if (response && response.success) {
                    const data = response.data;
                    const prediction = data.prediction; // 'phishing' or 'legitimate'
                    const prob = (data.probability * 100).toFixed(1);
                    
                    const title = prediction === 'phishing' ? '🚨 Phishing Detected!' : '✅ Safe URL';
                    const detail = prediction === 'phishing' ? 
                                    `Probability: ${prob}% malicious` : 
                                    `Probability: ${(100 - prob).toFixed(1)}% safe`;
                    
                    showTooltip(title, detail, prediction, x, y);
                } else {
                    showTooltip('Error', 'Failed to scan URL. Backend offline?', 'phishing', x, y);
                }
                currentButton.remove();
                currentButton = null;
            });
        } catch (err) {
            showTooltip('Error', 'Extension error occurred.', 'phishing', x, y);
            currentButton.remove();
        }
    });

    document.body.appendChild(currentButton);
}

function showTooltip(title, detail, typeClass, x, y) {
    removeExistingElements();

    currentTooltip = document.createElement('div');
    currentTooltip.className = `phishguard-tooltip ${typeClass}`;
    currentTooltip.style.left = `${x + 10}px`;
    currentTooltip.style.top = `${y + 15}px`;

    currentTooltip.innerHTML = `
        <button class="phishguard-close-btn">&times;</button>
        <div class="title">${title}</div>
        <div class="details">${detail}</div>
    `;

    currentTooltip.querySelector('.phishguard-close-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        removeExistingElements();
    });

    document.body.appendChild(currentTooltip);
}
