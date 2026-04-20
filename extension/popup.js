chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const url = tabs[0].url;

    fetch("http://127.0.0.1:8000/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url })
    })
        .then(res => res.json())
        .then(data => {
            const circle = document.getElementById("statusCircle");
            const text = document.getElementById("statusText");
            const msg = document.getElementById("statusMsg");

            if (data.detail) {
                circle.className = "status-circle";
                text.innerText = "Error";
                msg.innerText = typeof data.detail === 'string' ? data.detail : "Backend error";
                return;
            }

            if (data.prediction === "phishing") {
                circle.className = "status-circle phishing";
                text.innerText = "⚠ Phishing";
                const confidence = (data.probability * 100).toFixed(1);
                msg.innerText = `This website is dangerous. Avoid entering any data! (Confidence: ${confidence}%)`;
            }
            else {
                circle.className = "status-circle safe";
                text.innerText = "✓ Safe";
                const confidence = ((1 - data.probability) * 100).toFixed(1);
                msg.innerText = `This website is safe to use. (Confidence: ${confidence}%)`;
            }
        })
        .catch(() => {
            document.getElementById("statusText").innerText = "Offline";
            document.getElementById("statusMsg").innerText = "Backend not running";
        });
});
