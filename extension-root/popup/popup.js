// popup/popup.js
// Queries the background service worker for the latest inference result
// for the currently active tab, then renders the traffic light state.

(async () => {

    // ── DOM refs ──────────────────────────────────────────────────────────────
    const lightRed    = document.getElementById('light-red');
    const lightOrange = document.getElementById('light-orange');
    const lightGreen  = document.getElementById('light-green');
    const verdictBox  = document.getElementById('verdict-box');
    const verdictIcon = document.getElementById('verdict-icon');
    const verdictTitle = document.getElementById('verdict-title');
    const verdictSub   = document.getElementById('verdict-sub');
    const confSection  = document.getElementById('confidence-section');
    const confValue    = document.getElementById('confidence-value');
    const confFill     = document.getElementById('confidence-fill');
    const statusDot    = document.getElementById('status-dot');
    const footerText   = document.getElementById('footer-text');

    // ── Helpers ───────────────────────────────────────────────────────────────

    function setIdle() {
        [lightRed, lightOrange, lightGreen].forEach(l => l.classList.remove('active'));
        verdictBox.className = 'verdict-box';
        verdictIcon.textContent = '🔍';
        verdictTitle.textContent = 'Waiting for video...';
        verdictSub.textContent = 'Open a page with a playing video';
        confSection.style.display = 'none';
        statusDot.classList.remove('active');
        footerText.textContent = 'No active analysis';
    }

    function setNoVideo() {
        [lightRed, lightOrange, lightGreen].forEach(l => l.classList.remove('active'));
        verdictBox.className = 'verdict-box';
        verdictIcon.textContent = '📺';
        verdictTitle.textContent = 'No video detected';
        verdictSub.textContent = 'Play a video on this page to start';
        confSection.style.display = 'none';
        statusDot.classList.remove('active');
        footerText.textContent = 'Extension active — no video';
    }

    function applyResult(confidence) {
        const score = Number(confidence);
        if (!Number.isFinite(score)) { setIdle(); return; }

        statusDot.classList.add('active');
        confSection.style.display = 'flex';
        confValue.textContent = score.toFixed(1) + '%';
        confFill.style.width = Math.min(score, 100) + '%';

        [lightRed, lightOrange, lightGreen].forEach(l => l.classList.remove('active'));

        if (score < 40) {
            lightRed.classList.add('active');
            confFill.className = 'confidence-fill low';
            verdictBox.className = 'verdict-box ai';
            verdictIcon.textContent = '⚠️';
            verdictTitle.textContent = 'Likely AI-Generated';
            verdictSub.textContent = `Low authenticity score (${score.toFixed(1)}%)`;
            footerText.textContent = 'AI-generated content suspected';

        } else if (score <= 75) {
            lightOrange.classList.add('active');
            confFill.className = 'confidence-fill mid';
            verdictBox.className = 'verdict-box mixed';
            verdictIcon.textContent = '🟡';
            verdictTitle.textContent = 'Uncertain — Mixed Signals';
            verdictSub.textContent = `Moderate authenticity score (${score.toFixed(1)}%)`;
            footerText.textContent = 'Content authenticity unclear';

        } else {
            lightGreen.classList.add('active');
            confFill.className = 'confidence-fill high';
            verdictBox.className = 'verdict-box real';
            verdictIcon.textContent = '✅';
            verdictTitle.textContent = 'Likely Real Content';
            verdictSub.textContent = `High authenticity score (${score.toFixed(1)}%)`;
            footerText.textContent = 'Content appears authentic';
        }
    }

    // ── Main: query background for active tab's latest result ─────────────────

    setIdle();

    let activeTabId;
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        activeTabId = tab?.id;
    } catch (e) {
        footerText.textContent = 'Could not read active tab';
        return;
    }

    if (typeof activeTabId !== 'number') {
        setIdle();
        return;
    }

    // Ask the background for this tab's cached result.
    chrome.runtime.sendMessage(
        { type: 'GET_STATUS', tabId: activeTabId },
        (response) => {
            if (chrome.runtime.lastError) {
                // Background not ready yet.
                setIdle();
                return;
            }

            if (!response || response.confidence === null || response.confidence === undefined) {
                setNoVideo();
                return;
            }

            applyResult(response.confidence);
        }
    );

    // Also listen for live updates while popup is open.
    chrome.runtime.onMessage.addListener((message) => {
        if (message?.type === 'UPDATE_UI' && message.targetTabId === activeTabId) {
            applyResult(message.confidence);
        }
    });

})();
