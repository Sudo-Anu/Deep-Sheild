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

    let activeTab;
    try {
        [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
    } catch (e) {
        footerText.textContent = 'Could not read active tab';
        return;
    }

    const activeTabId = activeTab?.id;
    if (typeof activeTabId !== 'number') {
        setIdle();
        return;
    }

    /**
     * Queries the content script for live video status.
     * If the content script isn't injected yet, injects it first then retries once.
     * @param {boolean} [isRetry=false]
     */
    function queryVideoStatus(isRetry = false) {
        chrome.tabs.sendMessage(activeTabId, { type: 'GET_VIDEO_STATUS' }, (csResponse) => {
            // Content script not running — inject it programmatically and retry.
            if ((chrome.runtime.lastError || !csResponse) && !isRetry) {
                footerText.textContent = 'Injecting extension...';
                verdictIcon.textContent = '⚙️';
                verdictTitle.textContent = 'Activating...';
                verdictSub.textContent = 'Injecting extension into this page';

                Promise.all([
                    chrome.scripting.executeScript({
                        target: { tabId: activeTabId, allFrames: true },
                        files: ['content/content.js']
                    }),
                    chrome.scripting.insertCSS({
                        target: { tabId: activeTabId, allFrames: true },
                        files: ['content/content.css']
                    })
                ])
                .then(() => {
                    // Give the script a moment to initialize, then retry.
                    setTimeout(() => queryVideoStatus(true), 300);
                })
                .catch((injectErr) => {
                    // Injection failed — likely a restricted page (chrome://, Web Store, etc.).
                    verdictIcon.textContent = '🚫';
                    verdictTitle.textContent = 'Cannot access this page';
                    verdictSub.textContent = 'Extension cannot run on browser internal pages';
                    footerText.textContent = 'Navigate to a normal website';
                    console.warn('[Popup] Script injection failed:', injectErr);
                });
                return;
            }

            // Injection succeeded but still no response — restricted page.
            if (chrome.runtime.lastError || !csResponse) {
                verdictIcon.textContent = '🚫';
                verdictTitle.textContent = 'Cannot access this page';
                verdictSub.textContent = 'Extension cannot run on browser internal pages';
                footerText.textContent = 'Navigate to a normal website';
                return;
            }

            // ── Content script responded ───────────────────────────────────────
            if (csResponse.status === 'no_video') {
                if (csResponse.videoCount > 0) {
                    verdictIcon.textContent = '📺';
                    verdictTitle.textContent = 'Video not eligible';
                    verdictSub.textContent =
                        `Found ${csResponse.videoCount} video(s) — none are playing or in view`;
                    footerText.textContent = 'Play the video and scroll it into view';
                } else {
                    setNoVideo();
                }
                return;
            }

            // Video found — show "Analysing" and also pull cached confidence.
            statusDot.classList.add('active');
            verdictIcon.textContent = '⚙️';
            verdictTitle.textContent = 'Analysing...';
            verdictSub.textContent = 'Video found — running inference';
            footerText.textContent = 'Processing frame...';

            chrome.runtime.sendMessage(
                { type: 'GET_STATUS', tabId: activeTabId },
                (bgResponse) => {
                    if (chrome.runtime.lastError) return;
                    const state = bgResponse?.state;
                    if (state?.confidence != null) {
                        applyResult(state.confidence);
                    }
                }
            );
        });
    }

    queryVideoStatus();

    // Live updates while popup is open.
    chrome.runtime.onMessage.addListener((message) => {
        if (message?.type === 'UPDATE_UI' && message.targetTabId === activeTabId) {
            applyResult(message.confidence);
        }
    });

})();
