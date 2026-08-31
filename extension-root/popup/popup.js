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

    // Audio confidence DOM refs.
    const audioConfSection = document.getElementById('audio-confidence-section');
    const audioConfValue   = document.getElementById('audio-confidence-value');
    const audioConfFill    = document.getElementById('audio-confidence-fill');

    // ── Helpers ───────────────────────────────────────────────────────────────

    function setIdle() {
        [lightRed, lightOrange, lightGreen].forEach(l => l.classList.remove('active'));
        verdictBox.className = 'verdict-box';
        verdictIcon.textContent = '🔍';
        verdictTitle.textContent = 'Waiting for video...';
        verdictSub.textContent = 'Open a page with a playing video';
        confSection.style.display = 'none';
        audioConfSection.style.display = 'none';
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
        audioConfSection.style.display = 'none';
        statusDot.classList.remove('active');
        footerText.textContent = 'Extension active — no video';
    }

    function applyResult(confidence, audioConfidence, audioStatus) {
        // Worst-case merged score: lowest of the two available signals.
        const videoScore = Number.isFinite(Number(confidence)) ? Number(confidence) : null;
        const audioScore = Number.isFinite(Number(audioConfidence)) ? Number(audioConfidence) : null;

        // Determine display score — worst-case when both present.
        const score =
            videoScore !== null && audioScore !== null ? Math.min(videoScore, audioScore) :
            videoScore !== null ? videoScore :
            audioScore !== null ? audioScore : null;

        if (score === null) { setIdle(); return; }

        // ── Video bar ─────────────────────────────────────────
        if (videoScore !== null) {
            confSection.style.display = 'flex';
            confValue.textContent = videoScore.toFixed(1) + '%';
            confFill.style.width = Math.min(videoScore, 100) + '%';
            confFill.className = 'confidence-fill ' + scoreClass(videoScore);
        }

        // ── Audio bar ─────────────────────────────────────────
        if (audioScore !== null) {
            audioConfSection.style.display = 'flex';
            audioConfValue.textContent = audioScore.toFixed(1) + '%';
            audioConfFill.style.width = Math.min(audioScore, 100) + '%';
            audioConfFill.className = 'confidence-fill ' + scoreClass(audioScore);
        } else if (audioStatus === 'no_model') {
            // ONNX model file not present yet — tell the user clearly.
            audioConfSection.style.display = 'flex';
            audioConfValue.textContent = 'No model';
            audioConfFill.style.width = '0%';
            audioConfFill.className = 'confidence-fill';
        } else {
            // Model loaded but still buffering the first 3-second chunk.
            audioConfSection.style.display = 'flex';
            audioConfValue.textContent = 'Scanning…';
            audioConfFill.style.width = '0%';
            audioConfFill.className = 'confidence-fill';
        }

        // ── Shared traffic-light + verdict (worst-case merged score) ────────
        statusDot.classList.add('active');
        [lightRed, lightOrange, lightGreen].forEach(l => l.classList.remove('active'));

        if (score < 40) {
            lightRed.classList.add('active');
            verdictBox.className = 'verdict-box ai';
            verdictIcon.textContent = '⚠️';
            verdictTitle.textContent = 'Likely AI-Generated';
            verdictSub.textContent = `Merged authenticity score (${score.toFixed(1)}%)`;
            footerText.textContent = 'AI-generated content suspected';

        } else if (score <= 75) {
            lightOrange.classList.add('active');
            verdictBox.className = 'verdict-box mixed';
            verdictIcon.textContent = '🟡';
            verdictTitle.textContent = 'Uncertain — Mixed Signals';
            verdictSub.textContent = `Merged authenticity score (${score.toFixed(1)}%)`;
            footerText.textContent = 'Content authenticity unclear';

        } else {
            lightGreen.classList.add('active');
            verdictBox.className = 'verdict-box real';
            verdictIcon.textContent = '✅';
            verdictTitle.textContent = 'Likely Real Content';
            verdictSub.textContent = `Merged authenticity score (${score.toFixed(1)}%)`;
            footerText.textContent = 'Content appears authentic';
        }
    }

    /** Maps a score to a CSS modifier class for the confidence bar. */
    function scoreClass(score) {
        if (score < 40)  return 'low';
        if (score <= 75) return 'mid';
        return 'high';
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
            if (csResponse.status === 'audio_only') {
                // Audio-only page (e.g. .mp3 opened directly in browser)
                statusDot.classList.add('active');
                verdictIcon.textContent = '🎙️';
                verdictTitle.textContent = 'Audio-only content';
                verdictSub.textContent = 'Monitoring audio for voice cloning';
                footerText.textContent = 'Video analysis not applicable';
                // Still pull any cached audio result.
                chrome.runtime.sendMessage(
                    { type: 'GET_STATUS', tabId: activeTabId },
                    (bgResponse) => {
                        if (chrome.runtime.lastError) return;
                        const state = bgResponse?.state;
                        if (state?.audioConfidence != null) {
                            applyResult(null, state.audioConfidence, state.audioStatus);
                        }
                    }
                );
                return;
            }

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
                    if (state?.confidence != null || state?.audioConfidence != null) {
                        applyResult(state.confidence, state.audioConfidence);
                    }
                }
            );
        });
    }

    queryVideoStatus();

    // Live updates while popup is open.
    chrome.runtime.onMessage.addListener((message) => {
        if (message?.type === 'UPDATE_UI' && message.targetTabId === activeTabId) {
            applyResult(message.confidence, message.audioConfidence, message.audioStatus);
        }
    });

})();
