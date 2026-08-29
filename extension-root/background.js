const OFFSCREEN_DOCUMENT_PATH = 'offscreen/offscreen.html';

/**
 * Storage key helper — namespaces per-tab state in chrome.storage.session.
 * @param {number} tabId
 * @returns {string}
 */
function tabKey(tabId) {
    return `tab_state_${tabId}`;
}

/**
 * Ensures that the required Offscreen Document is active.
 *
 * @param {string} path - Relative path to the offscreen HTML document.
 * @returns {Promise<void>}
 */
async function setupOffscreenDocument(path) {
    try {
        console.log('[Background] Checking Offscreen Document status...');

        const hasDocument = await chrome.offscreen.hasDocument();

        if (hasDocument) {
            console.log('[Background] Offscreen Document is already active.');
            return;
        }

        console.log('[Background] Offscreen Document not found. Creating...');

        await chrome.offscreen.createDocument({
            url: path,
            reasons: [chrome.offscreen.Reason.WORKERS],
            justification:
                'Executing local ONNX neural network inference via WebAssembly.'
        });

        console.log('[Background] Offscreen Document created successfully.');
    } catch (error) {
        console.error(
            '[Background] Failed to initialize Offscreen Document:',
            error
        );

        throw error;
    }
}

/**
 * Centralized communication broker.
 *
 * Routes messages between:
 * Content Script <-> Background Service Worker <-> Offscreen Document
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('[Background] Message received:', message);

    /**
     * Node A: Tab Screenshot Capture
     *
     * Content Script sends the video's bounding rect.
     * Background screenshots the tab with captureVisibleTab() — this works on
     * ANY video regardless of origin (file://, cross-origin, DRM) because it
     * captures rendered screen pixels, not the video element directly.
     */
    if (message?.type === 'REQUEST_TAB_CAPTURE') {
        const originatingTabId = sender?.tab?.id;
        const { rect } = message;

        if (typeof originatingTabId !== 'number') {
            console.error('[Background] REQUEST_TAB_CAPTURE rejected: sender tab ID unavailable.');
            sendResponse({ success: false, error: 'Tab ID unavailable.' });
            return true;
        }

        console.log(`[Background] REQUEST_TAB_CAPTURE from tab ${originatingTabId}, rect:`, rect);

        (async () => {
            try {
                await setupOffscreenDocument(OFFSCREEN_DOCUMENT_PATH);

                // captureVisibleTab screenshots the ACTIVE tab at device resolution.
                // No canvas/CORS restrictions apply — works on any video source.
                const screenshotDataUrl = await chrome.tabs.captureVisibleTab(
                    null, // current window
                    { format: 'jpeg', quality: 70 }
                );

                await chrome.runtime.sendMessage({
                    type: 'PROCESS_SCREENSHOT',
                    payload: screenshotDataUrl,
                    rect,                        // video bounding rect (CSS px + DPR)
                    targetTabId: originatingTabId
                });

                sendResponse({ success: true });
            } catch (error) {
                console.error(`[Background] Tab capture failed for tab ${originatingTabId}:`, error);
                sendResponse({ success: false, error: error.message });
            }
        })();

        return true; // async response
    }

    /**
     * Node P: Popup Status Query
     *
     * Popup -> Background: returns cached confidence for the requested tab.
     */
    if (message?.type === 'GET_STATUS') {
        const { tabId } = message;
        const key = tabKey(tabId);
        chrome.storage.session.get([key], (result) => {
            sendResponse({ state: result[key] ?? null });
        });
        return true; // async response
    }

    /**
     * Node C: Content script reports no video found on this tab.
     */
    if (message?.type === 'NO_VIDEO') {
        const tabId = sender?.tab?.id;
        if (typeof tabId === 'number') {
            const key = tabKey(tabId);
            // Only overwrite if there's no real confidence result yet.
            chrome.storage.session.get([key], (result) => {
                if (!result[key]?.confidence) {
                    chrome.storage.session.set({ [key]: { status: 'no_video' } });
                }
            });
        }
        return false;
    }

    /**
     * Node D: Content script reports canvas cross-origin taint.
     */
    if (message?.type === 'CANVAS_TAINTED') {
        const tabId = sender?.tab?.id;
        if (typeof tabId === 'number') {
            chrome.storage.session.set({ [tabKey(tabId)]: { status: 'tainted' } });
        }
        return false;
    }

    /**
     * Node B: Model Inference Result
     *
     * Offscreen Document -> Background -> Content Script
     */
    if (message?.type === 'INFERENCE_RESULT') {
        const { confidence, targetTabId } = message;

        if (typeof targetTabId !== 'number') {
            console.error('[Background] INFERENCE_RESULT rejected: target tab ID is invalid.');
            sendResponse({ success: false, error: 'Invalid target tab ID.' });
            return true;
        }

        // Persist result so popup reads it even after SW restart.
        chrome.storage.session.set({ [tabKey(targetTabId)]: { confidence } });

        console.log(`[Background] INFERENCE_RESULT for tab ${targetTabId}. Confidence: ${confidence}`);

        (async () => {
            // ── 1. Broadcast to all extension pages (popup, etc.) ────────────────
            // chrome.tabs.sendMessage only reaches content scripts.
            // chrome.runtime.sendMessage broadcasts to ALL extension pages,
            // so the popup receives live updates while it is open.
            chrome.runtime.sendMessage({
                type: 'UPDATE_UI',
                confidence,
                targetTabId
            }).catch(() => {
                // Popup may not be open — ignore the "no listener" error.
            });

            // ── 2. Also push to the in-page traffic light overlay ─────────────
            try {
                await chrome.tabs.sendMessage(targetTabId, {
                    type: 'UPDATE_UI',
                    confidence
                });
                console.log(`[Background] UPDATE_UI delivered to tab ${targetTabId}.`);
            } catch (tabErr) {
                // Tab may have closed or navigated — not fatal.
                console.warn(`[Background] Could not deliver UPDATE_UI to tab ${targetTabId}:`, tabErr);
            }

            sendResponse({ success: true });
        })();

        return true;
    }

    console.log(
        `[Background] Ignoring unsupported message type: ${message?.type ?? 'undefined'}`
    );

    sendResponse({
        success: false,
        error: 'Unsupported message type.'
    });

    return true;
});

console.log('[Background] Manifest V3 Service Worker initialized.');