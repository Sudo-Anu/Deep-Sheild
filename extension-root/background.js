const OFFSCREEN_DOCUMENT_PATH = 'offscreen/offscreen.html';

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
            reasons: [chrome.offscreen.Reason.BLOBS],
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
     * Node A: Frame Capture
     *
     * Content Script -> Background -> Offscreen Document
     */
    if (message?.type === 'FRAME_DATA') {
        const originatingTabId = sender?.tab?.id;

        if (typeof originatingTabId !== 'number') {
            console.error(
                '[Background] FRAME_DATA rejected: sender tab ID is unavailable.'
            );

            sendResponse({
                success: false,
                error: 'Originating tab ID is unavailable.'
            });

            return true;
        }

        console.log(
            `[Background] FRAME_DATA received from tab ${originatingTabId}.`
        );

        (async () => {
            try {
                // Ensure the inference engine's Offscreen Document is running.
                await setupOffscreenDocument(OFFSCREEN_DOCUMENT_PATH);

                console.log(
                    `[Background] Forwarding FRAME_DATA from tab ${originatingTabId} to Offscreen Document...`
                );

                await chrome.runtime.sendMessage({
                    type: 'PROCESS_FRAME',
                    payload: message.payload,
                    targetTabId: originatingTabId
                });

                console.log(
                    `[Background] FRAME_DATA successfully forwarded for tab ${originatingTabId}.`
                );

                sendResponse({
                    success: true
                });
            } catch (error) {
                console.error(
                    `[Background] Failed to forward FRAME_DATA for tab ${originatingTabId}:`,
                    error
                );

                sendResponse({
                    success: false,
                    error: error instanceof Error ? error.message : String(error)
                });
            }
        })();

        // Keep the message channel open for the asynchronous response.
        return true;
    }

    /**
     * Node B: Model Inference Result
     *
     * Offscreen Document -> Background -> Content Script
     */
    if (message?.type === 'INFERENCE_RESULT') {
        const { confidence, targetTabId } = message;

        console.log(
            `[Background] INFERENCE_RESULT received for tab ${targetTabId}. Confidence: ${confidence}`
        );

        if (typeof targetTabId !== 'number') {
            console.error(
                '[Background] INFERENCE_RESULT rejected: target tab ID is invalid.'
            );

            sendResponse({
                success: false,
                error: 'Invalid target tab ID.'
            });

            return true;
        }

        (async () => {
            try {
                console.log(
                    `[Background] Sending UPDATE_UI to originating tab ${targetTabId}...`
                );

                await chrome.tabs.sendMessage(targetTabId, {
                    type: 'UPDATE_UI',
                    confidence
                });

                console.log(
                    `[Background] UPDATE_UI successfully delivered to tab ${targetTabId}.`
                );

                sendResponse({
                    success: true
                });
            } catch (error) {
                // The tab may have been closed, navigated, or reloaded while
                // inference was running. Prevent an unhandled rejection.
                console.warn(
                    `[Background] Could not deliver UPDATE_UI to tab ${targetTabId}:`,
                    error
                );

                sendResponse({
                    success: false,
                    error: error instanceof Error ? error.message : String(error)
                });
            }
        })();

        // Keep the message channel open for the asynchronous response.
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