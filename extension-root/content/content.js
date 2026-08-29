// content/content.js

(() => {
    "use strict";

    // ============================================================
    // Configuration
    // ============================================================

    const FRAME_INTERVAL = 1000; // 1 FPS
    const CANVAS_SIZE = 224;

    // ============================================================
    // Traffic Light UI
    // ============================================================

    const indicator = document.createElement("div");
    indicator.id = "ai-confidence-indicator";

    indicator.innerHTML = `
        <div class="confidence-light red" data-level="red"></div>
        <div class="confidence-light orange" data-level="orange"></div>
        <div class="confidence-light green" data-level="green"></div>
    `;

    // Styles are defined in content.css (injected via manifest content_scripts).
    // Do NOT add duplicate inline styles here — content.css wins and the
    // duplicate caused a flex-direction conflict (column vs row).
    document.body?.appendChild(indicator);

    // Handle pages where body is not available immediately.
    if (!document.body) {
        window.addEventListener("DOMContentLoaded", () => {
            if (!indicator.isConnected) {
                document.body.appendChild(indicator);
            }
        }, { once: true });
    }

    const lights = {
        red: indicator.querySelector('[data-level="red"]'),
        orange: indicator.querySelector('[data-level="orange"]'),
        green: indicator.querySelector('[data-level="green"]')
    };

    // ============================================================
    // Video Detection
    // ============================================================

    function findTargetVideo() {
        const videos = Array.from(document.querySelectorAll("video"));

        let largestVisibleVideo = null;
        let largestArea = 0;

        for (const video of videos) {
            if (!isVideoVisible(video)) {
                continue;
            }

            const rect = video.getBoundingClientRect();
            const area = rect.width * rect.height;

            if (area > largestArea) {
                largestArea = area;
                largestVisibleVideo = video;
            }
        }

        return largestVisibleVideo;
    }

    // ============================================================
    // Video Visibility / Playback Guard
    // ============================================================

    function isVideoVisible(video) {
        if (!video) return false;

        // Accept any video that has at least metadata (dimensions known).
        // HAVE_CURRENT_DATA (2) is ideal but HAVE_METADATA (1) allows paused frames.
        if (video.readyState < HTMLMediaElement.HAVE_METADATA) {
            return false;
        }

        // Reject only fully ended videos — paused is OK for deepfake analysis.
        if (video.ended) {
            return false;
        }

        const rect = video.getBoundingClientRect();

        // Must have non-zero rendered dimensions.
        if (rect.width <= 0 || rect.height <= 0) {
            return false;
        }

        // Must overlap the viewport at least partially.
        const vw = window.innerWidth;
        const vh = window.innerHeight;

        const inViewport =
            rect.right > 0 && rect.left < vw &&
            rect.bottom > 0 && rect.top < vh;

        return inViewport;
    }

    // ============================================================
    // Frame Capture
    // ============================================================

    // Capture loop interval reference — kept in outer scope so captureFrame
    // can clear it when it detects the extension context has been invalidated.
    let captureInterval = null;

    /**
     * Sends a message to the background, self-terminating if the extension
     * context has been invalidated (i.e. the extension was reloaded while this
     * content script instance was still running as a zombie).
     * @param {object} msg
     * @returns {Promise<void>}
     */
    function safeSendMessage(msg) {
        let promise;
        try {
            promise = chrome.runtime.sendMessage(msg);
        } catch (syncErr) {
            // sendMessage throws synchronously on fully invalidated contexts.
            if (syncErr?.message?.includes('Extension context')) {
                clearInterval(captureInterval);
            }
            return Promise.resolve();
        }
        return promise.catch((asyncErr) => {
            if (asyncErr?.message?.includes('Extension context')) {
                clearInterval(captureInterval);
            }
        });
    }

    function captureFrame() {
        // Quick guard — belt-and-suspenders check before doing any work.
        if (!chrome.runtime?.id) {
            clearInterval(captureInterval);
            return;
        }

        const allVideos = Array.from(document.querySelectorAll('video'));
        const video = findTargetVideo();

        if (!video) {
            if (allVideos.length === 0) {
                console.debug('[AI Detector] No <video> elements found on this page.');
            } else {
                console.debug(
                    `[AI Detector] Found ${allVideos.length} video(s) but none passed visibility check.`,
                    allVideos.map(v => ({
                        src: v.currentSrc?.slice(0, 60),
                        readyState: v.readyState,
                        paused: v.paused,
                        ended: v.ended,
                        size: `${Math.round(v.getBoundingClientRect().width)}x${Math.round(v.getBoundingClientRect().height)}`
                    }))
                );
            }
            safeSendMessage({ type: 'NO_VIDEO' });
            return;
        }

        // Send only the video's bounding rect to the background.
        // The background uses captureVisibleTab() to take the actual screenshot,
        // which bypasses all canvas cross-origin and file:// restrictions.
        const rect = video.getBoundingClientRect();
        safeSendMessage({
            type: 'REQUEST_TAB_CAPTURE',
            rect: {
                x: Math.round(rect.x),
                y: Math.round(rect.y),
                width: Math.round(rect.width),
                height: Math.round(rect.height),
                devicePixelRatio: window.devicePixelRatio || 1
            }
        });
    }

    // ============================================================
    // Traffic Light State Management
    // ============================================================

    function updateTrafficLight(confidence) {
        const score = Number(confidence);

        if (!Number.isFinite(score)) {
            return;
        }

        lights.red.classList.remove("active");
        lights.orange.classList.remove("active");
        lights.green.classList.remove("active");

        if (score < 40) {
            lights.red.classList.add("active");

        } else if (score <= 75) {
            lights.orange.classList.add("active");

        } else {
            lights.green.classList.add("active");
        }
    }

    // ============================================================
    // Background Communication
    // ============================================================

    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
        // ── Live UI update from background after inference ──
        if (message?.type === 'UPDATE_UI') {
            updateTrafficLight(message.confidence);
            return false;
        }

        // ── Direct status query from popup ──────────────────
        // Popup uses this to bypass the background cache,
        // which may be empty after a service worker restart.
        if (message?.type === 'GET_VIDEO_STATUS') {
            const allVideos = Array.from(document.querySelectorAll('video'));
            const video = findTargetVideo();

            if (!video) {
                sendResponse({
                    status: 'no_video',
                    videoCount: allVideos.length,
                    detail: allVideos.map(v => ({
                        readyState: v.readyState,
                        paused: v.paused,
                        ended: v.ended,
                        w: Math.round(v.getBoundingClientRect().width),
                        h: Math.round(v.getBoundingClientRect().height)
                    }))
                });
            } else {
                sendResponse({ status: 'found', videoCount: allVideos.length });
                // Kick off an immediate capture so inference starts ASAP.
                captureFrame();
            }
            return true;
        }

        return false;
    });

    // ============================================================
    // 1 FPS Capture Loop
    // ============================================================

    captureInterval = setInterval(captureFrame, FRAME_INTERVAL);

})();