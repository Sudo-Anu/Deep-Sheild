// content/content.js

(() => {
    "use strict";

    // ============================================================
    // Configuration
    // ============================================================

    const FRAME_INTERVAL = 1000; // 1 FPS
    const CANVAS_SIZE = 224;

    // ============================================================
    // Global Canvas
    // Created once and reused for every frame.
    // ============================================================

    const canvas = document.createElement("canvas");
    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;

    const ctx = canvas.getContext("2d", {
        alpha: false,
        willReadFrequently: false
    });

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

    const style = document.createElement("style");

    style.textContent = `
        #ai-confidence-indicator {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 2147483647;

            display: flex;
            align-items: center;
            gap: 8px;

            padding: 8px 10px;

            background: rgba(20, 20, 20, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;

            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);

            box-shadow:
                0 4px 20px rgba(0, 0, 0, 0.25);

            pointer-events: none;
            user-select: none;
        }

        .confidence-light {
            width: 14px;
            height: 14px;

            border-radius: 50%;

            opacity: 0.2;
            transform: scale(0.85);

            transition:
                opacity 250ms ease,
                transform 250ms ease,
                box-shadow 250ms ease;
        }

        .confidence-light.red {
            background: #ff3b30;
        }

        .confidence-light.orange {
            background: #ff9500;
        }

        .confidence-light.green {
            background: #34c759;
        }

        .confidence-light.active {
            opacity: 1;
            transform: scale(1.15);
        }

        .confidence-light.red.active {
            box-shadow: 0 0 12px rgba(255, 59, 48, 0.8);
        }

        .confidence-light.orange.active {
            box-shadow: 0 0 12px rgba(255, 149, 0, 0.8);
        }

        .confidence-light.green.active {
            box-shadow: 0 0 12px rgba(52, 199, 89, 0.8);
        }
    `;

    // Inject UI styles and indicator.
    document.documentElement.appendChild(style);
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
        if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
            return false;
        }

        if (video.paused || video.ended) {
            return false;
        }

        if (video.readyState < HTMLMediaElement.HAVE_FUTURE_DATA) {
            return false;
        }

        const rect = video.getBoundingClientRect();

        if (
            rect.width <= 0 ||
            rect.height <= 0
        ) {
            return false;
        }

        // Ensure the video is inside the viewport.
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;

        const horizontallyVisible =
            rect.right > 0 &&
            rect.left < viewportWidth;

        const verticallyVisible =
            rect.bottom > 0 &&
            rect.top < viewportHeight;

        if (!horizontallyVisible || !verticallyVisible) {
            return false;
        }

        // Require the center point to be inside the viewport.
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;

        if (
            centerX < 0 ||
            centerX > viewportWidth ||
            centerY < 0 ||
            centerY > viewportHeight
        ) {
            return false;
        }

        return true;
    }

    // ============================================================
    // Frame Capture
    // ============================================================

    function captureFrame() {
        const video = findTargetVideo();

        if (!video) {
            return;
        }

        try {
            ctx.drawImage(
                video,
                0,
                0,
                CANVAS_SIZE,
                CANVAS_SIZE
            );

            const frameDataString = canvas.toDataURL(
                "image/jpeg",
                0.6
            );

            chrome.runtime.sendMessage({
                type: "FRAME_DATA",
                payload: frameDataString
            }).catch(() => {
                // Background worker may temporarily be unavailable.
            });

        } catch (error) {
            console.warn(
                "[AI Detector] Frame capture failed:",
                error
            );
        }
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

    chrome.runtime.onMessage.addListener((message) => {
        if (!message || message.type !== "UPDATE_UI") {
            return;
        }

        updateTrafficLight(message.confidence);
    });

    // ============================================================
    // 1 FPS Capture Loop
    // ============================================================

    const captureInterval = setInterval(
        captureFrame,
        FRAME_INTERVAL
    );

    // Prevent accidental unused-variable optimization warnings
    // while keeping the interval alive for the lifetime of the page.
    void captureInterval;

})();