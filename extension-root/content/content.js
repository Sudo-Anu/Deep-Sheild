// content/content.js

(() => {
    "use strict";

    // ============================================================
    // Configuration
    // ============================================================

    const FRAME_INTERVAL      = 1000; // 1 FPS — video frame capture rate
    const CANVAS_SIZE          = 224;

    // Audio capture: collect 3 seconds of PCM then send for inference.
    const AUDIO_BUFFER_SECONDS = 3;
    const AUDIO_SAMPLE_RATE    = 44100; // native browser rate; offscreen resamples to 16kHz
    const SCRIPT_BUFFER_SIZE   = 4096;  // ScriptProcessorNode buffer (samples per callback)

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

    // ============================================================
    // Audio Capture State
    // ============================================================

    let audioContext       = null;  // Web Audio context
    let audioSourceNode    = null;  // MediaStreamAudioSourceNode
    let scriptProcessor    = null;  // ScriptProcessorNode collecting PCM
    let pcmBuffer          = [];    // Accumulated Float32 samples
    let currentAudioVideo  = null;  // The <video> whose stream is currently captured
    const TARGET_SAMPLES   = AUDIO_SAMPLE_RATE * AUDIO_BUFFER_SECONDS;

    /**
     * Tears down any existing Web Audio pipeline.
     * Safe to call even if no pipeline is active.
     */
    function stopAudioCapture() {
        try { scriptProcessor?.disconnect(); } catch (_) {}
        try { audioSourceNode?.disconnect(); } catch (_) {}
        try { if (audioContext?.state !== 'closed') audioContext?.close(); } catch (_) {}
        scriptProcessor   = null;
        audioSourceNode   = null;
        audioContext      = null;
        currentAudioVideo = null;
        pcmBuffer         = [];
    }

    /**
     * Attaches a Web Audio capture pipeline to a <video> element.
     * Collects PCM samples and fires REQUEST_AUDIO_INFERENCE every 3 seconds.
     *
     * Gracefully no-ops if:
     * - captureStream() is unavailable or throws (DRM / cross-origin).
     * - AudioContext construction fails (e.g. restrictive CSP).
     *
     * @param {HTMLVideoElement} video
     */
    function startAudioCapture(video) {
        // Already capturing this video — nothing to do.
        if (currentAudioVideo === video) return;

        // Tear down any previous capture before attaching to the new video.
        stopAudioCapture();

        let stream;
        try {
            // captureStream() may throw on cross-origin videos (SecurityError).
            stream = video.captureStream();
        } catch (err) {
            console.debug('[AI Detector] captureStream() unavailable:', err.message);
            return;
        }

        // Ensure the stream has at least one audio track.
        if (!stream.getAudioTracks().length) {
            console.debug('[AI Detector] Video has no audio tracks — skipping audio capture.');
            return;
        }

        try {
            audioContext = new AudioContext({ sampleRate: AUDIO_SAMPLE_RATE });
        } catch (err) {
            console.warn('[AI Detector] Could not create AudioContext:', err.message);
            return;
        }

        try {
            audioSourceNode = audioContext.createMediaStreamSource(stream);

            // ScriptProcessorNode is deprecated but universally available in MV3
            // offscreen and content script contexts without SharedArrayBuffer.
            // An AudioWorklet would require fetching an external worklet script,
            // which conflicts with MV3 CSP — ScriptProcessor is the safe choice here.
            scriptProcessor = audioContext.createScriptProcessor(
                SCRIPT_BUFFER_SIZE,
                1,   // mono input
                1    // mono output (passthrough, speakers not affected)
            );

            scriptProcessor.onaudioprocess = (event) => {
                if (!chrome.runtime?.id) {
                    stopAudioCapture();
                    return;
                }

                const channelData = event.inputBuffer.getChannelData(0);
                // Append samples into the rolling buffer.
                for (let i = 0; i < channelData.length; i++) {
                    pcmBuffer.push(channelData[i]);
                }

                if (pcmBuffer.length >= TARGET_SAMPLES) {
                    // Snapshot the buffer and reset immediately (non-blocking).
                    const chunk = pcmBuffer.slice(0, TARGET_SAMPLES);
                    pcmBuffer = [];

                    // Send as plain Array — structured-clone can transfer this
                    // reliably across the content -> background boundary.
                    safeSendMessage({
                        type:       'REQUEST_AUDIO_INFERENCE',
                        samples:    Array.from(chunk),
                        sampleRate: AUDIO_SAMPLE_RATE
                    });

                    console.debug(`[AI Detector] Audio chunk sent: ${chunk.length} samples @ ${AUDIO_SAMPLE_RATE}Hz`);
                }
            };

            // Connect: source → processor → destination (required to keep graph alive).
            audioSourceNode.connect(scriptProcessor);
            scriptProcessor.connect(audioContext.destination);

            currentAudioVideo = video;
            console.debug('[AI Detector] Audio capture started.');
        } catch (err) {
            console.warn('[AI Detector] Failed to build audio graph:', err.message);
            stopAudioCapture();
        }
    }


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

        // Check whether this element actually has a video track.
        // Chrome's built-in audio player renders audio files as a <video> element
        // with videoWidth === 0 — running the vision model on its UI screenshot
        // produces meaningless scores. Audio-only elements go straight to audio capture.
        const hasVideoTrack = video.videoWidth > 0 && video.videoHeight > 0;

        if (!hasVideoTrack) {
            console.debug('[AI Detector] Audio-only element — skipping video inference.');
            // Still attempt audio capture, then tell background/popup this tab is audio-only.
            startAudioCapture(video);
            safeSendMessage({ type: 'AUDIO_ONLY' });
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

        // Start audio capture on the same video element (no-op if already running).
        startAudioCapture(video);
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
            // Prefer mergedScore (worst-case of video + audio) when available.
            // Fall back to whichever individual score is present.
            const displayScore =
                message.mergedScore   != null ? message.mergedScore   :
                message.confidence    != null ? message.confidence    :
                message.audioConfidence != null ? message.audioConfidence : null;

            updateTrafficLight(displayScore);
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