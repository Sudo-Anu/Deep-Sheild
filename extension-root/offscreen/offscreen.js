// offscreen.js
// Runs inside the Offscreen Document context. Manages the local ONNX
// inference session lifecycle and processes forwarded video frames.

// ---------------------------------------------------------------------------
// ONNX Runtime WASM Configuration (v1.29.0)
//
// ORT 1.29.0 unconditionally runs a dynamic import() of the .jsep.mjs
// bootstrap before reading env flags. We must:
//   1. Point ALL asset lookups (both .wasm and .mjs) to our local lib/ folder
//      using the string-prefix form of wasmPaths (object form only intercepts
//      .wasm fetches, not ES module dynamic imports).
//   2. Keep numThreads = 1 so only the simd-threaded variant is attempted
//      (not a multi-threaded pool), which is the file we have locally.
// ---------------------------------------------------------------------------

// 1. Redirect ALL ORT asset lookups to local lib/ (covers .wasm AND .mjs files)
ort.env.wasm.wasmPaths = '../lib/';

// 2. Single inference thread — prevents worker-pool spawning
ort.env.wasm.numThreads = 1;

let session = null;

/**
 * Loads the local ONNX model and configures the WASM execution provider.
 * Runs once on script load; subsequent calls reuse the cached session.
 */
async function initializeSession() {
    try {
        console.log('[offscreen] Initializing ONNX inference session...');
        const options = { executionProviders: ['wasm'] };
        session = await ort.InferenceSession.create('../assets/mobilenetv2-7-quantized.onnx', options);
        console.log('[offscreen] Model ready. Input names:', session.inputNames, 'Output names:', session.outputNames);
    } catch (err) {
        console.error('[offscreen] Failed to initialize inference session:', err);
        session = null;
    }
}

// Kick off model preloading immediately.
initializeSession();

/**
 * Decodes a JPEG data URL and extracts a 224x224 RGB pixel buffer.
 * If srcRect is provided, the image is cropped to that region first
 * (used when the dataUrl is a full-tab screenshot rather than a pre-cropped frame).
 *
 * @param {string} dataUrl
 * @param {{ x: number, y: number, width: number, height: number, devicePixelRatio: number }|null} srcRect
 * @returns {Promise<Uint8ClampedArray>}
 */
function decodeFrameToPixels(dataUrl, srcRect = null) {
    return new Promise((resolve, reject) => {
        try {
            const canvas = document.getElementById('inference-canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();

            img.onload = () => {
                try {
                    ctx.clearRect(0, 0, 224, 224);

                    if (srcRect) {
                        // Full-tab screenshot: crop to video element position.
                        // captureVisibleTab returns pixels at device resolution,
                        // so multiply CSS coords by devicePixelRatio.
                        const dpr = srcRect.devicePixelRatio || 1;
                        ctx.drawImage(
                            img,
                            srcRect.x * dpr, srcRect.y * dpr,   // source crop origin
                            srcRect.width * dpr, srcRect.height * dpr, // source crop size
                            0, 0, 224, 224                       // dest: full 224x224 canvas
                        );
                    } else {
                        // Pre-cropped frame — draw directly.
                        ctx.drawImage(img, 0, 0, 224, 224);
                    }

                    resolve(ctx.getImageData(0, 0, 224, 224).data);
                } catch (drawErr) {
                    reject(drawErr);
                }
            };

            img.onerror = (imgErr) => {
                reject(new Error('Failed to load frame image data: ' + imgErr));
            };

            img.src = dataUrl;
        } catch (err) {
            reject(err);
        }
    });
}

/**
 * Converts interleaved RGBA pixel data into a planar, normalized
 * Float32Array laid out as RRR...GGG...BBB... for MobileNet-style input.
 * @param {Uint8ClampedArray} imgData
 * @returns {Float32Array}
 */
function preprocessPixels(imgData) {
    const numPixels = 224 * 224;
    const processedData = new Float32Array(3 * numPixels);

    // ImageNet normalization constants
    const mean = [0.485, 0.456, 0.406];
    const std  = [0.229, 0.224, 0.225];

    for (let i = 0; i < numPixels; i++) {
      // Extract RGB channels (ignoring Alpha at index i * 4 + 3)
      const r = imgData[i * 4] / 255.0;
      const g = imgData[i * 4 + 1] / 255.0;
      const b = imgData[i * 4 + 2] / 255.0;

      // Planar Layout format (All Red channels, then Green, then Blue)
      processedData[i]               = (r - mean[0]) / std[0]; // Red
      processedData[i + numPixels]     = (g - mean[1]) / std[1]; // Green
      processedData[i + 2 * numPixels] = (b - mean[2]) / std[2]; // Blue
    }

    return processedData;
}

/**
 * Applies softmax over all logits in the output tensor and returns the
 * highest class probability scaled to a 0–100 percentage.
 *
 * Using raw logits as a confidence score is incorrect because their magnitude
 * is unbounded and not comparable to the percentage thresholds used in the
 * traffic-light UI (< 40 = red, ≤ 75 = orange, > 75 = green).
 *
 * @param {ort.Tensor} outputTensor
 * @returns {number} Confidence percentage in [0, 100]
 */
function extractSoftmaxConfidence(outputTensor) {
    const data = outputTensor.data;
    const len  = data.length;

    // Numerically stable softmax: subtract max before exp.
    let maxLogit = -Infinity;
    for (let i = 0; i < len; i++) {
        if (data[i] > maxLogit) maxLogit = data[i];
    }

    let sumExp = 0;
    let maxExp = 0;
    for (let i = 0; i < len; i++) {
        const e = Math.exp(data[i] - maxLogit);
        sumExp += e;
        if (e > maxExp) maxExp = e;
    }

    // Highest softmax probability → scale to [0, 100]
    return (maxExp / sumExp) * 100;
}

/**
 * Full pipeline: decode/crop -> preprocess -> run inference -> dispose tensors.
 * @param {string} payload   JPEG data URL (full screenshot or pre-cropped frame)
 * @param {number} targetTabId
 * @param {object|null} rect Video bounding rect with devicePixelRatio (for screenshots)
 */
async function processFrame(payload, targetTabId, rect = null) {
    let inputTensor = null;
    let outputMap = null;

    try {
        if (!session) {
            console.warn('[offscreen] Session not ready yet; attempting to initialize now.');
            await initializeSession();
            if (!session) {
                throw new Error('Inference session unavailable after initialization attempt.');
            }
        }

        const imgData = await decodeFrameToPixels(payload, rect);
        const processedData = preprocessPixels(imgData);

        // Wrap in the ONNX input tensor
        inputTensor = new ort.Tensor('float32', processedData, [1, 3, 224, 224]);

        const inputName = session.inputNames && session.inputNames[0] ? session.inputNames[0] : 'input';
        outputMap = await session.run({ [inputName]: inputTensor });

        const outputName = session.outputNames && session.outputNames[0] ? session.outputNames[0] : Object.keys(outputMap)[0];
        const outputTensor = outputMap[outputName];
        const highestConfidenceScore = extractSoftmaxConfidence(outputTensor);

        chrome.runtime.sendMessage({
            type: 'INFERENCE_RESULT',
            confidence: highestConfidenceScore,
            targetTabId: targetTabId
        });

        console.log('[offscreen] Inference complete. Confidence:', highestConfidenceScore);
    } catch (err) {
        console.error('[offscreen] Error during frame processing:', err);
    } finally {
        // Free WASM memory allocations to prevent leaks.
        try {
            if (inputTensor && typeof inputTensor.dispose === 'function') {
                inputTensor.dispose();
            }
            if (outputMap) {
                for (const key of Object.keys(outputMap)) {
                    const tensor = outputMap[key];
                    if (tensor && typeof tensor.dispose === 'function') {
                        tensor.dispose();
                    }
                }
            }
        } catch (disposeErr) {
            console.error('[offscreen] Error disposing tensors:', disposeErr);
        }
    }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    // Full-tab screenshot with crop rect — primary path (no canvas taint issues).
    if (message?.type === 'PROCESS_SCREENSHOT') {
        const { payload, rect, targetTabId } = message;
        processFrame(payload, targetTabId, rect);
        return false;
    }

    // Pre-cropped frame — legacy / fallback path.
    if (message?.type === 'PROCESS_FRAME') {
        const { payload, targetTabId } = message;
        processFrame(payload, targetTabId, null);
        return false;
    }

    return false;
});