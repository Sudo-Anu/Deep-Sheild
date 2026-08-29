// offscreen.js
// Runs inside the Offscreen Document context. Manages the local ONNX
// inference session lifecycle and processes forwarded video frames.

// ---------------------------------------------------------------------------
// ONNX Runtime WASM Configuration
// Force single-threaded CPU fallback to prevent the loader from attempting
// to fetch unavailable SIMD/multi-threaded variants (e.g. ort-wasm-simd-threaded.jsep.mjs).
// ---------------------------------------------------------------------------

// 1. Force ONNX Runtime to disable advanced features that require extra files
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = false;

// 2. Map the file loading path strictly to your local, single-threaded binary
ort.env.wasm.wasmPaths = {
    'ort-wasm.wasm': '../lib/ort-wasm.wasm'
};

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
 * Decodes a base64 JPEG data URL into a 224x224 RGB pixel buffer using the
 * hidden canvas element.
 * @param {string} dataUrl
 * @returns {Promise<Uint8ClampedArray>}
 */
function decodeFrameToPixels(dataUrl) {
    return new Promise((resolve, reject) => {
        try {
            const canvas = document.getElementById('inference-canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();

            img.onload = () => {
                try {
                    ctx.clearRect(0, 0, 224, 224);
                    ctx.drawImage(img, 0, 0, 224, 224);
                    // Preprocess raw canvas pixels [224x224 RGB] into standard ImageNet format
                    const imgData = ctx.getImageData(0, 0, 224, 224).data;
                    resolve(imgData);
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
 * Full pipeline: decode -> preprocess -> run inference -> dispose tensors.
 * @param {string} payload Base64 JPEG data URL
 * @param {number} targetTabId
 */
async function processFrame(payload, targetTabId) {
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

        const imgData = await decodeFrameToPixels(payload);
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
    if (message && message.type === 'PROCESS_FRAME') {
        const { payload, targetTabId } = message;
        processFrame(payload, targetTabId);
        // No synchronous response needed; result is sent via a separate message.
        return false;
    }
    return false;
});