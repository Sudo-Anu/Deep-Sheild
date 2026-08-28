// offscreen.js
// Runs inside the Offscreen Document context. Manages the local ONNX
// inference session lifecycle and processes forwarded video frames.

let session = null;

/**
 * Loads the local ONNX model and configures the WASM execution provider.
 * Runs once on script load; subsequent calls reuse the cached session.
 */
async function initializeSession() {
    try {
        console.log('[offscreen] Initializing ONNX inference session...');
        const options = { executionProviders: ['wasm'] };
        session = await ort.InferenceSession.create('../assets/model.onnx', options);
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
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
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
    const width = 224;
    const height = 224;
    const numPixels = width * height;
    const processedData = new Float32Array(3 * numPixels);

    // ImageNet mean/std normalization, applied per channel.
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let i = 0; i < numPixels; i++) {
        const offset = i * 4; // RGBA stride in source data
        const r = imgData[offset] / 255;
        const g = imgData[offset + 1] / 255;
        const b = imgData[offset + 2] / 255;

        processedData[i] = (r - mean[0]) / std[0];                  // R plane
        processedData[numPixels + i] = (g - mean[1]) / std[1];      // G plane
        processedData[2 * numPixels + i] = (b - mean[2]) / std[2];  // B plane
    }

    return processedData;
}

/**
 * Finds the highest confidence score in a model output tensor.
 * @param {ort.Tensor} outputTensor
 * @returns {number}
 */
function extractHighestConfidence(outputTensor) {
    const data = outputTensor.data;
    let highest = -Infinity;
    for (let i = 0; i < data.length; i++) {
        if (data[i] > highest) {
            highest = data[i];
        }
    }
    return highest;
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

        inputTensor = new ort.Tensor('float32', processedData, [1, 3, 224, 224]);

        const inputName = session.inputNames && session.inputNames[0] ? session.inputNames[0] : 'input';
        outputMap = await session.run({ [inputName]: inputTensor });

        const outputName = session.outputNames && session.outputNames[0] ? session.outputNames[0] : Object.keys(outputMap)[0];
        const outputTensor = outputMap[outputName];
        const highestConfidenceScore = extractHighestConfidence(outputTensor);

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