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

// ---------------------------------------------------------------------------
// Audio ONNX Session
//
// A second, independent inference session for voice-clone detection.
// Loaded from assets/audio_clone_detector.onnx.
// Input:  [1, 1, 128, 128]  — 128 mel-filter-bank × 128 time-frame log-mel spectrogram
// Output: [1, 2]            — [real_logit, fake_logit] (same layout as vision model)
// ---------------------------------------------------------------------------
let audioSession = null;


/**
 * Loads the local ONNX model and configures the WASM execution provider.
 * Runs once on script load; subsequent calls reuse the cached session.
 */
async function initializeSession() {
    try {
        console.log('[offscreen] Initializing ONNX inference session...');
        const options = { executionProviders: ['wasm'] };
        session = await ort.InferenceSession.create('../assets/deepfake_efficientnet_b0_int8.onnx', options);
        console.log('[offscreen] Model ready. Input names:', session.inputNames, 'Output names:', session.outputNames);
    } catch (err) {
        console.error('[offscreen] Failed to initialize inference session:', err);
        session = null;
    }
}

// Kick off model preloading immediately.
initializeSession();

/**
 * Loads the audio clone detector ONNX model.
 * Runs once; subsequent calls reuse the cached audioSession.
 */
async function initializeAudioSession() {
    try {
        console.log('[offscreen] Initializing audio ONNX session...');
        const options = { executionProviders: ['wasm'] };
        audioSession = await ort.InferenceSession.create('../assets/audio_clone_detector.onnx', options);
        console.log('[offscreen] Audio model ready. Inputs:', audioSession.inputNames, 'Outputs:', audioSession.outputNames);
    } catch (err) {
        // Non-fatal: model may not be present yet. Audio inference will be skipped.
        console.warn('[offscreen] Audio model not loaded (expected if model file is absent):', err.message);
        audioSession = null;
    }
}

// Kick off audio model preloading immediately (non-blocking).
initializeAudioSession();


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
 * Converts the model's 2-class output logits into an authenticity confidence
 * score in the range [0, 100].
 *
 * Model output layout (EfficientNet-B0 deepfake classifier):
 *   index 0 = "real" logit
 *   index 1 = "fake" logit
 *
 * We apply softmax to get probabilities, then return:
 *   real_probability × 100  →  score near 100 = authentic, near 0 = AI-generated
 *
 * This maps cleanly to the traffic-light thresholds:
 *   score < 40   → Red  (Likely AI-generated)
 *   score ≤ 75   → Orange (Uncertain)
 *   score > 75   → Green (Likely Real)
 *
 * @param {ort.Tensor} outputTensor
 * @returns {number} Authenticity percentage in [0, 100]
 */
function extractConfidence(outputTensor) {
    const data = outputTensor.data;

    if (data.length === 1) {
        // Single sigmoid output: value near 1 = fake, near 0 = real.
        // Convert to authenticity: (1 - fake_prob) * 100
        const fakeProbability = 1 / (1 + Math.exp(-data[0]));
        return (1 - fakeProbability) * 100;
    }

    // 2-class softmax output: [real_logit, fake_logit]
    const realLogit = data[0];
    const fakeLogit = data[1];

    // Numerically stable softmax
    const maxLogit = Math.max(realLogit, fakeLogit);
    const expReal = Math.exp(realLogit - maxLogit);
    const expFake = Math.exp(fakeLogit - maxLogit);
    const sum     = expReal + expFake;

    const realProbability = expReal / sum;

    return realProbability * 100;
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
        const highestConfidenceScore = extractConfidence(outputTensor);

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

    // Raw PCM audio chunk for voice-clone detection.
    if (message?.type === 'PROCESS_AUDIO') {
        const { samples, sampleRate, targetTabId } = message;
        processAudioChunk(samples, sampleRate, targetTabId);
        return false;
    }

    return false;
});

// ===========================================================================
// Audio Pipeline — Mel-Spectrogram + Voice-Clone Inference
// ===========================================================================

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const AUDIO_SAMPLE_RATE   = 16000;  // target sample rate (model trained at 16kHz)
const MEL_BINS            = 128;    // number of mel-filter banks
const MEL_FRAMES          = 128;    // number of time frames in the spectrogram
const FFT_SIZE            = 512;    // samples per FFT window (32ms @ 16kHz)
const HOP_LENGTH          = Math.floor(AUDIO_SAMPLE_RATE * 3 / MEL_FRAMES); // ~375 samples

/**
 * Generates a Hann window of the given length.
 * @param {number} length
 * @returns {Float32Array}
 */
function hannWindow(length) {
    const win = new Float32Array(length);
    for (let i = 0; i < length; i++) {
        win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (length - 1)));
    }
    return win;
}

/** Pre-built Hann window (reused across calls). */
const HANN_WIN = hannWindow(FFT_SIZE);

/**
 * Minimal in-place DFT for real-valued input — returns magnitude spectrum.
 * Only computes the first (FFT_SIZE/2 + 1) bins (positive frequencies).
 *
 * Performance note: this is an O(N²) DFT, adequate for FFT_SIZE=512 at 1 chunk/3s.
 * Replace with an FFT library if inference rate increases.
 *
 * @param {Float32Array} frame  Windowed PCM frame, length = FFT_SIZE
 * @returns {Float32Array}      Magnitude spectrum, length = FFT_SIZE/2 + 1
 */
function rfftMagnitude(frame) {
    const N    = frame.length;
    const half = Math.floor(N / 2) + 1;
    const mag  = new Float32Array(half);

    for (let k = 0; k < half; k++) {
        let re = 0, im = 0;
        const twoPiKoverN = (2 * Math.PI * k) / N;
        for (let n = 0; n < N; n++) {
            re += frame[n] * Math.cos(twoPiKoverN * n);
            im -= frame[n] * Math.sin(twoPiKoverN * n);
        }
        mag[k] = Math.sqrt(re * re + im * im);
    }
    return mag;
}

/**
 * Builds a triangular mel-filter bank matrix.
 * Returns a Float32Array of shape [MEL_BINS × (FFT_SIZE/2+1)] stored row-major.
 *
 * @param {number} sampleRate
 * @returns {Float32Array}
 */
function buildMelFilterbank(sampleRate) {
    const numFreqBins = Math.floor(FFT_SIZE / 2) + 1;

    // Convert Hz to mel and back.
    const hzToMel = hz => 2595 * Math.log10(1 + hz / 700);
    const melToHz = mel => 700 * (Math.pow(10, mel / 2595) - 1);

    const melMin = hzToMel(0);
    const melMax = hzToMel(sampleRate / 2);

    // (MEL_BINS + 2) evenly-spaced mel points.
    const melPoints = new Float32Array(MEL_BINS + 2);
    for (let i = 0; i < MEL_BINS + 2; i++) {
        melPoints[i] = melToHz(melMin + (i / (MEL_BINS + 1)) * (melMax - melMin));
    }

    // Convert mel-centre frequencies to FFT bin indices.
    const freqBins = melPoints.map(f => Math.floor((FFT_SIZE + 1) * f / sampleRate));

    const filterbank = new Float32Array(MEL_BINS * numFreqBins); // row-major [mel, freq]

    for (let m = 1; m <= MEL_BINS; m++) {
        const lo  = freqBins[m - 1];
        const mid = freqBins[m];
        const hi  = freqBins[m + 1];

        for (let k = lo; k < mid && k < numFreqBins; k++) {
            if (mid !== lo) filterbank[(m - 1) * numFreqBins + k] = (k - lo) / (mid - lo);
        }
        for (let k = mid; k <= hi && k < numFreqBins; k++) {
            if (hi !== mid) filterbank[(m - 1) * numFreqBins + k] = (hi - k) / (hi - mid);
        }
    }

    return filterbank;
}

/** Cached mel filterbank (built once, reused across audio chunks). */
let _melFilterbank = null;
function getMelFilterbank(sampleRate) {
    if (!_melFilterbank) _melFilterbank = buildMelFilterbank(sampleRate);
    return _melFilterbank;
}

/**
 * Linearly resamples a Float32Array from `srcRate` to `dstRate`.
 * Simple linear interpolation — adequate for 44100→16000 conversion.
 *
 * @param {Float32Array} samples
 * @param {number} srcRate
 * @param {number} dstRate
 * @returns {Float32Array}
 */
function resample(samples, srcRate, dstRate) {
    if (srcRate === dstRate) return samples;
    const ratio      = srcRate / dstRate;
    const outLen     = Math.floor(samples.length / ratio);
    const resampled  = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
        const pos  = i * ratio;
        const idx  = Math.floor(pos);
        const frac = pos - idx;
        const a    = samples[idx]     ?? 0;
        const b    = samples[idx + 1] ?? 0;
        resampled[i] = a + frac * (b - a);
    }
    return resampled;
}

/**
 * Computes a log-mel spectrogram from raw PCM samples.
 *
 * Output shape: [MEL_BINS × MEL_FRAMES] stored row-major (mel first),
 * then reshaped into [1, 1, MEL_BINS, MEL_FRAMES] for the model.
 *
 * @param {Float32Array} pcm        Raw PCM at AUDIO_SAMPLE_RATE
 * @returns {Float32Array}          Normalised log-mel values, length = MEL_BINS × MEL_FRAMES
 */
function computeLogMelSpectrogram(pcm) {
    const numFreqBins = Math.floor(FFT_SIZE / 2) + 1;
    const filterbank  = getMelFilterbank(AUDIO_SAMPLE_RATE);

    // Output buffer [MEL_BINS × MEL_FRAMES]
    const spec = new Float32Array(MEL_BINS * MEL_FRAMES);
    let minVal = Infinity, maxVal = -Infinity;

    for (let t = 0; t < MEL_FRAMES; t++) {
        const start = t * HOP_LENGTH;

        // Extract and window the frame.
        const frame = new Float32Array(FFT_SIZE);
        for (let i = 0; i < FFT_SIZE; i++) {
            const s = start + i < pcm.length ? pcm[start + i] : 0;
            frame[i] = s * HANN_WIN[i];
        }

        // Magnitude spectrum.
        const mag = rfftMagnitude(frame);

        // Apply mel filterbank and log-compress.
        for (let m = 0; m < MEL_BINS; m++) {
            let energy = 0;
            const rowOffset = m * numFreqBins;
            for (let k = 0; k < numFreqBins; k++) {
                energy += filterbank[rowOffset + k] * mag[k];
            }
            const logEnergy = Math.log(Math.max(energy, 1e-9));
            spec[m * MEL_FRAMES + t] = logEnergy;
            if (logEnergy < minVal) minVal = logEnergy;
            if (logEnergy > maxVal) maxVal = logEnergy;
        }
    }

    // Normalize to [0, 1] across the full spectrogram.
    const range = maxVal - minVal || 1;
    for (let i = 0; i < spec.length; i++) {
        spec[i] = (spec[i] - minVal) / range;
    }

    return spec;
}

/**
 * Full audio pipeline: resample → log-mel spectrogram → ONNX inference → dispose tensors.
 *
 * @param {number[]} rawSamples   PCM samples as a plain Array (transferred from content.js)
 * @param {number}   srcRate      Sample rate of `rawSamples` (e.g. 44100, 48000)
 * @param {number}   targetTabId
 */
async function processAudioChunk(rawSamples, srcRate, targetTabId) {
    let inputTensor  = null;
    let outputMap    = null;

    try {
        if (!audioSession) {
            console.warn('[offscreen] Audio session not ready; attempting to initialize now.');
            await initializeAudioSession();
            if (!audioSession) {
                console.warn('[offscreen] Audio model unavailable — notifying UI.');
                // Broadcast immediately so the popup stops showing "Scanning…"
                // instead of waiting indefinitely for a result that will never come.
                chrome.runtime.sendMessage({
                    type:        'AUDIO_INFERENCE_RESULT',
                    audioStatus: 'no_model',
                    targetTabId
                });
                return;
            }
        }

        // 1. Convert plain Array → Float32Array and resample to 16kHz.
        const pcmF32    = new Float32Array(rawSamples);
        const pcm16k    = resample(pcmF32, srcRate, AUDIO_SAMPLE_RATE);

        // 2. Compute log-mel spectrogram  → Float32Array of length MEL_BINS × MEL_FRAMES
        const specData  = computeLogMelSpectrogram(pcm16k);

        // 3. Wrap in ONNX tensor [batch=1, channels=1, mel=128, frames=128]
        inputTensor = new ort.Tensor('float32', specData, [1, 1, MEL_BINS, MEL_FRAMES]);

        const inputName  = audioSession.inputNames[0]  ?? 'input';
        outputMap        = await audioSession.run({ [inputName]: inputTensor });

        const outputName = audioSession.outputNames[0] ?? Object.keys(outputMap)[0];
        const audioScore = extractConfidence(outputMap[outputName]);

        // 4. Send result back to background.
        chrome.runtime.sendMessage({
            type:          'AUDIO_INFERENCE_RESULT',
            audioConfidence: audioScore,
            targetTabId
        });

        console.log('[offscreen] Audio inference complete. Score:', audioScore);
    } catch (err) {
        console.error('[offscreen] Error during audio processing:', err);
    } finally {
        // Dispose WASM memory — MUST mirror the video pipeline's finally block.
        try {
            if (inputTensor && typeof inputTensor.dispose === 'function') inputTensor.dispose();
            if (outputMap) {
                for (const key of Object.keys(outputMap)) {
                    const t = outputMap[key];
                    if (t && typeof t.dispose === 'function') t.dispose();
                }
            }
        } catch (disposeErr) {
            console.error('[offscreen] Error disposing audio tensors:', disposeErr);
        }
    }
}