# Browser Extension — File Structure & Data Flow

extension-root/
│
├── manifest.json
├── background.js
│
├── content/
│   ├── content.js
│   └── content.css
│
├── offscreen/
│   ├── offscreen.html
│   └── offscreen.js
│
├── popup/
│   ├── popup.html
│   ├── popup.css
│   └── popup.js
│
├── lib/
│   ├── ort.min.js
│   ├── ort-wasm-simd-threaded.jsep.mjs
│   ├── ort-wasm-simd-threaded.jsep.wasm
│   ├── ort-wasm-simd-threaded.wasm
│   └── ort-wasm.wasm
│
└── assets/
    ├── deepfake_efficientnet_b0_int8.onnx   ← video deepfake model
    ├── audio_clone_detector.onnx            ← voice-clone detection model [NEW]
    ├── icon-16.png
    ├── icon-48.png
    └── icon-128.png

══════════════════════════════════════════════════════════════════════
                               SUPPORT FILES
══════════════════════════════════════════════════════════════════════

  lib/ort.min.js
       │
       └── loads ──► lib/ort-wasm*.wasm / lib/ort-wasm-simd-threaded*.wasm

  offscreen.js
       │
       ├── uses ──► lib/ort.min.js
       └── loads ─► assets/deepfake_efficientnet_b0_int8.onnx

  content.js
       │
       └── uses ──► content.css

  popup.js
       │
       └── uses ──► popup.css

  manifest.json
       ├── registers ──► background.js
       ├── registers ──► popup/popup.html
       └── registers ──► content/content.js


══════════════════════════════════════════════════════════════════════
                              VIDEO PIPELINE
══════════════════════════════════════════════════════════════════════

                    ┌─────────────────────┐
                    │    manifest.json    │
                    │  Extension Config   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    content.js       │
                    │ Find <video>        │
                    │ Capture frames      │
                    │ Downsample → 1 FPS  │
                    └──────────┬──────────┘
                               │
                         Frame Payload
                               │
                               ▼
                    ┌─────────────────────┐
                    │    background.js    │
                    │ Message Bridge      │
                    └──────────┬──────────┘
                               │
                       Creates / routes to
                               │
                               ▼
                    ┌─────────────────────┐
                    │   offscreen.html    │
                    │ Offscreen Runtime   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    offscreen.js     │
                    │ Video Inference     │
                    │ (session)           │
                    └──────────┬──────────┘
                               │
                  Loads ONNX Runtime + Model
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────────────┐
        │   ort.min.js    │         │ deepfake_...int8.onnx   │
        │ ONNX Runtime    │         │ Vision ML Model         │
        └────────┬────────┘         └────────┬────────────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                    ┌─────────────────────┐
                    │   ONNX Inference    │
                    │ Tensor Processing   │
                    │ tensor.dispose()    │
                    └──────────┬──────────┘
                               │
                        Confidence Score
                               │
                               ▼
                    ┌─────────────────────┐
                    │    background.js    │
                    │ Route Result        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     content.js      │
                    │ Update UI State     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    content.css      │
                    │ Traffic Light UI    │
                    │ 🟢 Green            │
                    │ 🟡 Yellow           │
                    │ 🔴 Red              │
                    └─────────────────────┘

══════════════════════════════════════════════════════════════════════
                              AUDIO PIPELINE  [NEW]
══════════════════════════════════════════════════════════════════════

  content.js detects a <video>, calls video.captureStream() and
  pipes audio through the Web Audio API. Every 3 seconds a PCM chunk
  is sent to background.js → offscreen.js for voice-clone inference.

                    ┌─────────────────────┐
                    │    content.js       │
                    │ video.captureStream │
                    │ AudioContext        │
                    │ ScriptProcessorNode │
                    │ Buffer 3 s PCM      │
                    └──────────┬──────────┘
                               │
                   REQUEST_AUDIO_INFERENCE
                               │
                               ▼
                    ┌─────────────────────┐
                    │    background.js    │
                    │ Forward PROCESS_    │
                    │ AUDIO to offscreen  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    offscreen.js     │
                    │ Audio Inference     │
                    │ (audioSession)      │
                    │ Resample → 16kHz    │
                    │ Log-Mel Spectrogram │
                    │ [1,1,128,128]       │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────────────┐
        │   ort.min.js    │         │ audio_clone_detector    │
        │ ONNX Runtime    │         │ .onnx (Audio ML Model)  │
        └────────┬────────┘         └────────┬────────────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                    ┌─────────────────────┐
                    │   ONNX Inference    │
                    │ softmax([real,fake]) │
                    │ tensor.dispose()    │
                    └──────────┬──────────┘
                               │
                    AUDIO_INFERENCE_RESULT
                               │
                               ▼
                    ┌─────────────────────┐
                    │    background.js    │
                    │ Worst-case merge    │
                    │ min(video, audio)   │
                    └──────────┬──────────┘
                               │
                         UPDATE_UI
                  { confidence, audioConfidence,
                    mergedScore, targetTabId }
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             content.js              popup.js
           Traffic-light         Dual score bars
           (mergedScore)       🎬 Video + 🎙️ Audio
