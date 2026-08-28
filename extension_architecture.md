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
├── lib/
│   ├── ort.min.js
│   └── ort-wasm.wasm
│
└── assets/
    ├── model.onnx
    ├── icon-16.png
    ├── icon-48.png
    └── icon-128.png

══════════════════════════════════════════════════════════════════════
                              SUPPORT FILES
══════════════════════════════════════════════════════════════════════

  lib/ort.min.js
       │
       └── loads ──► lib/ort-wasm.wasm

  offscreen.js
       │
       ├── uses ──► lib/ort.min.js
       └── loads ─► assets/model.onnx

  content.js
       │
       └── uses ──► content.css

  manifest.json
       ├── registers ──► background.js
       └── registers ──► content/content.js


══════════════════════════════════════════════════════════════════════
                         DATA / EXECUTION FLOW
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
                    │ Inference Engine    │
                    └──────────┬──────────┘
                               │
                  Loads ONNX Runtime + Model
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐
        │   ort.min.js    │         │   model.onnx    │
        │ ONNX Runtime    │         │ ML Model        │
        └────────┬────────┘         └────────┬────────┘
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

