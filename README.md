# 🛡️ DeepShield

### Privacy-First, On-Device Deepfake Detection for the Web

**DeepShield** is a Chrome extension designed to detect **AI-generated and manipulated video and audio directly on the user's device** without requiring API keys, cloud inference, or uploading media to an external server.

The project focuses on one simple idea:

> **Your media should stay on your device.**

DeepShield uses locally bundled **ONNX (Open Neural Network Exchange)** models with **ONNX Runtime Web** to perform inference inside the browser.

---

## ✨ Features

* 🎥 **Video Deepfake Detection**
* 🎙️ **AI Voice / Audio Clone Detection**
* 🔒 **Privacy-first, on-device inference**
* 🚫 **No API keys required**
* ☁️ **No media uploads to external servers**
* ⚡ **Hardware acceleration through WebGPU / WebAssembly where available**
* 🧠 **ONNX-based machine learning inference**
* 📊 **Separate video and audio confidence scores**
* 🚦 **Real-time traffic-light risk indicator**
* 🧹 **Explicit tensor disposal to reduce memory leaks**
* 🧩 **Chrome Manifest V3 architecture**
* 🔄 **Combined audio + video risk assessment**

---

## 🎯 Why DeepShield?

Deepfake detection services commonly rely on cloud APIs.

That creates several problems:

* Media must leave the user's device.
* API keys or paid services may be required.
* Uploading private conversations or videos creates privacy concerns.
* Network latency affects real-time detection.
* Availability depends on an external service.

DeepShield takes a different approach.

### Traditional approach

```text
User Media
    ↓
Browser
    ↓
Cloud API
    ↓
Remote ML Model
    ↓
Detection Result
```

### DeepShield approach

```text
User Media
    ↓
Browser
    ↓
Local Processing
    ↓
ONNX Runtime Web
    ↓
Local ML Model
    ↓
Detection Result
```

The media does not need to be sent to a remote inference server.

---

# 🧠 How It Works

DeepShield analyzes both **visual** and **audio** characteristics of media.

## 🎥 Video Pipeline

The extension detects video elements on the current webpage and periodically captures frames.

```text
Webpage
   │
   ▼
content.js
   │
   │ Find <video>
   │ Capture frames
   │ Downsample → 1 FPS
   ▼
background.js
   │
   │ Message Bridge
   ▼
Offscreen Document
   │
   ▼
offscreen.js
   │
   ├── ONNX Runtime Web
   │
   └── EfficientNet-B0 Model
   │
   ▼
Video Confidence Score
   │
   ▼
background.js
   │
   ▼
content.js
   │
   ▼
Traffic-Light UI
```

The current architecture uses an **EfficientNet-B0-based ONNX model** for video analysis.

---

## 🎙️ Audio Pipeline

DeepShield also analyzes audio to identify potential AI-generated or cloned voices.

The browser captures audio from the video and processes it locally.

```text
Video Audio
    │
    ▼
content.js
    │
    ├── AudioContext
    ├── ScriptProcessorNode
    └── 3-second PCM buffer
    │
    ▼
background.js
    │
    ▼
offscreen.js
    │
    ├── Resample → 16 kHz
    │
    ├── Log-Mel Spectrogram
    │
    └── [1, 1, 128, 128] Tensor
    │
    ▼
Audio ONNX Model
    │
    ▼
Softmax
    │
    ├── Real
    └── Fake
    │
    ▼
Audio Confidence
```

The audio pipeline converts the raw audio into a **Log-Mel Spectrogram**, which is then passed to the local ONNX model.

---

# 🔀 Combined Detection

Instead of relying exclusively on video or audio, DeepShield evaluates both signals.

```text
              ┌───────────────┐
              │ Video Model   │
              └───────┬───────┘
                      │
                Video Score
                      │
                      ▼
                 ┌─────────┐
                 │         │
                 │  Merge  │
                 │         │
                 └────┬────┘
                      ▲
                      │
                Audio Score
                      │
              ┌───────┴───────┐
              │ Audio Model   │
              └───────────────┘
```

The current implementation uses a **worst-case merge**, taking the minimum confidence value between the video and audio signals:

```text
mergedScore = min(videoScore, audioScore)
```

The resulting score is sent back to the webpage UI, while the popup can display the individual video and audio scores.

---

# 🔐 Privacy

Privacy is one of the primary design goals of DeepShield.

### DeepShield does NOT require:

* ❌ API keys
* ❌ Cloud inference
* ❌ Media uploads
* ❌ Third-party detection servers
* ❌ User accounts

### Processing happens locally:

```text
Media
  ↓
Browser
  ↓
Local JavaScript
  ↓
ONNX Runtime Web
  ↓
Local ONNX Model
  ↓
Result
```

This makes DeepShield particularly useful for analyzing sensitive media where uploading the content to an external service would be undesirable.

---

# ⚡ Performance

DeepShield uses **ONNX Runtime Web** to execute machine-learning models inside the browser.

Depending on the user's hardware and browser capabilities, the runtime can make use of technologies such as:

* **WebGPU** — GPU acceleration through the browser
* **WebAssembly (WASM)** — native-like execution inside the browser
* **SIMD** — Single Instruction, Multiple Data acceleration
* **Multi-threaded WebAssembly** — parallel CPU execution

The extension bundles its ONNX Runtime and WebAssembly assets locally to avoid depending on external runtime downloads.

---

# 🧹 Memory Management

Running ML inference continuously inside a browser can create significant memory pressure.

DeepShield therefore explicitly disposes of tensors after inference:

```javascript
tensor.dispose();
```

This is important because tensors can contain large numerical buffers, and failing to release them during repeated inference can cause memory usage to continuously increase.

The architecture specifically includes tensor disposal as part of the inference pipeline.

---

# 🏗️ Architecture

DeepShield is implemented as a **Chrome Manifest V3 extension**.

```text
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
    ├── deepfake_efficientnet_b0_int8.onnx
    ├── audio_clone_detector.onnx
    ├── icon-16.png
    ├── icon-48.png
    └── icon-128.png
```

The repository currently contains the extension implementation, local runtime assets, model assets, and an architecture document describing the data flow.

---

# 📂 Component Responsibilities

| Component                            | Responsibility                             |
| ------------------------------------ | ------------------------------------------ |
| `manifest.json`                      | Chrome extension configuration             |
| `background.js`                      | Central message bridge                     |
| `content.js`                         | Detects video and captures media data      |
| `content.css`                        | Traffic-light detection UI                 |
| `offscreen.html`                     | Hosts the background inference environment |
| `offscreen.js`                       | Executes local ML inference                |
| `popup.html`                         | Extension popup interface                  |
| `popup.js`                           | Displays detection information             |
| `popup.css`                          | Popup styling                              |
| `ort.min.js`                         | ONNX Runtime Web                           |
| `*.wasm`                             | WebAssembly execution backend              |
| `deepfake_efficientnet_b0_int8.onnx` | Video deepfake model                       |
| `audio_clone_detector.onnx`          | Audio / voice-clone model                  |

The repository's architecture documentation describes these components and their relationships in detail.

---

# 🔄 Communication Architecture

Chrome Manifest V3 imposes restrictions on long-running extension pages.

DeepShield therefore separates responsibilities between the content script, background service worker, and offscreen document.

```text
                 ┌──────────────────┐
                 │     Webpage      │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │    content.js    │
                 │ Media Capture    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │   background.js  │
                 │ Message Bridge   │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │  offscreen.html  │
                 │  offscreen.js    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ ONNX Runtime Web │
                 └────────┬─────────┘
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
          Video Model          Audio Model
                │                   │
                └─────────┬─────────┘
                          ▼
                    Risk Scores
                          │
                          ▼
                 ┌──────────────────┐
                 │   background.js  │
                 └────────┬─────────┘
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
             Webpage UI        Extension Popup
```

---

# 🧰 Tech Stack

### Browser

* Google Chrome
* Chrome Extension Manifest V3

### Programming Languages

* **JavaScript** — extension logic, media processing and inference pipeline
* **HTML** — extension interfaces
* **CSS** — user interface
* **Python** — model preparation / generation utilities

### Machine Learning

* ONNX
* ONNX Runtime Web
* EfficientNet-B0
* Audio classification model
* INT8 quantization for the video model

### Browser Compute

* WebGPU
* WebAssembly
* SIMD
* Multi-threaded WASM

### Audio Processing

* Web Audio API
* PCM audio
* 16 kHz resampling
* Log-Mel Spectrogram

---

# 🚀 Installation

DeepShield is currently intended to be installed as an **unpacked Chrome extension**.

### 1. Clone the repository

```bash
git clone https://github.com/Sudo-Anu/Deep-Sheild.git
cd Deep-Sheild
```

### 2. Open Chrome Extensions

Navigate to:

```text
chrome://extensions/
```

### 3. Enable Developer Mode

Enable **Developer mode** in the top-right corner.

### 4. Load the extension

Click:

```text
Load unpacked
```

Select:

```text
Deep-Sheild/extension-root
```

Chrome should now load DeepShield as an unpacked extension.

---

# 🧪 Usage

1. Open a webpage containing a video.
2. Load the DeepShield extension.
3. Start playing the video.
4. DeepShield captures video frames and available audio.
5. The local models process the media.
6. The extension calculates confidence scores.
7. The webpage displays the traffic-light result.

### Result indicators

🟢 **Green**
Lower detected deepfake probability.

🟡 **Yellow**
Uncertain / suspicious result.

🔴 **Red**
Higher detected deepfake probability.

The popup also provides separate video and audio confidence information.

---

# ⚠️ Limitations

DeepShield is an experimental detection system and should **not be treated as an absolute source of truth**.

Deepfake detection models can produce:

* False positives
* False negatives
* Uncertain results
* Poor performance on previously unseen manipulation techniques

Detection confidence should therefore be interpreted as a **risk signal**, not definitive proof that media is genuine or fake.

Performance can also vary depending on:

* Browser implementation
* CPU/GPU capabilities
* Video resolution
* Audio quality
* Compression
* Model accuracy
* Available hardware acceleration

---

# 🛣️ Roadmap

Potential future improvements include:

* [ ] Better temporal video analysis
* [ ] Improved audio deepfake detection
* [ ] More robust model ensembles
* [ ] Better confidence calibration
* [ ] Support for additional Chromium-based browsers
* [ ] Improved WebGPU acceleration
* [ ] More efficient audio processing
* [ ] Automatic model updates
* [ ] Detection history
* [ ] Detailed forensic explanations
* [ ] More granular risk categories
* [ ] Improved memory/performance profiling

---

# 🤝 Contributing

Contributions are welcome.

If you want to improve DeepShield:

```bash
git clone https://github.com/Sudo-Anu/Deep-Sheild.git
cd Deep-Sheild
```

Then create a branch:

```bash
git checkout -b feature/your-feature
```

Make your changes, test the extension locally, and submit a pull request.

Ideas for contributions include:

* Model optimization
* Browser compatibility
* Audio processing improvements
* UI improvements
* Memory optimization
* Detection accuracy
* Security hardening
* Documentation

---

# 🔒 Security & Privacy

If you discover a security issue, please avoid publicly exposing sensitive details before the issue can be addressed.

DeepShield is designed around local processing, but browser extensions still operate within the security model and permissions provided by the browser.

Always review extension permissions and source code before deploying experimental security software.

---

# 📜 License

See the repository for the current licensing information.

---

# 👨‍💻 Author

**Sudo-Anu**

Cybersecurity Engineering Student & Developer

GitHub:
https://github.com/Sudo-Anu

---

## ⭐ Support the Project

If you find DeepShield interesting or useful, consider giving the repository a ⭐ on GitHub.

Every star helps the project get more visibility and motivates further development.

---

> **DeepShield — Detect deepfakes locally. Keep your media private.**
