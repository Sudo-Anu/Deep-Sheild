import torch
import torch.nn as nn
import os

class LiteAudioCNN(nn.Module):
    def __init__(self):
        super(LiteAudioCNN, self).__init__()
        # Input: [1, 1, 128, 128] (B, C, H, W) -> Batch, Channels, Mel-bins, Time-frames
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = LiteAudioCNN()
model.eval()

# Fixed input shape exactly matching offscreen.js: [1, 1, 128, 128]
dummy_input = torch.randn(1, 1, 128, 128)

onnx_path = r"c:\Users\Aniruddha Raut\Documents\Projects\SIH2026\extension-root\assets\audio_clone_detector.onnx"

# We remove dynamic_axes and set opset_version=17 to avoid external data splitting
# and buggy onnx fallback converters in newer PyTorch versions.
torch.onnx.export(model, dummy_input, onnx_path,
                  export_params=True,
                  opset_version=17,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'])

print(f"Model exported successfully to {onnx_path}")
