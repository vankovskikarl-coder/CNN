# CNN from scratch in NumPy + live handwritten digit recognizer

A convolutional neural network for MNIST implemented **entirely from scratch using only
NumPy** — no PyTorch, no TensorFlow, no autograd. Every operation is derived and coded
by hand: forward and backward convolutions, max-pooling / unpooling, ReLU, softmax +
cross-entropy, mini-batch SGD. Even the MNIST binary files are parsed manually with `struct`.

The repo also includes a **Pygame drawing app**: draw a digit (or a two-digit number)
with your mouse and the trained network recognizes it live.

**Test accuracy: 98.94%** — only 106 errors out of 10,000 MNIST test images.

<!-- Optional: record a short screen capture, save it as demo.gif and uncomment:
![demo](demo.gif)
-->

## What is implemented manually

- **Convolution (forward)** — im2col via `sliding_window_view` + matrix multiplication, zero "same" padding
- **Convolution (backward)** — gradients w.r.t. kernels and inputs; input gradients computed as convolutions with flipped kernels
- **Max-pooling (forward + backward)** — 2×2 / stride 2, gradients routed through stored argmax positions (unpooling)
- **Fully connected layers, ReLU, numerically stable softmax, cross-entropy gradient**
- **Mini-batch SGD** with per-layer uniform weight initialization
- **MNIST loading** — raw IDX binary format parsed with `struct`, no dataset libraries

## Architecture

    Input 28×28×1
    Conv 3×3, 32 filters (same padding) → ReLU
    MaxPool 2×2                         → 14×14×32
    Conv 3×3, 64 filters (same padding) → ReLU
    MaxPool 2×2                         → 7×7×64
    Flatten                             → 3136
    FC 3136 → 512 → ReLU
    FC 512 → 512 → ReLU
    FC 512 → 10 → Softmax

~1.89M parameters · cross-entropy loss · SGD (batch 32, lr 0.0133, 30,000 steps)

## Quick start — drawing app

Pretrained weights are included in the repo, no training needed.

```bash
git clone https://github.com/RealBigDick/CNN.git
cd CNN
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python draw.py
```

- Draw with the **left mouse button** — one digit, or two digits side by side (0–99)
- The prediction updates live on the right panel
- **ESC** clears the canvas

How it works: the canvas is a 28×56 grid → digits are separated by empty-column gaps →
each digit is cropped, centered and padded to 28×28 (MNIST-style preprocessing) →
forward pass → predicted class.

## Retrain from scratch (optional)

```bash
python download_mnist.py    # downloads MNIST (~12 MB) and unpacks it
python CNN_train.py         # pure NumPy, slow on CPU; overwrites the .npy weights
```

`download_mnist.py` fetches the four IDX files from the Google CVDF mirror and saves
them under the names the trainer expects:

| File | Content |
|---|---|
| `mnistrain` | 60,000 training images |
| `mnilabel` | 60,000 training labels |
| `10k_test_images` | 10,000 test images |
| `10k_test_labels` | 10,000 test labels |

You can also place the uncompressed IDX files next to `CNN_train.py` manually under
these names. After training, the script prints the number of test errors and saves all
weights (`W_*.npy`, `B_*.npy`, `cores_*.npy`) — `draw.py` picks them up automatically.

## Repository layout

    draw.py            — Pygame drawing app (inference only)
    CNN_train.py       — training loop with manual backprop, saves weights
    download_mnist.py  — MNIST download helper
    *.npy              — pretrained weights (~15 MB)
    requirements.txt   — numpy, pygame

## Requirements

- Python 3.9+
- `numpy >= 1.20` (training + inference)
- `pygame >= 2.5` (drawing app only)
