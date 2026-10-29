# DAM-RFRNet: Dual-Attention & Multi-Scale Residual Feature Refinement Network

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IEEEtran%20Ready-success.svg)](paper/main.tex)

> **Official Implementation and Research Package**  
> **Paper Title:** *Dual-Attention and Multi-Scale Residual Feature Refinement Network (DAM-RFRNet) for High-Fidelity Single Image Dehazing and Defogging*  
> **Authors:** Dr. L. V. Rajani Kumari, Hanwitha Bandi, S. Rishi Nandhan, P. Angel  
> **Affiliation:** Department of Electronics and Communication Engineering (ECE), VNR Vignana Jyothi Institute of Engineering and Technology (VNRVJIET), Hyderabad, India.

---

## 📌 Overview

Single-image dehazing under non-uniform atmospheric scattering presents fundamental challenges:
1. **Spatial Homogeneity Assumption:** Standard channel-only attention treats all pixels equally, failing to selectively restore spatially dense fog clusters.
2. **High-Frequency Texture Attenuation:** Multiscale downsampling/upsampling introduces textural blurring and structural attenuation along sharp object boundaries.

**DAM-RFRNet** resolves these challenges by introducing:
* **Dual Spatial-Channel Attention Mechanism (DSCA):** Concurrently recalibrates inter-channel dependencies (SE-Block) and isolates spatially heterogeneous fog density (SAM via $7\times7$ convolutions).
* **Residual Feature Refinement Block (RFRB):** Multiscale parallel receptive field branch combining standard $3\times3$ convolutions with dilated convolutions ($\text{rate}=2$), $1\times1$ feature fusion, and an identity skip connection to recover high-frequency textural details.
* **Downstream Autonomous Perception Recovery:** Integrated with YOLOv8/Faster R-CNN to quantitatively verify object detection and confidence score recovery under adverse atmospheric conditions.
* **Temporal Video Defogging Engine:** Video stream dehazing with Exponential Moving Average (EMA) temporal smoothing ($\alpha=0.75$) to eliminate inter-frame luminance flicker.

---

## 🏗️ System Architecture

![DAM-RFRNet System Architecture](figures/fig1_architecture.jpg)

### End-to-End Formulation
The network reformulates the atmospheric scattering physics $I(x) = J(x)t(x) + A(1 - t(x))$ into a unified coefficient matrix $K(x)$:

$$J(x) = K(x) \cdot I(x) - K(x) + b$$

Where $K(x)$ combines both transmission map estimation $t(x)$ and atmospheric light $A$, eliminating error propagation from separate estimation stages.

---

## 📊 Benchmark & Quantitative Results

Evaluated across an 800 unseen test-image benchmark (4,000 image dataset split 80:20):

| Model Configuration | PSNR (dB) ↑ | SSIM ↑ | Latency (ms/img) ↓ | FPS ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **Traditional Dark Channel Prior (DCP)** | 14.28 dB | 0.7312 | 148.50 ms | ~6.7 FPS |
| **Baseline AOD-Net (No Novel Blocks)** | 25.21 dB | 0.8846 | 27.32 ms | ~36.6 FPS |
| **Proposed DAM-RFRNet (Full Architecture)** | **43.30 dB** | **0.9885** | **31.76 ms** | **~31.5 FPS** |
| *Empirical Improvement over Baseline* | *+18.09 dB* | *+0.1039* | — | — |

---

## 📁 Repository Structure

```
DAM-RFRNet/
├── app.py                         # Streamlit interactive web platform
├── run_major_project.py           # Standalone desktop dashboard with Tkinter GUI
├── main.py                        # Unified image & video dehazing pipeline
├── manual_infer.py                # Command-line single image inference
├── run_minor_project.bat          # One-click Windows batch launcher
│
├── model.py                       # DAM-RFRNet, DSCA Attention, RFRB, and AODNet
├── model_utils.py                 # PSNR/SSIM metrics and checkpoint managers
├── dataset.py                     # Custom PyTorch Defogging dataset loaders
├── detection_module.py            # Adverse weather AI object detection evaluator (YOLOv8)
├── video_processor.py             # Temporal EMA video defogging engine
│
├── train.py                       # Hybrid loss (MS-SSIM + L1 + VGG Perceptual) training
├── evaluate_test_set.py           # Comprehensive 800 test-pair benchmark suite
├── compare_all_3_models.py        # 3-way comparative ablation study
│
├── checkpoints/
│   └── dam_rfrnet_best.pth        # Pretrained state dictionary (~83 KB)
├── figures/                       # Publication figures, ablation charts, and architecture diagrams
├── outputs/                       # Quantitative reports, comparison plots, and metric logs
├── paper/
│   ├── main.tex                   # IEEEtran calibrated 6-page LaTeX manuscript
│   ├── DAM_RFRNet_Paper.pdf       # Compiled conference paper PDF
│   └── DAM_RFRNet_Research_Documentation.md
├── requirements.txt               # Environment dependencies
└── .gitignore                     # Git ignore rules for clean repo management
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/<your-username>/DAM-RFRNet.git
cd DAM-RFRNet
pip install -r requirements.txt
```

### 2. Interactive Streamlit Web Platform
Launch the modern dark-themed web platform for real-time image dehazing, parameter tuning, and downstream object detection analysis:
```bash
streamlit run app.py
```

### 3. Desktop Application
Run the desktop GUI with file-picker dialog and 3-stage visual dashboard:
```bash
python run_major_project.py
```

### 4. Single Image CLI Inference
Run fast inference on an individual image:
```bash
python manual_infer.py --image example_image_1.png --checkpoint checkpoints/dam_rfrnet_best.pth
```

### 5. Video Dehazing
Dehaze an input video stream with temporal flicker suppression:
```bash
python -c "from video_processor import VideoDehazeProcessor; p = VideoDehazeProcessor(); p.process_video_file('input.mp4', 'outputs/dehazed.mp4')"
```

---

## 🔬 Training & Evaluation

### Training from Scratch
```bash
python train.py --epochs 40 --batch_size 16 --lr 0.0001 --model_type dam_rfrnet
```

### Quantitative Test Evaluation
Evaluate the checkpoint on 800 unseen test pairs:
```bash
python evaluate_test_set.py
```

### 3-Model Comparative Benchmark
Reproduce the comparison between DAM-RFRNet, Traditional DCP, and Baseline:
```bash
python compare_all_3_models.py
```

---

## 📄 Publication & LaTeX Source

The complete publication-ready manuscript is located in the [`paper/`](paper/) directory:
* **LaTeX Source:** `paper/main.tex` (calibrated for strict 6-page IEEE conference format).
* **Direct Overleaf Upload:** Simply upload `paper/main.tex` together with `figures/` to Overleaf and set compiler to `pdfLaTeX`.
* **Compiled PDF:** `paper/DAM_RFRNet_Paper.pdf`.

---

## ✍️ Citation

If you use this work, codebase, or model architecture in your research, please cite:

```bibtex
@article{rajani2026damrfrnet,
  title={Dual-Attention and Multi-Scale Residual Feature Refinement Network (DAM-RFRNet) for High-Fidelity Single Image Dehazing and Defogging},
  author={Rajani Kumari, L. V. and Bandi, Hanwitha and Rishi Nandhan, S. and Angel, P.},
  institution={Department of Electronics and Communication Engineering, VNR Vignana Jyothi Institute of Engineering and Technology},
  year={2026}
}
```

---

## 📜 License
This project is released under the [MIT License](LICENSE).
