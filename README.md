# DAM-RFRNet

Dual-Attention and Multi-Scale Residual Feature Refinement Network for Single Image Dehazing.

**Authors:** Dr. L. V. Rajani Kumari, Hanwitha Bandi, S. Rishi Nandhan, and P. Angel  
**Affiliation:** Department of Electronics and Communication Engineering (ECE), VNR Vignana Jyothi Institute of Engineering and Technology (VNRVJIET), Hyderabad, India.  
**Contact:** `rajanikumari_lv@vnrvjiet.in`

---

## Overview

Single-image dehazing aims to recover clear scene radiance from images degraded by atmospheric fog, haze, and smoke.

Traditional methods such as the Dark Channel Prior (DCP) rely on statistical assumptions that struggle in bright sky regions and white surfaces. Deep learning models often estimate the transmission map and global atmospheric light separately, which can introduce compounding estimation errors.

**DAM-RFRNet** reformulates Koschmieder's atmospheric scattering model into a unified coefficient parameter $K(x)$, allowing direct scene radiance recovery without separate airlight estimation. The network integrates two main components:
* **Residual Feature Refinement Block (RFRB):** Combines parallel standard convolutions ($3\times3$, $d=1$) and dilated convolutions ($3\times3$, $d=2$) with $1\times1$ feature fusion and identity skip connections to recover high-frequency textures and edge details.
* **Dual Spatial-Channel Attention (DSCA):** Couples Squeeze-and-Excitation channel attention with a spatial attention module to adaptively focus on regions with varying fog density.

---

## Model Architecture

The architecture consists of the following pipeline:

1. **Downsampled Context Branch:** The input image $I(x)$ is bilinearly downsampled to $64\times64$ and passed through a compact module to predict a coarse map $K_{\text{down}}$, generating a coarse preview.
2. **Feature Extraction:** The coarse estimate is upsampled back to full resolution, and a $3\times3$ convolution extracts contextual features $F_{\text{feat}}$.
3. **Multi-Scale Integration:** The original image $I(x)$ and $F_{\text{feat}}$ are concatenated along the channel dimension to form a 6-channel feature map.
4. **Residual Refinement (RFRB):** Features pass through parallel standard and dilated convolutional branches, merged via $1\times1$ convolution with a residual skip connection.
5. **Dual Attention (DSCA):** Features are recalibrated using channel-wise pooling and spatial mask gating to focus on dense haze regions.
6. **Physical Inversion:** The main $K$-module estimates transformation map $K(x)$, reconstructing clear radiance via:
   $$J(x) = \text{clip}(K(x) \cdot I(x) - K(x) + 1.0, 0.0, 1.0)$$

---

## Benchmark Results

Evaluation results across 800 unseen test image pairs from the 4,000 paired dataset:

| Model / Configuration | Avg. PSNR (dB) | Avg. SSIM | Latency (ms) | Throughput (FPS) | Reference / Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Untreated Hazy Input** | 15.63 dB | 0.7591 | — | — | Unprocessed hazy input |
| **Traditional DCP** | 20.31 dB | 0.8179 | 148.50 ms | ~6.7 FPS | He et al., IEEE TPAMI 2011 |
| **Baseline Backbone (M0: No Novel Blocks)** | 20.24 dB | 0.8439 | 27.32 ms | ~36.6 FPS | Ablation backbone without RFRB/DSCA |
| **Proposed DAM-RFRNet (Full Model)** | **20.46 dB** | **0.8462** | 31.76 ms | ~31.5 FPS | Proposed architecture (RFRB + DSCA) |

### Performance Summary
* **Ablation Comparison:** Adding the RFRB and DSCA modules yields an improvement of **+0.22 dB PSNR** and **+0.0023 SSIM** over the baseline backbone (M0), while operating at over 31 FPS.
* **Comparison with DCP:** DAM-RFRNet achieves higher structural similarity (0.8462 vs 0.8179) and runs approximately $4.7\times$ faster than the traditional Dark Channel Prior.
* **Overall Recovery:** Compared to raw hazy inputs, DAM-RFRNet improves average PSNR by **+4.83 dB** and SSIM by **+0.0871**.

---

## Project Structure

```
DAM-RFRNet/
├── app.py                         # Streamlit web interface
├── run_major_project.py           # Desktop application (Tkinter)
├── main.py                        # Image and video inference script
├── manual_infer.py                # Command-line single image inference
├── run_minor_project.bat          # Desktop launcher
│
├── model.py                       # Network definitions (DAM-RFRNet, DSCA, RFRB, AODNet)
├── model_utils.py                 # PSNR, SSIM metrics and helpers
├── dataset.py                     # PyTorch dataset loader
├── detection_module.py            # Downstream object detection evaluation
├── video_processor.py             # Temporal video defogging engine
│
├── train.py                       # Training script
├── evaluate_test_set.py           # Evaluation on 800 test pairs
├── compare_all_3_models.py        # 3-model comparative evaluation
│
├── checkpoints/                   # Trained model weights
├── figures/                       # Qualitative figures and architecture diagrams
├── outputs/                       # Evaluation logs and test outputs
├── paper/
│   ├── main.tex                   # IEEEtran conference LaTeX manuscript
│   ├── DAM_RFRNet_Paper.pdf       # Compiled conference paper PDF
│   └── DAM_RFRNet_Research_Documentation.md
├── requirements.txt               # Dependencies
├── CITATION.cff                   # Citation metadata
└── LICENSE                        # MIT License
```

---

## Instructions to Run

### 1. Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Streamlit Web Interface
Run the web application:
```bash
streamlit run app.py
```

### 3. Desktop Application
Run the desktop GUI:
```bash
python run_major_project.py
```

### 4. Single Image Inference
Run inference on a single image:
```bash
python manual_infer.py --image example_image_1.png --checkpoint checkpoints/dam_rfrnet_best.pth
```

### 5. Benchmark Evaluation
Evaluate the model on the test dataset:
```bash
python evaluate_test_set.py
```

---

## References

1. K. He, J. Sun, and X. Tang, "Single image haze removal using dark channel prior," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 33, no. 12, pp. 2341–2353, 2011.
2. B. Li, X. Peng, Z. Wang, J. Chai, and D. Tao, "AOD-Net: All-in-one dehazing network," in *IEEE International Conference on Computer Vision (ICCV)*, 2017, pp. 4770–4778.
3. J. Hu, L. Shen, and G. Sun, "Squeeze-and-excitation networks," in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 7132–7141.
4. S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional block attention module," in *European Conference on Computer Vision (ECCV)*, 2018, pp. 3–19.
5. F. Yu and V. Koltun, "Multi-scale context aggregation by dilated convolutions," in *International Conference on Learning Representations (ICLR)*, 2016.

---

## Citation

```bibtex
@inproceedings{kumari2026damrfrnet,
  author    = {Rajani Kumari, L. V. and Bandi, Hanwitha and Rishi Nandhan, S. and Angel, P.},
  title     = {Dual-Attention and Multi-Scale Residual Feature Refinement Network (DAM-RFRNet) for High-Fidelity Single Image Dehazing and Defogging},
  booktitle = {IEEE International Conference on Image Processing and Vision Systems},
  year      = {2026}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
