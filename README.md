# AI Cover Deepfake Audio Detection 🎙️🔒

[![Journal](https://img.shields.io/badge/Journal-JANT%20(2025)-orange.svg)](http://dx.doi.org/10.12673/jant.2025.29.6.816)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-flat&logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

Official implementation of the paper: **"Consideration for CNN-Based Detection of AI Cover Deepfake Audio Leveraging Voice Conversion Vulnerabilities"** (*Journal of Advanced Navigation Technology*, 2025)[cite: 2].

This framework targets structural loopholes in state-of-the-art Singing Voice Synthesis (SVS) and Singing Voice Conversion (SVC) models (e.g., DiffSinger, XiaoiceSing) to robustly monitor and detect unauthorized synthetic music and voice clones[cite: 2].

---

## 📌 Project Overview
The rise of generative AI allows users to easily replicate artist vocals for unauthorized "AI Cover Songs"[cite: 2]. This repository implements a robust deep learning network that exploits specific, hardware/algorithmic **vulnerabilities in VC models**—such as artificial pitch-shift artifacts, fine-grained expressive limitations, and micro-noise contamination[cite: 2].

### Key Features & VC Vulnerability Exploitation
1.  **Vocal-Only Analysis:** Incorporates AI-driven source separation to isolate raw singing tracks, discarding instrumental biases[cite: 2].
2.  **Vulnerability-Targeted Augmentation:** Intentionally applies cross-gender pitch-shifting and micro-Gaussian noise injection to maximize the model's sensitivity to synthetic anomalies[cite: 2].
3.  **Multi-Domain Feature Embedding Fusion:** Extracts 3 distinct embedding domains—capturing **Acoustic Nuances (Prosody/Pitch)**, **Contextual Pronunciation (Articulation)**, and **Speaker Timbre** concurrently[cite: 2].

---

## 🛠️ System Architecture

The pipeline processes input tracks by isolating vocal stems, applying targeted signal augmentations, projecting them into specialized neural encoders, and feeding the concatenated tensors into a hierarchical CNN classifier[cite: 2].

### 1. Data Preprocessing & Targeted Augmentation Pipeline
- **Audio 2Stem Split:** Uses `Demucs` to split audio vectors into clean Vocal and Accompaniment stems, mitigating backing track interference[cite: 2].
- **Gaussian Noise Injection:** Randomly infuses fine $25\text{–}40\text{ dB}$ SNR Gaussian noise into real data to mimic extraction noise artifacts[cite: 2].
- **Cross-Gender Pitch Shifting:** Shifts female-to-male ($\text{Semitone } +6 \sim +7$) and male-to-female ($\text{Semitone } -6 \sim -7$) registers to isolate breakdown behaviors in SVC alignment lattices[cite: 2].

<p align="center">
  <img src="Info/DeepfakeAudio_DataProcessing.png" width="70%" alt="Data Preprocessing Pipeline" />
</p>

### 2. Multi-Domain Feature & Vector Embedding Extractors
Rather than relying on basic frame-level data, the system converts raw tracks into high-dimensional embedding maps across three core properties[cite: 2]:

*   **Prosody & Pitch Stability Analysis (ECAPA-TDNN):** Extracts **Log Mel-Spectrograms** through an emphasized channel attention time-delay network to trap $F_0$ jitter and dynamic instability[cite: 2].
*   **Contextual Articulation Tracker (GRO Encoder):** Processes Linear Frequency Cepstral Coefficients (**LFCC**) through a Transformer Encoder stack to spot misaligned phoneme/syllable gaps[cite: 2].
*   **Timbre Distortion Detector (Resemblyzer):** Converts Extended Constant-Q Cepstral Coefficients (**eCQCC**) through a 3-Layer Bi-LSTM with exponential compression, accentuating micro-spectral variances[cite: 2].

<p align="center">
  <img src="Info/DeepfakeAudio_ModelArchitecture.png" width="60%" alt="Model Architecture" />
</p>

---

## 🔬 Core Classification Network
The compiled multi-view input tensor $(BatchSize, 3, EmbeddingDim)$ is parsed by a dedicated Convolutional Neural Network (CNN)[cite: 2]:
- **Initial Stage:** $7\times7\text{ Convolutional layer (64 Filters, Stride 2)} \rightarrow 3\times3\text{ Max Pooling}$[cite: 2].
- **Feature Extraction:** Progressively deepens channel capacity through repeating $3\times3$ convolutions spanning **64 $\rightarrow$ 128 $\rightarrow$ 256 $\rightarrow$ 512 channels** to evaluate structural correlation maps[cite: 2].
- **Classification Output:** Aggregated via Global Average Pooling into a Fully Connected layer with a Softmax activation step for robust **Real vs Fake** binary validation[cite: 2].

---

## 📊 Evaluation & Baseline Metrics

### 1. Feature Domain Extraction Comparison
Comprehensive comparative framework mapping applied feature components[cite: 2]:

| Method | Extraction Domain / Transform | Principal Application Field |
| :--- | :--- | :--- |
| **Log Mel-Spectrogram** | Mel scale filter bank + Logarithmic compression | Prosodic drift, TTS, and Speech Generation parsing[cite: 2] |
| **LFCC** | Linear frequency scale filter bank | High-frequency noise artifacts & Spoofing detection[cite: 2] |
| **eCQCC** | Constant-Q Transform + Exponential compression | Micro-spectral fluctuations & Voice Authentication[cite: 2] |

### 2. Speaker Verification Baseline Performance
Performance benchmarks of underlying backbone structures evaluated on reference benchmarks (VoxCeleb1 / VoxSRC19)[cite: 2]:

| Model Architecture | Params | VoxCeleb1 EER (%) | VoxCeleb1 MinDCF | VoxSRC19 EER (%) |
| :--- | :---: | :---: | :---: | :---: |
| E-TDNN Baseline | $6.8\text{M}$ | $1.49\%$ | 0.1604 | $1.81\%$ |
| ResNet-34 Baseline | $23.9\text{M}$ | $1.19\%$ | 0.1592 | $1.57\%$ |
| **ECAPA-TDNN (C-1024)** | **$14.7\text{M}$** | **$0.87\%$** | **0.1066** | **$1.22\%$** |

---

## 🚀 Getting Started

### Prerequisites
Install PyTorch alongside the required audio signal toolkits[cite: 2]:
```bash
pip install torch torchvision torchaudio numpy pandas librosa matplotlib demucs
