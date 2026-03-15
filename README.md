# ISSR SCREENING TEST SUBMISSION FOR TEAM-COMMUNICATION-PROJECT

## Task 1 — Dataset Identification and Evaluation

Identifies and evaluates open-access audio corpora as proxies for TRIP Lab simulator audio. Three datasets compared — AMI Meeting Corpus, ICSI, LibriCSS — against project-specific criteria.

**Selected dataset:** AMI Meeting Corpus, Session ES2008a
- 4-person role-assigned meeting (PM, designer, marketing, UI)
- Synchronized far-field array + close-talk headset channels
- The only evaluated dataset providing this synchronized pair — required for supervised Wave-U-Net training

**Key measurements from exploratory analysis:**
| Metric | Value |
|---|---|
| Noise floor ratio | 4.4× higher in array vs headset |
| Array VAD activity | 76.8% (realistic: 40–60%) |
| Overlap proxy | 25.8% of frames |

**Outputs:** Dataset selection with full validity assessment, answers to both required questions (how it will be used, why it is the best option)

---

## Task 2: Audio Enhancement Algorithms and Method Evaluation

Implements and compares 5 audio enhancement methods on a 3-minute sample of AMI ES2008a array audio. Evaluated using reference-free metrics (SQUIM) to avoid cross-microphone mismatch issues inherent in array/headset comparison.

**Methods implemented:**
| # | Method | Type |
|---|---|---|
| 1 | Spectral Subtraction | DSP baseline |
| 2 | Wiener Filter | DSP — statistically optimal |
| 3 | MMSE-LSA | DSP — perceptually motivated |
| 4 | Multi-band Wiener | DSP — frequency-adaptive |
| 5 | SpectralMaskNet | Neural — LSTM soft mask (architecture demo) |

**Results:**
| Method | SQUIM-STOI | SI-SDR | VAD After | CQI |
|---|---|---|---|---|
| Noisy Input | 0.821 | 3.6 dB | 76.8% | — |
| Spectral Sub. | 0.931 | 9.8 dB | 61.8% | +0.2418 |
| Wiener | 0.901 | 7.2 dB | 60.3% | +0.1456 |
| MMSE-LSA | 0.901 | 7.4 dB | 62.7% | +0.1447 |
| **Multi-band Wiener** | **0.947** | **11.3 dB** | **60.6%** | **+0.3562 🏆** |
| SpectralMaskNet | 0.874 | 9.0 dB | 67.2% | +0.2090 |

**Evaluation framework:**
$$\text{CQI} = 0.50 \cdot \Delta\text{SQUIM} + 0.30 \cdot \Delta\text{STOI} + 0.20 \cdot \Delta\text{WER}$$

**Outputs:** Audio analysis plots, enhanced WAV files, CQI method ranking, radar chart comparison

---

## Audio Samples

| File | Description |
|---|---|
| `input_audio.wav` | Raw far-field array mic — ES2008a |
| `output_audio.wav` | Multi-band Wiener output — best CQI (+0.3562) |

---

## Research Report

`Research_Report_TEAMS_COMMS.pdf` — Personal technical research document exploring approaches, dataset comparisons, enhancement methods, and pipeline decisions before proposal writing. Includes pipeline decision based on Task 2 findings: Multi-band Wiener confirmed as DSP pre-cleaning stage for Wave-U-Net hybrid pipeline.

---

## How to Run

### Requirements
```bash
pip install librosa pystoi openai-whisper jiwer scipy numpy matplotlib pandas
pip install torch torchaudio
```

### Audio files:
Running the 'Load Audio' cell auto downloads the audio data used.

---

## Full Project Pipeline

Based on Task 2 evaluation, the confirmed pipeline is:
```
Raw Array Audio
      ↓
High-Pass Filter (80 Hz)
      ↓
Multi-band Wiener  ←  DSP pre-cleaning (best Task 2 performer)
      ↓
Wave-U-Net        
      ↓
Evaluation: SQUIM · STOI · WER · CQI
```

DSP baseline CQI of **+0.3562** is the benchmark Wave-U-Net will be evaluated against.
