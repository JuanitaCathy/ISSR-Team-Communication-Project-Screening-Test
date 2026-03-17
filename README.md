# ISSR SCREENING TEST SUBMISSION FOR TEAM-COMMUNICATION-PROJECT
## Team Communication Processing and Analysis in Human-Factors Simulated Environment
**Organisation:** HumanAI Foundation <br>
**Name:** Juanita Cathy  

<img width="835" height="455" alt="Screenshot 2026-03-16 181651" src="https://github.com/user-attachments/assets/51180f75-88f9-4296-a46f-87044ea2c0d1" />

## Overview
1. **Notebook 1** — Pick and evaluate a public dataset suitable for team communication audio research
2. **Notebook 2** — Build and compare audio enhancement methods on a sample from that dataset

I approached both notebooks in a research prototype style: given that TRIP Lab is research-oriented, I wanted the work to reflect that rather than just presenting minimal outputs. The second notebook is intentionally detailed, covering 5 enhancement methods, a reference-free evaluation framework, and documented reasoning for every decision.

I also put together a **Research Report** independently before starting implementation: it documents my method comparisons, architecture decisions, and pipeline reasoning. It's not a proposal, just technical groundwork I did for my own reference. I've included it since it might be useful context.

---

## Task 1: Dataset Identification and Evaluation
**File:** `TASK1-TeamCommunication_Data_Identification_Notebook.ipynb`

### What is done?
I've identified and evaluated open-access audio corpora as proxies for TRIP Lab simulator audio. Three datasets were compared: AMI Meeting Corpus, ICSI, LibriCSS against project-specific criteria such as:
- synchronized array and headset channels
- real multi-speaker team interaction
- natural overlap and diarization annotations.

The **deciding factor** was the **synchronized array and headset pair from the same session**: the array mic gives the far-field noisy input that mirrors simulator conditions, and the headset gives the clean close-talk reference needed for supervised Wave-U-Net training. 

### **Why I Selected this dataset:** 
**AMI Meeting Corpus, Session ES2008a : [https://groups.inf.ed.ac.uk/ami/corpus/](https://groups.inf.ed.ac.uk/ami/corpus/)**
- Real groups of 4 people doing structured tasks together, with natural turn-taking and interruptions
- Synchronized far-field array + close-talk headset from the same recording: This is the only evaluated dataset offering this
- Full diarization, overlap and word-level transcript annotations per speaker
- ~100 hours across sessions, free for academic use
- The array/headset degradation gap is large and quantifiable which is ideal for enhancement benchmarking
  
*note: Considering the similarities between the Trip Lab's recording setup and this, the results can be applied to the actual lab data in the future.*

### **Key Numbers from exploratory analysis:**
| Metric | Value |
|---|---|
| Noise floor ratio | 4.4× higher |
| Array VAD activity | 76.8% (realistic: 40–60%) |
| Overlap proxy | 25.8% of frames |

### What I Analysed with it's representations:

- Waveform comparison: array vs headset, same session
- Spectrogram comparison: noise floor and harmonic structure differences
- RMS energy distribution and noise floor estimation
- Speech activity detection (VAD)
- Overlap proxy: spectral centroid + VAD heuristic for multi-speaker complexity
- Sample validity assessment: 7 project-specific criteria scored

**Outputs:** Dataset selection with full validity assessment, answers to both required questions (how it will be used, why it is the best option)

---

## Task 2: Audio Enhancement Algorithms and Method Evaluation
**File:**`TASL2_Audio_Enhancement_Methods_and_Evaluations.ipynb`

### What is done?
Implemented and compared 5 audio enhancement methods on a 3-minute sample of AMI ES2008a audio. Evaluated using reference-free metrics (SQUIM) to avoid cross-microphone mismatch issues inherent in array/headset comparison.

## **Enhancement Methods Used:**
I implemented 5 enhancement methods in order of increasing sophistication:

| Method | Type | What it does |
|---|---|---|
| Spectral Subtraction | DSP baseline | Estimates noise from lead-in silence, subtracts per frame |
| Wiener Filter | DSP | Statistically optimal SNR based gain per frequency bin |
| MMSE-LSA | DSP | Log-spectral amplitude estimator minimises perceptual error |
| Multi-band Wiener | DSP | 5 perceptual bands with different suppression floors per band |
| SpectralMaskNet | Neural | 2-layer LSTM soft TF mask, Wave-U-Net architectural precursor |

### A Note on Metrics ( SQUIM )

I wanted to use PESQ as the primary perceptual metric (it's in the research report) but ran into a fundamental problem: array and headset are physically different microphones. Computing `array − headset` measures microphone mismatch, not enhancement quality. This showed up empirically: SegSNR was flat at −5.2 to −5.6 dB regardless of algorithm or parameters.

So I switched to **SQUIM** as the primary metric: a reference-free neural quality estimator that scores audio on its own acoustic properties with no reference mic needed. This is what production pipelines use when matched references aren't available. STOI and WER are reported alongside it as supporting evidence.

**CQI (Communication Quality Index):**
```
CQI = 0.50 × SQUIM + 0.30 × STOI + 0.20 × WER
```
## **What I found:**
| Method | SQUIM-STOI | SI-SDR | VAD After | CQI |
|---|---|---|---|---|
| Noisy Input | 0.821 | 3.6 dB | 76.8% | — |
| Spectral Sub. | 0.931 | 9.8 dB | 61.8% | +0.2418 |
| Wiener | 0.901 | 7.2 dB | 60.3% | +0.1456 |
| MMSE-LSA | 0.901 | 7.4 dB | 62.7% | +0.1447 |
| **Multi-band Wiener** | **0.947** | **11.3 dB** | **60.6%** | **+0.3562** |
| SpectralMaskNet | 0.874 | 9.0 dB | 67.2% | +0.2090 |

**Outputs:** Audio analysis plots, enhanced WAV files, CQI method ranking, radar chart comparison

---

## Audio Samples

| File | Description |
|---|---|
| `input_audio.wav` | Raw far-field array mic — ES2008a |
| `output_audio.wav` | Multi-band Wiener output — best CQI (+0.3562) |

---

## Research Report

`Research_Report_TEAMS_COMMS.pdf`: Personal technical research document exploring approaches, dataset comparisons, enhancement methods, and pipeline decisions before proposal writing. Includes pipeline decision based on Task 2 findings: Multi-band Wiener confirmed as DSP pre-cleaning stage for Wave-U-Net hybrid pipeline.

---

## How to Run

### Requirements
```bash
pip install librosa pystoi openai-whisper jiwer scipy numpy matplotlib pandas
pip install torch torchaudio
```

### Audio files:
Running the 'Load Audio' cell auto downloads the audio data used or the AMI Link has.
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
