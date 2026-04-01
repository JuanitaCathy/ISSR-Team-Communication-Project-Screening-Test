import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.signal import butter, sosfilt
from scipy.special import expi
from pystoi import stoi as compute_stoi
import tempfile, os, io, urllib.request
from scipy.io.wavfile import write as wav_write
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Team Audio Enhancement Dashboard",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin: 4px;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #79c0ff; }
    .metric-label { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }
    .metric-delta-pos { color: #56d364; font-size: 0.9rem; }
    .metric-delta-neg { color: #f78166; font-size: 0.9rem; }
    .hero-banner {
        background: linear-gradient(135deg, #1f4e79 0%, #2e75b6 100%);
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
    }
    .hero-title { font-size: 1.8rem; font-weight: bold; color: white; }
    .hero-sub { color: #d5e8f0; font-size: 1rem; margin-top: 8px; }
    .section-header {
        border-left: 4px solid #2e75b6;
        padding-left: 12px;
        margin: 20px 0 12px 0;
        font-size: 1.1rem;
        font-weight: bold;
        color: #c9d1d9;
    }
    div[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    .stSelectbox label, .stFileUploader label { color: #c9d1d9 !important; }
    h1, h2, h3 { color: #c9d1d9 !important; }
    .stTabs [data-baseweb="tab"] { color: #8b949e; }
    .stTabs [aria-selected="true"] { color: #79c0ff; }
    .stRadio label { color: #c9d1d9 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
SR = 16000
N_FFT, HOP = 1024, 256
C = {
    'noisy':  '#f78166',
    'ss':     '#79c0ff',
    'wiener': '#56d364',
    'mmsa':   '#d2a8ff',
    'mband':  '#ffa657',
}

# ── DSP Functions ──────────────────────────────────────────────────────────────
def high_pass_filter(signal, sr=SR, cutoff_hz=80, order=4):
    sos = butter(order, cutoff_hz / (sr / 2), btype='high', output='sos')
    return sosfilt(sos, signal).astype(np.float32)

def normalise(signal, target=0.95):
    peak = np.max(np.abs(signal))
    return (signal / peak * target).astype(np.float32) if peak > 1e-8 else signal

def get_noise_profile(magnitude, sr=SR, hop=HOP, ref_sec=2.0):
    ref_frames = max(1, int(np.ceil(ref_sec * sr / hop)))
    ref_frames = min(ref_frames, magnitude.shape[1])
    return np.mean(magnitude[:, :ref_frames] ** 2, axis=1, keepdims=True)

def enhance_spectral_subtraction(signal, alpha=1.0, beta=0.02):
    signal = high_pass_filter(signal)
    D = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP)
    mag, ph = np.abs(D), np.angle(D)
    noise_p = get_noise_profile(mag)
    enhanced_sq = np.maximum(mag**2 - alpha * noise_p, beta * noise_p)
    out = librosa.istft(np.sqrt(enhanced_sq) * np.exp(1j * ph),
                        hop_length=HOP, length=len(signal))
    return normalise(out.astype(np.float32))

def enhance_wiener(signal):
    signal = high_pass_filter(signal)
    D = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP)
    mag, ph = np.abs(D), np.angle(D)
    noise_p = get_noise_profile(mag)
    signal_p = np.maximum(mag**2 - noise_p, 1e-8)
    xi = signal_p / np.maximum(noise_p, 1e-8)
    gain = xi / (1.0 + xi)
    out = librosa.istft(gain * mag * np.exp(1j * ph),
                        hop_length=HOP, length=len(signal))
    return normalise(out.astype(np.float32))

def enhance_mmse_lsa(signal):
    signal = high_pass_filter(signal)
    D = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP)
    mag, ph = np.abs(D), np.angle(D)
    noise_p = get_noise_profile(mag)
    gamma = mag**2 / np.maximum(noise_p, 1e-8)
    xi = np.maximum(gamma - 1.0, 1e-8)
    nu = np.maximum((xi / (1.0 + xi)) * gamma, 1e-10)
    gain = np.clip((xi / (1.0 + xi)) * np.exp(0.5 * expi(nu)), 0, 1)
    out = librosa.istft(gain * mag * np.exp(1j * ph),
                        hop_length=HOP, length=len(signal))
    return normalise(out.astype(np.float32))

def enhance_multiband_wiener(signal):
    BANDS_HZ = [
        (0,    150,  0.95),
        (150,  500,  0.80),
        (500,  2000, 0.50),
        (2000, 4000, 0.70),
        (4000, 8000, 0.85),
    ]
    signal = high_pass_filter(signal)
    D = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP)
    mag, ph = np.abs(D), np.angle(D)
    noise_p = get_noise_profile(mag)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    gain = np.zeros_like(mag)
    for lo, hi, suppression in BANDS_HZ:
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if len(idx) == 0:
            continue
        xi_b = np.maximum(mag[idx, :]**2 - noise_p[idx, :], 1e-8) / \
               np.maximum(noise_p[idx, :], 1e-8)
        g_b = xi_b / (1.0 + xi_b)
        gain[idx, :] = np.maximum(g_b, 1.0 - suppression)
    out = librosa.istft(gain * mag * np.exp(1j * ph),
                        hop_length=HOP, length=len(signal))
    return normalise(out.astype(np.float32))

ENHANCERS = {
    'Spectral Subtraction': enhance_spectral_subtraction,
    'Wiener Filter':        enhance_wiener,
    'MMSE-LSA':             enhance_mmse_lsa,
    'Multi-band Wiener ★':  enhance_multiband_wiener,
}

# ── Default audio loader ───────────────────────────────────────────────────────
@st.cache_data
def load_default_audio():
    """Load and cache the local AMI ES2008a array mic sample."""
    local_path = "ES2008a.Array1-01.wav"
    try:
        audio, _ = librosa.load(local_path, sr=SR, mono=True)
        return audio, "AMI ES2008a — Array Microphone (far-field, noisy)"
    except Exception as e:
        np.random.seed(42)
        t = np.linspace(0, 30, 30 * SR)

        speech = (0.3 * np.sin(2 * np.pi * 200 * t) +
                  0.2 * np.sin(2 * np.pi * 400 * t) +
                  0.1 * np.sin(2 * np.pi * 800 * t))
        noise  = 0.15 * np.random.randn(len(t))
        signal = (speech + noise).astype(np.float32)
        return signal, "Demo signal"

# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(noisy, enhanced, sr=SR):
    n = min(len(noisy), len(enhanced))

    try:
        stoi_before = compute_stoi(noisy[:n], noisy[:n], sr, extended=False)
        stoi_after  = compute_stoi(noisy[:n], enhanced[:n], sr, extended=False)
    except Exception:
        stoi_before = stoi_after = 0.0

    noisy_rms    = float(np.sqrt(np.mean(noisy**2)))
    enhanced_rms = float(np.sqrt(np.mean(enhanced**2)))

    mag_n = np.abs(librosa.stft(noisy,    n_fft=N_FFT, hop_length=HOP))
    mag_e = np.abs(librosa.stft(enhanced, n_fft=N_FFT, hop_length=HOP))

    sf_n = float(np.mean(np.exp(np.mean(np.log(mag_n + 1e-8), axis=0)) /
                          (np.mean(mag_n, axis=0) + 1e-8)))
    sf_e = float(np.mean(np.exp(np.mean(np.log(mag_e + 1e-8), axis=0)) /
                          (np.mean(mag_e, axis=0) + 1e-8)))

    rms_n = librosa.feature.rms(y=noisy,    hop_length=HOP)[0]
    rms_e = librosa.feature.rms(y=enhanced, hop_length=HOP)[0]
    vad_n = float(np.mean(rms_n > np.mean(rms_n) * 0.5)) * 100
    vad_e = float(np.mean(rms_e > np.mean(rms_e) * 0.5)) * 100

    return {
        'stoi_before': round(stoi_before, 4),
        'stoi_after':  round(stoi_after,  4),
        'rms_before':  round(noisy_rms,    4),
        'rms_after':   round(enhanced_rms, 4),
        'sf_before':   round(sf_n, 4),
        'sf_after':    round(sf_e, 4),
        'vad_before':  round(vad_n, 1),
        'vad_after':   round(vad_e, 1),
    }

# ── Plot helpers ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#c9d1d9',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#c9d1d9',
    'grid.color':       '#21262d',
    'grid.alpha':       0.5,
    'font.size':        10,
})

def plot_waveforms(noisy, enhanced, sr=SR, method_name="Enhanced"):
    t = np.linspace(0, len(noisy) / sr, len(noisy))
    fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    axes[0].plot(t, noisy,    color=C['noisy'],  lw=0.5, alpha=0.9)
    axes[0].set_title("Noisy Input", fontweight='bold', color=C['noisy'])
    axes[0].set_ylabel("Amplitude"); axes[0].grid(True)
    axes[1].plot(t, enhanced, color='#79c0ff',   lw=0.5, alpha=0.9)
    axes[1].set_title(f"Enhanced — {method_name}", fontweight='bold', color='#79c0ff')
    axes[1].set_ylabel("Amplitude"); axes[1].set_xlabel("Time (s)"); axes[1].grid(True)
    plt.tight_layout()
    return fig

def plot_spectrograms(noisy, enhanced, sr=SR, method_name="Enhanced"):
    D_n = librosa.amplitude_to_db(
        np.abs(librosa.stft(noisy,    n_fft=N_FFT, hop_length=HOP)), ref=np.max)
    D_e = librosa.amplitude_to_db(
        np.abs(librosa.stft(enhanced, n_fft=N_FFT, hop_length=HOP)), ref=np.max)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    img0 = librosa.display.specshow(D_n, sr=sr, hop_length=HOP,
                                    x_axis='time', y_axis='log', ax=axes[0], cmap='magma')
    img1 = librosa.display.specshow(D_e, sr=sr, hop_length=HOP,
                                    x_axis='time', y_axis='log', ax=axes[1], cmap='magma')
    axes[0].set_title("Noisy Input Spectrogram",
                      fontweight='bold', color=C['noisy'])
    axes[1].set_title(f"Enhanced — {method_name}",
                      fontweight='bold', color='#79c0ff')
    fig.colorbar(img0, ax=axes[0], format='%+2.0f dB')
    fig.colorbar(img1, ax=axes[1], format='%+2.0f dB')
    plt.tight_layout()
    return fig

def plot_rms_comparison(noisy, enhanced, sr=SR):
    rms_n = librosa.feature.rms(y=noisy,    hop_length=HOP)[0]
    rms_e = librosa.feature.rms(y=enhanced, hop_length=HOP)[0]
    t_rms = librosa.frames_to_time(np.arange(len(rms_n)), sr=sr, hop_length=HOP)
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(t_rms, rms_n, color=C['noisy'], lw=0.8, label='Noisy')
    axes[0].plot(t_rms, rms_e, color='#79c0ff',  lw=0.8, label='Enhanced')
    axes[0].set_title("Frame-Level RMS Energy", fontweight='bold')
    axes[0].set_ylabel("RMS"); axes[0].legend(fontsize=9); axes[0].grid(True)
    delta = rms_n - rms_e
    axes[1].fill_between(t_rms, delta, 0, where=(delta > 0),
                         color=C['noisy'], alpha=0.5, label='Noise removed')
    axes[1].fill_between(t_rms, delta, 0, where=(delta <= 0),
                         color='#79c0ff', alpha=0.5, label='Speech retained')
    axes[1].axhline(0, color='#8b949e', lw=0.8)
    axes[1].set_title("Energy Delta (Noisy − Enhanced)", fontweight='bold')
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("ΔRMS")
    axes[1].legend(fontsize=9); axes[1].grid(True)
    plt.tight_layout()
    return fig

def plot_radar(results_dict):
    categories = ['STOI', 'VAD\nnorm', 'RMS\nreduction', 'Spectral\nFlatness↓']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_facecolor('#161b22')
    colors = ['#79c0ff', '#56d364', '#d2a8ff', '#ffa657']
    for (name, m), col in zip(results_dict.items(), colors):
        stoi_norm = np.clip(m['stoi_after'], 0, 1)
        vad_norm  = np.clip(1 - abs(m['vad_after'] / 100 - 0.50) / 0.50, 0, 1)
        rms_red   = np.clip(1 - m['rms_after'] / max(m['rms_before'], 1e-8), 0, 1)
        sf_red    = np.clip(1 - m['sf_after']  / max(m['sf_before'],  1e-8), 0, 1)
        vals = [stoi_norm, vad_norm, rms_red, sf_red] + [stoi_norm]
        ax.plot(angles, vals, color=col, lw=2, label=name)
        ax.fill(angles, vals, color=col, alpha=0.06)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, color='#c9d1d9')
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, color='#30363d', lw=0.8)
    ax.xaxis.grid(True, color='#30363d', lw=0.8)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=9)
    ax.set_title("Method Comparison Radar", fontweight='bold', pad=20)
    return fig

def audio_bytes(signal, sr=SR):
    buf = io.BytesIO()
    wav_write(buf, sr, (np.clip(signal, -1, 1) * 32767).astype(np.int16))
    buf.seek(0)
    return buf.read()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Audio Source")
    source = st.radio(
        "",
        ["Use default AMI sample", "Upload my own"],
        index=0,
        help="Default loads AMI ES2008a array mic — a real far-field meeting recording"
    )

    uploaded = None
    if source == "Upload my own":
        uploaded = st.file_uploader(
            "Upload a WAV file",
            type=["wav"],
            help="Any WAV file works. Far-field or noisy recordings show the most improvement."
        )

    st.markdown("### Enhancement Method")
    method = st.selectbox(
        "Select method",
        list(ENHANCERS.keys()),
        index=3,
        help="Multi-band Wiener ★ is the recommended method — best CQI (+0.3562) in Task 2 evaluation"
    )

    st.markdown("### Clip Duration")
    clip_sec = st.slider(
        "Seconds to process", 10, 120, 60, 10,
        help="Shorter clips process faster. 60s is recommended."
    )

    run_all = st.checkbox(
        "Compare all methods",
        value=False,
        help="Runs all 4 methods and shows a radar chart comparison. Takes ~30s."
    )

    st.markdown("---")
    st.markdown("""
**Pipeline:**
```
Raw Audio
    ↓
HPF (80 Hz)
    ↓
Selected Method
    ↓
Peak Normalisation
    ↓
Metrics + Visualisation
```
""")
    st.markdown("---")
    st.caption("Juanita Cathy · GSoC 2026 · HumanAI")

# ══════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🎙️ Team Communication Audio Enhancement</div>
    <div class="hero-sub">
        GSoC 2026 · HumanAI Foundation · TRIP Lab, University of Alabama<br>
        Real far-field meeting audio · 4 DSP enhancement methods · 
        Reference-free evaluation (SQUIM · STOI · VAD · CQI)
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD AUDIO
# ══════════════════════════════════════════════════════════════════════════════
if source == "Use default AMI sample":
    with st.spinner("Loading AMI ES2008a array mic sample..."):
        audio_full, source_label = load_default_audio()
elif uploaded is not None:
    with st.spinner("Loading uploaded audio..."):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(uploaded.read())
            tmp_path = f.name
        audio_full, _ = librosa.load(tmp_path, sr=SR, mono=True)
        os.remove(tmp_path)
        source_label = uploaded.name
else:
    # Upload mode but nothing uploaded yet
    st.info("👈 Upload a WAV file in the sidebar to get started.")
    st.markdown("### About This Dashboard")
    st.markdown("""
| Method | Type | CQI (Task 2) |
|---|---|---|
| Spectral Subtraction | DSP | +0.2418 |
| Wiener Filter | DSP | +0.1456 |
| MMSE-LSA | DSP | +0.1447 |
| **Multi-band Wiener ★** | **DSP** | **+0.3562 🏆** |

Switch to **"Use default AMI sample"** in the sidebar to see the dashboard in action immediately.
    """)
    st.stop()

n_samples = min(int(clip_sec * SR), len(audio_full))
noisy     = audio_full[:n_samples].astype(np.float32)

st.success(f"✅ **{source_label}** — {len(noisy)/SR:.1f}s @ {SR} Hz")

# ══════════════════════════════════════════════════════════════════════════════
# RUN ENHANCEMENT
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner(f"Running {method}..."):
    enhanced = ENHANCERS[method](noisy.copy())
    metrics  = compute_metrics(noisy, enhanced)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Metrics", "🌊 Waveform", "🔊 Spectrogram", "📈 All Methods"]
)

# ── TAB 1: METRICS ────────────────────────────────────────────────────────────
with tab1:
    st.markdown(f'<div class="section-header">Results — {method}</div>',
                unsafe_allow_html=True)

    d_stoi = metrics['stoi_after'] - metrics['stoi_before']
    d_vad  = metrics['vad_after']  - metrics['vad_before']
    d_rms  = metrics['rms_after']  - metrics['rms_before']
    d_sf   = metrics['sf_after']   - metrics['sf_before']

    def delta_color(v, invert=False):
        if invert: v = -v
        return "metric-delta-pos" if v >= 0 else "metric-delta-neg"

    def delta_arrow(v, invert=False):
        if invert: v = -v
        return "▲" if v >= 0 else "▼"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['stoi_after']:.3f}</div>
            <div class="metric-label">STOI (after)</div>
            <div class="{delta_color(d_stoi)}">{delta_arrow(d_stoi)} {d_stoi:+.4f}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['vad_after']:.1f}%</div>
            <div class="metric-label">VAD Activity (after)</div>
            <div class="{delta_color(d_vad, invert=True)}">{delta_arrow(d_vad, invert=True)} {d_vad:+.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['rms_after']:.4f}</div>
            <div class="metric-label">RMS Energy (after)</div>
            <div class="{delta_color(d_rms, invert=True)}">{delta_arrow(d_rms, invert=True)} {d_rms:+.4f}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['sf_after']:.4f}</div>
            <div class="metric-label">Spectral Flatness (after)</div>
            <div class="{delta_color(d_sf, invert=True)}">{delta_arrow(d_sf, invert=True)} {d_sf:+.4f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Before vs After Summary")
        df = pd.DataFrame({
            'Metric':  ['STOI', 'VAD Activity (%)', 'RMS Energy', 'Spectral Flatness'],
            'Before':  [metrics['stoi_before'], metrics['vad_before'],
                        metrics['rms_before'],  metrics['sf_before']],
            'After':   [metrics['stoi_after'],  metrics['vad_after'],
                        metrics['rms_after'],   metrics['sf_after']],
            'Δ':       [round(d_stoi, 4), round(d_vad, 1),
                        round(d_rms,  4), round(d_sf,  4)],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("#### Screening Test Results (AMI ES2008a)")
        df_ref = pd.DataFrame({
            'Method':     ['Noisy Input', 'Spectral Sub.', 'Wiener',
                           'MMSE-LSA', 'Multi-band Wiener ★'],
            'SQUIM-STOI': [0.821, 0.931, 0.901, 0.901, 0.947],
            'SI-SDR':     ['3.6 dB', '9.8 dB', '7.2 dB', '7.4 dB', '11.3 dB'],
            'VAD After':  ['76.8%', '61.8%', '60.3%', '62.7%', '60.6%'],
            'CQI':        ['—', '+0.2418', '+0.1456', '+0.1447', '+0.3562 🏆'],
        })
        st.dataframe(df_ref, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("#### VAD Normalisation")
        fig_vad, ax = plt.subplots(figsize=(5, 4))
        ax.bar(['Before\n(Noisy)', 'After\n(Enhanced)'],
               [metrics['vad_before'], metrics['vad_after']],
               color=[C['noisy'], '#79c0ff'],
               edgecolor='#30363d', width=0.4)
        ax.axhspan(40, 60, alpha=0.15, color='#56d364', label='Realistic range (40–60%)')
        ax.set_ylabel("VAD Activity (%)")
        ax.set_ylim(0, 100)
        ax.set_title("VAD Ratio — Noise Floor Proof", fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, axis='y')
        for x, v in enumerate([metrics['vad_before'], metrics['vad_after']]):
            ax.text(x, v + 1.5, f"{v:.1f}%", ha='center',
                    fontsize=11, fontweight='bold')
        st.pyplot(fig_vad)
        plt.close()

    st.markdown("---")
    st.markdown("#### 🎧 Audio Playback")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("**Noisy Input**")
        st.audio(audio_bytes(noisy), format='audio/wav')
    with col_p2:
        st.markdown(f"**Enhanced — {method}**")
        st.audio(audio_bytes(enhanced), format='audio/wav')
        st.download_button(
            "⬇️ Download Enhanced Audio",
            data=audio_bytes(enhanced),
            file_name=f"enhanced_{method.replace(' ','_').replace('★','').strip().lower()}.wav",
            mime="audio/wav"
        )

# ── TAB 2: WAVEFORM ───────────────────────────────────────────────────────────
with tab2:
    st.markdown(f'<div class="section-header">Waveform — {method}</div>',
                unsafe_allow_html=True)
    fig_w = plot_waveforms(noisy, enhanced, method_name=method)
    st.pyplot(fig_w); plt.close()

    st.markdown("---")
    st.markdown("#### Frame-Level Energy Delta")
    fig_rms = plot_rms_comparison(noisy, enhanced)
    st.pyplot(fig_rms); plt.close()

# ── TAB 3: SPECTROGRAM ────────────────────────────────────────────────────────
with tab3:
    st.markdown(f'<div class="section-header">Spectrogram — {method}</div>',
                unsafe_allow_html=True)
    fig_s = plot_spectrograms(noisy, enhanced, method_name=method)
    st.pyplot(fig_s); plt.close()
    st.markdown("""
> **What to look for:**
> - Suppressed dark background in silent regions → less noise energy
> - Clearer harmonic striations in speech regions → better speech preservation
    """)

# ── TAB 4: ALL METHODS ────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">All Methods Comparison</div>',
                unsafe_allow_html=True)

    if not run_all:
        st.info("☑️ Enable **Compare all methods** in the sidebar to run all 4 methods.")
        st.markdown("""
| Method | SQUIM-STOI | SI-SDR | VAD After | CQI |
|---|---|---|---|---|
| Noisy Input | 0.821 | 3.6 dB | 76.8% ⚠️ | — |
| Spectral Subtraction | 0.931 | 9.8 dB | 61.8% ✅ | +0.2418 |
| Wiener Filter | 0.901 | 7.2 dB | 60.3% ✅ | +0.1456 |
| MMSE-LSA | 0.901 | 7.4 dB | 62.7% ✅ | +0.1447 |
| **Multi-band Wiener ★** | **0.947** | **11.3 dB** | **60.6% ✅** | **+0.3562 🏆** |
        """)
    else:
        results_all = {}
        prog = st.progress(0, text="Running all methods...")
        for i, (name, fn) in enumerate(ENHANCERS.items()):
            prog.progress((i + 1) / len(ENHANCERS), text=f"Running {name}...")
            enh = fn(noisy.copy())
            results_all[name] = compute_metrics(noisy, enh)
        prog.empty()
        st.success("✅ All methods complete")

        rows = []
        for name, m in results_all.items():
            rows.append({
                'Method':    name,
                'STOI':      m['stoi_after'],
                'VAD After': f"{m['vad_after']:.1f}%",
                'RMS After': m['rms_after'],
                'ΔSTOI':     round(m['stoi_after'] - m['stoi_before'], 4),
                'ΔVAD':      f"{m['vad_after'] - m['vad_before']:+.1f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("#### Radar Chart")
            fig_radar = plot_radar(results_all)
            st.pyplot(fig_radar); plt.close()

        with col_r2:
            st.markdown("#### STOI Comparison")
            fig_bar, ax = plt.subplots(figsize=(6, 4))
            names  = list(results_all.keys())
            stoi_v = [results_all[n]['stoi_after'] for n in names]
            colors = ['#79c0ff', '#56d364', '#d2a8ff', '#ffa657']
            bars   = ax.bar(range(len(names)), stoi_v, color=colors,
                            edgecolor='#30363d', width=0.5)
            for bar, v in zip(bars, stoi_v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.3f}", ha='center', fontsize=9, fontweight='bold')
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8)
            ax.set_ylabel("STOI")
            ax.set_title("STOI After Enhancement", fontweight='bold')
            ax.grid(True, axis='y'); ax.set_ylim(0, 1)
            st.pyplot(fig_bar); plt.close()

        st.markdown("#### Audio Comparison")
        cols = st.columns(len(ENHANCERS))
        for col, (name, fn) in zip(cols, ENHANCERS.items()):
            enh = fn(noisy.copy())
            with col:
                st.markdown(f"**{name}**")
                st.audio(audio_bytes(enh), format='audio/wav')
