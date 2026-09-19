import streamlit as st
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram
from scipy.stats import kurtosis
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & RESPONSIVE CUSTOM CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Chassis Vibration Analysis System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0F172A;
            color: #F8FAFC;
        }
        /* Metric cards custom container styling */
        div[data-testid="stMetric"] {
            background-color: #1E293B;
            padding: 10px 12px !important;
            border-radius: 8px;
            border: 1px solid #334155;
            min-height: 80px;
        }
        /* Metric Labels: smaller size and wrapping for tight viewports */
        div[data-testid="stMetricLabel"] > label {
            color: #94A3B8 !important;
            font-size: 11px !important;
            font-weight: 600 !important;
            white-space: normal !important;
            word-wrap: break-word !important;
        }
        /* Metric Values: adjusted size to prevent truncation */
        div[data-testid="stMetricValue"] > div {
            color: #F8FAFC !important;
            font-size: 16px !important;
            font-weight: 700 !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. SIMULATION ENGINE & PIPELINE
# -----------------------------------------------------------------------------
FS = 1000  # Sampling frequency (Hz)
DURATION = 5  # seconds
T = np.arange(0, DURATION, 1 / FS)


def extract_features(signal):
    """Extracts statistical time-domain and FFT spectral metrics."""
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    sig_kurtosis = kurtosis(signal)
    crest_factor = peak / (rms + 1e-6)

    # Frequency-domain spectral features
    spectrum = np.abs(fft(signal))
    half = len(signal) // 2
    spectrum = spectrum[:half]
    frequencies = fftfreq(len(signal), 1 / FS)[:half]
    spectrum[0] = 0  # Remove DC component

    dominant_frequency = frequencies[np.argmax(spectrum)]
    frequency_energy = np.mean(spectrum**2)

    return (
        rms,
        peak,
        sig_kurtosis,
        crest_factor,
        dominant_frequency,
        frequency_energy,
    )


@st.cache_resource
def train_isolation_forest():
    """Trains the baseline Isolation Forest model once and caches it."""
    training_features = []
    np.random.seed(42)
    for _ in range(120):
        sig = (
            0.20 * np.sin(2 * np.pi * 20 * T)
            + 0.05 * np.sin(2 * np.pi * 45 * T)
            + 0.02 * np.random.randn(len(T))
        )
        rms, peak, k_val, cf_val, dom_freq, freq_energy = extract_features(sig)
        training_features.append(
            [rms, peak, k_val, cf_val, dom_freq, freq_energy]
        )

    model = IsolationForest(contamination=0.08, random_state=42)
    model.fit(np.array(training_features))
    return model


model = train_isolation_forest()


# -----------------------------------------------------------------------------
# 3. SIGNAL GENERATION LOGIC
# -----------------------------------------------------------------------------
def generate_signal(condition):
    np.random.seed(42)
    if condition == "Normal":
        sig = 0.20 * np.sin(2 * np.pi * 20 * T) + 0.05 * np.sin(
            2 * np.pi * 45 * T
        )
        noise = 0.02 * np.random.randn(len(T))
        return sig + noise

    elif condition == "Warning":
        sig = (
            0.22 * np.sin(2 * np.pi * 20 * T)
            + 0.08 * np.sin(2 * np.pi * 45 * T)
            + 0.12 * np.sin(2 * np.pi * 85 * T)
        )
        noise = 0.03 * np.random.randn(len(T))
        return sig + noise

    else:  # Critical
        sig = (
            0.25 * np.sin(2 * np.pi * 20 * T)
            + 0.10 * np.sin(2 * np.pi * 45 * T)
            + 0.35 * np.sin(2 * np.pi * 130 * T)
        )
        spikes = np.zeros(len(T))
        spike_indices = np.random.choice(len(T), size=15, replace=False)
        spikes[spike_indices] = np.random.choice([-0.8, 0.8], size=15)
        noise = 0.05 * np.random.randn(len(T))
        return sig + spikes + noise


# -----------------------------------------------------------------------------
# 4. SIDEBAR CONTROLS & HEADER
# -----------------------------------------------------------------------------
st.title("⚡ SMART CHASSIS VIBRATION DIAGNOSTICS")
st.caption(
    "Time-Frequency STFT • Kurtosis Transient Detection • Multi-Stage Risk Analytics"
)

st.sidebar.header("🕹️ Simulation Controls")
condition = st.sidebar.radio(
    "Select Operating Scenario:",
    ["Normal", "Warning", "Critical"],
    format_func=lambda x: {
        "Normal": "🟢 Simulate NORMAL",
        "Warning": "🟡 Simulate WARNING (Loose Joint)",
        "Critical": "🔴 Simulate CRITICAL (Crack)",
    }[x],
)

# -----------------------------------------------------------------------------
# 5. INFERENCE & MULTI-STAGE DIAGNOSTIC DECISION ENGINE
# -----------------------------------------------------------------------------
raw_signal = generate_signal(condition)
rms, peak, k_val, cf_val, dom_freq, freq_energy = extract_features(raw_signal)

features = np.array([[rms, peak, k_val, cf_val, dom_freq, freq_energy]])
anomaly_score = model.decision_function(features)[0]

if anomaly_score > -0.02 and k_val < 3.5:
    severity = "LEVEL 0: NORMAL"
    status_text = "STATUS: SYSTEM NOMINAL"
    color = "#10B981"  # Green
elif anomaly_score > -0.08 or (k_val >= 3.5 and k_val < 6.0):
    severity = "LEVEL 1: WARNING"
    status_text = "STATUS: SERVICE REQUIRED"
    color = "#F59E0B"  # Yellow/Orange
else:
    severity = "LEVEL 2: CRITICAL"
    status_text = "STATUS: CRITICAL FAULT"
    color = "#EF4444"  # Red

# -----------------------------------------------------------------------------
# 6. STATUS HEADER & COMPACT METRIC KPI CARDS
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Diagnostic Status")
st.sidebar.markdown(
    f"""
    <div style="
        color: {color}; 
        font-size: 15px; 
        font-weight: bold; 
        line-height: 1.2;
        padding: 4px 0px;
    ">
        {status_text}
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("RMS ACCELERATION", f"{rms:.3f} g")
m2.metric("PEAK ACCEL", f"{peak:.3f} g")
m3.metric("KURTOSIS", f"{k_val:.2f}")
m4.metric("DOMINANT FREQ", f"{dom_freq:.1f} Hz")
m5.metric("DIAGNOSTIC LEVEL", severity)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. MULTI-PANEL DIAGNOSTIC GRAPHICS GRID (2x2)
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 6.5), dpi=100)
fig.patch.set_facecolor("#1E293B")

time_ax, stft_ax, freq_ax, score_ax = axs.flatten()

for ax in (time_ax, stft_ax, freq_ax, score_ax):
    ax.set_facecolor("#0F172A")
    ax.tick_params(colors="#94A3B8", labelsize=8)
    ax.grid(True, color="#334155", linestyle="--", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color("#334155")

# 1. Time Signal Plot
display_samples = 2000
time_ax.plot(
    T[:display_samples],
    raw_signal[:display_samples],
    color=color,
    linewidth=1,
)
time_ax.set_title(
    f"Time-Domain Acceleration ({condition})",
    color="#F8FAFC",
    fontsize=10,
    pad=8,
)
time_ax.set_xlabel("Time (s)", color="#94A3B8", fontsize=8)
time_ax.set_ylabel("Accel (g)", color="#94A3B8", fontsize=8)

# 2. STFT Spectrogram
f_stft, t_stft, Sxx = spectrogram(raw_signal, FS, nperseg=128)
stft_ax.pcolormesh(
    t_stft, f_stft, 10 * np.log10(Sxx + 1e-10), cmap="magma", shading="gouraud"
)
stft_ax.set_ylim(0, 200)
stft_ax.set_title(
    "STFT Spectrogram (Time-Frequency)",
    color="#F8FAFC",
    fontsize=10,
    pad=8,
)
stft_ax.set_xlabel("Time (s)", color="#94A3B8", fontsize=8)
stft_ax.set_ylabel("Freq (Hz)", color="#94A3B8", fontsize=8)

# 3. FFT Frequency Spectrum
N = len(raw_signal)
spectrum = np.abs(fft(raw_signal))
frequencies = fftfreq(N, 1 / FS)
mask = frequencies >= 0

freq_ax.plot(frequencies[mask], spectrum[mask], color="#38BDF8", linewidth=1)
freq_ax.set_xlim(0, 200)
freq_ax.set_title(
    "Frequency Spectrum (FFT)", color="#F8FAFC", fontsize=10, pad=8
)
freq_ax.set_xlabel("Frequency (Hz)", color="#94A3B8", fontsize=8)
freq_ax.set_ylabel("Amplitude", color="#94A3B8", fontsize=8)

# 4. Multi-Stage Anomaly Risk Gauge
categories = ["Normal Level", "Warning Threshold", "Current Run Score"]
scores = [0.05, -0.02, anomaly_score]
bar_colors = ["#10B981", "#F59E0B", color]

score_ax.barh(categories, scores, color=bar_colors, height=0.5)
score_ax.set_xlim(-0.25, 0.15)
score_ax.set_title(
    "ML Anomaly Score Index", color="#F8FAFC", fontsize=10, pad=8
)
score_ax.set_xlabel("Decision Score", color="#94A3B8", fontsize=8)

plt.tight_layout(pad=2.0)
st.pyplot(fig)
