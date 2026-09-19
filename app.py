import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram
from scipy.stats import kurtosis
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import streamlit as st

# Set Streamlit Page Config
st.set_page_config(
    page_title="Smart Chassis Vibration Analytics",
    page_icon="🚗",
    layout="wide",
)

# Apply Dark Slate Custom Styling
st.markdown(
    """
    <style>
    .main { background-color: #0F172A; }
    .stMetric { background-color: #1E293B; padding: 12px; border-radius: 8px; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    </style>
    """,
    unsafe_allow_scope=True,
)


# ========================================================
# Core Signal Processing & ML Pipeline
# ========================================================
class SmartChassisAnalyzer:

    def __init__(self, fs=1000, duration=5):
        self.fs = fs
        self.duration = duration
        self.t = np.arange(0, self.duration, 1 / self.fs)

    def generate_normal_signal(self):
        signal = 0.20 * np.sin(2 * np.pi * 20 * self.t) + 0.05 * np.sin(
            2 * np.pi * 45 * self.t
        )
        noise = 0.02 * np.random.randn(len(self.t))
        return signal + noise

    def generate_warning_signal(self):
        signal = (
            0.22 * np.sin(2 * np.pi * 20 * self.t)
            + 0.08 * np.sin(2 * np.pi * 45 * self.t)
            + 0.12 * np.sin(2 * np.pi * 85 * self.t)
        )
        noise = 0.03 * np.random.randn(len(self.t))
        return signal + noise

    def generate_critical_signal(self):
        signal = (
            0.25 * np.sin(2 * np.pi * 20 * self.t)
            + 0.10 * np.sin(2 * np.pi * 45 * self.t)
            + 0.35 * np.sin(2 * np.pi * 130 * self.t)
        )
        spikes = np.zeros(len(self.t))
        spike_indices = np.random.choice(len(self.t), size=15, replace=False)
        spikes[spike_indices] = np.random.choice([-0.8, 0.8], size=15)
        noise = 0.05 * np.random.randn(len(self.t))
        return signal + spikes + noise

    def extract_features(self, signal):
        rms = np.sqrt(np.mean(signal**2))
        peak = np.max(np.abs(signal))
        sig_kurtosis = kurtosis(signal)
        crest_factor = peak / (rms + 1e-6)

        spectrum = np.abs(fft(signal))
        half = len(signal) // 2
        spectrum = spectrum[:half]
        frequencies = fftfreq(len(signal), 1 / self.fs)[:half]
        spectrum[0] = 0  # Remove DC offset

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


# Cache Model Training so it executes only ONCE on app launch
@st.cache_resource
def load_trained_model():
    analyzer = SmartChassisAnalyzer()
    training_features = []
    for _ in range(120):
        sig = (
            0.20 * np.sin(2 * np.pi * 20 * analyzer.t)
            + 0.05 * np.sin(2 * np.pi * 45 * analyzer.t)
            + 0.02 * np.random.randn(len(analyzer.t))
        )
        features = analyzer.extract_features(sig)
        training_features.append(features)

    model = IsolationForest(contamination=0.08, random_state=42)
    model.fit(np.array(training_features))
    return analyzer, model


# Initialize system resources
analyzer, model = load_trained_model()

# Handle State Persistence across re-runs
if "condition" not in st.session_state:
    st.session_state.condition = "Normal"

# ========================================================
# Streamlit Dashboard UI Layout
# ========================================================

# Header Title Block
st.title("⚡ SMART CHASSIS VIBRATION DIAGNOSTICS")
st.caption(
    "Time-Frequency STFT • Kurtosis Transient Detection • Multi-Stage Risk Analytics"
)
st.divider()

# Simulation Controls
st.subheader("🛠️ Simulation Controls")
col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    if st.button("🟢 Simulate NORMAL", use_container_width=True):
        st.session_state.condition = "Normal"

with col_btn2:
    if st.button("🟡 Simulate WARNING (Loose Joint)", use_container_width=True):
        st.session_state.condition = "Warning"

with col_btn3:
    if st.button("🔴 Simulate CRITICAL (Crack)", use_container_width=True):
        st.session_state.condition = "Critical"

# Run Diagnostics Engine based on state
cond = st.session_state.condition
if cond == "Normal":
    signal = analyzer.generate_normal_signal()
elif cond == "Warning":
    signal = analyzer.generate_warning_signal()
else:
    signal = analyzer.generate_critical_signal()

# Extract Features and Calculate ML Anomaly Score
rms, peak, k_val, cf_val, dom_freq, freq_energy = analyzer.extract_features(
    signal
)
features = np.array([[rms, peak, k_val, cf_val, dom_freq, freq_energy]])
anomaly_score = model.decision_function(features)[0]

# Multi-Stage Severity Classification Logic
if anomaly_score > -0.02 and k_val < 3.5:
    severity = "LEVEL 0: NORMAL"
    status_text = "SYSTEM NOMINAL"
    status_color = "#10B981"  # Green
elif anomaly_score > -0.08 or (k_val >= 3.5 and k_val < 6.0):
    severity = "LEVEL 1: WARNING"
    status_text = "SERVICE REQUIRED"
    status_color = "#F59E0B"  # Orange
else:
    severity = "LEVEL 2: CRITICAL"
    status_text = "CRITICAL FAULT"
    status_color = "#EF4444"  # Red

st.divider()

# Status Banner & KPI Metric Cards Display
st.markdown(
    f"<h3 style='color: {status_color}; text-align: left;'>STATUS: {status_text} ({severity})</h3>",
    unsafe_allow_html=True,
)

m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
m_col1.metric("RMS ACCEL", f"{rms:.3f} g")
m_col2.metric("PEAK ACCEL", f"{peak:.3f} g")
m_col3.metric("KURTOSIS", f"{k_val:.2f}")
m_col4.metric("DOMINANT FREQ", f"{dom_freq:.1f} Hz")
m_col5.metric("ANOMALY SCORE", f"{anomaly_score:.3f}")

st.write("")

# ========================================================
# Matplotlib Visualization Engine
# ========================================================
plt.style.use("dark_background")
fig, axs = plt.subplots(2, 2, figsize=(12, 6), facecolor="#1E293B")

for ax in axs.flat:
    ax.set_facecolor("#0F172A")
    ax.tick_params(colors="#94A3B8", labelsize=8)
    ax.grid(True, color="#334155", linestyle="--", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color("#334155")

# 1. Time Signal Trace
display_samples = 2000
axs[0, 0].plot(
    analyzer.t[:display_samples],
    signal[:display_samples],
    color=status_color,
    linewidth=1,
)
axs[0, 0].set_title(
    f"Time-Domain Acceleration ({cond})", color="#F8FAFC", fontsize=10, pad=6
)
axs[0, 0].set_xlabel("Time (s)", color="#94A3B8", fontsize=8)
axs[0, 0].set_ylabel("Accel (g)", color="#94A3B8", fontsize=8)

# 2. STFT Spectrogram
f_stft, t_stft, Sxx = spectrogram(signal, analyzer.fs, nperseg=128)
axs[0, 1].pcolormesh(
    t_stft, f_stft, 10 * np.log10(Sxx + 1e-10), cmap="magma", shading="gouraud"
)
axs[0, 1].set_ylim(0, 200)
axs[0, 1].set_title(
    "STFT Spectrogram (Time-Frequency)", color="#F8FAFC", fontsize=10, pad=6
)
axs[0, 1].set_xlabel("Time (s)", color="#94A3B8", fontsize=8)
axs[0, 1].set_ylabel("Freq (Hz)", color="#94A3B8", fontsize=8)

# 3. FFT Frequency Spectrum
N = len(signal)
spectrum = np.abs(fft(signal))
frequencies = fftfreq(N, 1 / analyzer.fs)
mask = frequencies >= 0

axs[1, 0].plot(
    frequencies[mask], spectrum[mask], color="#38BDF8", linewidth=1
)
axs[1, 0].set_xlim(0, 200)
axs[1, 0].set_title(
    "Frequency Spectrum (FFT)", color="#F8FAFC", fontsize=10, pad=6
)
axs[1, 0].set_xlabel("Frequency (Hz)", color="#94A3B8", fontsize=8)
axs[1, 0].set_ylabel("Amplitude", color="#94A3B8", fontsize=8)

# 4. Anomaly Decision Index
categories = ["Normal Level", "Warning Threshold", "Current Run Score"]
scores = [0.05, -0.02, anomaly_score]
bar_colors = ["#10B981", "#F59E0B", status_color]

axs[1, 1].barh(categories, scores, color=bar_colors, height=0.5)
axs[1, 1].set_xlim(-0.25, 0.15)
axs[1, 1].set_title(
    "ML Anomaly Score Index", color="#F8FAFC", fontsize=10, pad=6
)
axs[1, 1].set_xlabel("Decision Score", color="#94A3B8", fontsize=8)

fig.tight_layout(pad=1.8)

# Render Matplotlib Figure in Streamlit
st.pyplot(fig)
