import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram
from scipy.stats import kurtosis
from sklearn.ensemble import IsolationForest
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class SmartChassisApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Smart Chassis Vibration Analysis System - Advanced Diagnostics")
        self.root.geometry("1350x880")
        self.root.configure(bg="#0F172A")  # Dark Slate Theme

        # ----------------------------------------------------
        # Simulation Parameters
        # ----------------------------------------------------
        self.fs = 1000  # Sampling frequency (Hz)
        self.duration = 5  # seconds
        self.t = np.arange(0, self.duration, 1 / self.fs)

        # Train ML model & initialize GUI
        self.model = self.train_model()
        self.create_gui()
        self.run_simulation("Normal")

    # ========================================================
    # Signal Generation Engine
    # ========================================================

    def generate_normal_signal(self):
        # Baseline vehicle harmonics (20Hz wheel/body, 45Hz engine)
        signal = 0.20 * np.sin(2 * np.pi * 20 * self.t) + 0.05 * np.sin(
            2 * np.pi * 45 * self.t
        )
        noise = 0.02 * np.random.randn(len(self.t))
        return signal + noise

    def generate_warning_signal(self):
        # Loose joint / slight unbalance (Moderate harmonic addition)
        signal = (
            0.22 * np.sin(2 * np.pi * 20 * self.t)
            + 0.08 * np.sin(2 * np.pi * 45 * self.t)
            + 0.12 * np.sin(2 * np.pi * 85 * self.t)
        )
        noise = 0.03 * np.random.randn(len(self.t))
        return signal + noise

    def generate_critical_signal(self):
        # Structural crack / severe impact transients (High frequency + transient bursts)
        signal = (
            0.25 * np.sin(2 * np.pi * 20 * self.t)
            + 0.10 * np.sin(2 * np.pi * 45 * self.t)
            + 0.35 * np.sin(2 * np.pi * 130 * self.t)
        )
        # Transient impact spikes (Micro-crack clicks)
        spikes = np.zeros(len(self.t))
        spike_indices = np.random.choice(len(self.t), size=15, replace=False)
        spikes[spike_indices] = np.random.choice([-0.8, 0.8], size=15)

        noise = 0.05 * np.random.randn(len(self.t))
        return signal + spikes + noise

    # ========================================================
    # Advanced Feature Extraction
    # ========================================================

    def extract_features(self, signal):
        # Time-domain statistical features
        rms = np.sqrt(np.mean(signal**2))
        peak = np.max(np.abs(signal))
        sig_kurtosis = kurtosis(signal)  # Sharpness of transient impacts
        crest_factor = peak / (rms + 1e-6)  # Peak to RMS ratio

        # Frequency-domain spectral features
        spectrum = np.abs(fft(signal))
        half = len(signal) // 2
        spectrum = spectrum[:half]
        frequencies = fftfreq(len(signal), 1 / self.fs)[:half]
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

    # ========================================================
    # Model Training Pipeline
    # ========================================================

    def train_model(self):
        training_features = []
        for _ in range(120):
            # Normal variation baseline signals
            sig = (
                0.20 * np.sin(2 * np.pi * 20 * self.t)
                + 0.05 * np.sin(2 * np.pi * 45 * self.t)
                + 0.02 * np.random.randn(len(self.t))
            )
            rms, peak, k_val, cf_val, dom_freq, freq_energy = (
                self.extract_features(sig)
            )

            training_features.append(
                [rms, peak, k_val, cf_val, dom_freq, freq_energy]
            )

        model = IsolationForest(contamination=0.08, random_state=42)
        model.fit(np.array(training_features))
        return model

    # ========================================================
    # GUI Construction
    # ========================================================

    def create_gui(self):
        # ----------------------------------------------------
        # Header Panel
        # ----------------------------------------------------
        header_frame = tk.Frame(self.root, bg="#1E293B", pady=12, padx=20)
        header_frame.pack(fill="x", side="top")

        title_label = tk.Label(
            header_frame,
            text="SMART CHASSIS VIBRATION DIAGNOSTICS",
            font=("Segoe UI", 16, "bold"),
            fg="#F8FAFC",
            bg="#1E293B",
        )
        title_label.pack(anchor="w")

        subtitle_label = tk.Label(
            header_frame,
            text="Time-Frequency STFT • Kurtosis Transient Detection • Multi-Stage Risk Analytics",
            font=("Segoe UI", 9),
            fg="#94A3B8",
            bg="#1E293B",
        )
        subtitle_label.pack(anchor="w", pady=(2, 0))

        # ----------------------------------------------------
        # Control & Multi-Stage Status Bar
        # ----------------------------------------------------
        control_frame = tk.Frame(self.root, bg="#0F172A", pady=12, padx=20)
        control_frame.pack(fill="x")

        # Simulation Control Buttons
        btn_normal = tk.Button(
            control_frame,
            text="Simulate NORMAL",
            font=("Segoe UI", 9, "bold"),
            bg="#10B981",
            fg="white",
            activebackground="#059669",
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2",
            command=lambda: self.run_simulation("Normal"),
        )
        btn_normal.pack(side="left", padx=(0, 8))

        btn_warn = tk.Button(
            control_frame,
            text="Simulate WARNING (Loose Joint)",
            font=("Segoe UI", 9, "bold"),
            bg="#F59E0B",
            fg="white",
            activebackground="#D97706",
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2",
            command=lambda: self.run_simulation("Warning"),
        )
        btn_warn.pack(side="left", padx=(0, 8))

        btn_critical = tk.Button(
            control_frame,
            text="Simulate CRITICAL (Crack)",
            font=("Segoe UI", 9, "bold"),
            bg="#EF4444",
            fg="white",
            activebackground="#DC2626",
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2",
            command=lambda: self.run_simulation("Critical"),
        )
        btn_critical.pack(side="left")

        # System Risk Level Indicator Badge
        self.status_container = tk.Frame(
            control_frame, bg="#1E293B", padx=12, pady=5
        )
        self.status_container.pack(side="right")

        self.telltale = tk.Label(
            self.status_container,
            text="●",
            font=("Segoe UI", 12),
            fg="#10B981",
            bg="#1E293B",
        )
        self.telltale.pack(side="left", padx=(0, 6))

        self.status_label = tk.Label(
            self.status_container,
            text="STATUS: NORMAL",
            font=("Segoe UI", 10, "bold"),
            fg="#F8FAFC",
            bg="#1E293B",
        )
        self.status_label.pack(side="left")

        # ----------------------------------------------------
        # Metric Feature Cards
        # ----------------------------------------------------
        metrics_frame = tk.Frame(self.root, bg="#0F172A", padx=20)
        metrics_frame.pack(fill="x", pady=(0, 10))

        self.rms_label = self.create_metric_card(
            metrics_frame, "RMS ACCELERATION", "g"
        )
        self.peak_label = self.create_metric_card(
            metrics_frame, "PEAK ACCEL", "g"
        )
        self.kurtosis_label = self.create_metric_card(
            metrics_frame, "KURTOSIS", ""
        )
        self.freq_label = self.create_metric_card(
            metrics_frame, "DOMINANT FREQ", "Hz"
        )
        self.severity_label = self.create_metric_card(
            metrics_frame, "DIAGNOSTIC LEVEL", "", is_status=True
        )

        # ----------------------------------------------------
        # Visual Plotting Canvas (2x2 Grid Architecture)
        # ----------------------------------------------------
        plot_frame = tk.Frame(self.root, bg="#1E293B", padx=8, pady=8)
        plot_frame.pack(fill="both", expand=True, padx=20, pady=(0, 15))

        self.figure = Figure(figsize=(12, 5), dpi=100, facecolor="#1E293B")

        # 1. Time Signal Plot
        self.time_ax = self.figure.add_subplot(221)
        # 2. STFT Spectrogram
        self.stft_ax = self.figure.add_subplot(222)
        # 3. FFT Frequency Spectrum
        self.freq_ax = self.figure.add_subplot(223)
        # 4. Multi-Stage Anomaly Decision Space
        self.score_ax = self.figure.add_subplot(224)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_metric_card(self, parent, title, unit, is_status=False):
        card = tk.Frame(parent, bg="#1E293B", bd=0, padx=12, pady=8)
        card.pack(side="left", expand=True, fill="x", padx=4)

        title_lbl = tk.Label(
            card,
            text=title,
            font=("Segoe UI", 8, "bold"),
            fg="#94A3B8",
            bg="#1E293B",
        )
        title_lbl.pack(anchor="w")

        val_frame = tk.Frame(card, bg="#1E293B")
        val_frame.pack(anchor="w", pady=(2, 0))

        val_lbl = tk.Label(
            val_frame,
            text="--",
            font=("Segoe UI", 14, "bold"),
            fg="#F8FAFC",
            bg="#1E293B",
        )
        val_lbl.pack(side="left")

        if unit:
            unit_lbl = tk.Label(
                val_frame,
                text=f" {unit}",
                font=("Segoe UI", 9),
                fg="#64748B",
                bg="#1E293B",
            )
            unit_lbl.pack(side="left", anchor="s", pady=(0, 1))

        return val_lbl

    # ========================================================
    # Simulation Execution & Multi-Stage Analysis Logic
    # ========================================================

    def run_simulation(self, condition):
        if condition == "Normal":
            signal = self.generate_normal_signal()
        elif condition == "Warning":
            signal = self.generate_warning_signal()
        else:
            signal = self.generate_critical_signal()

        # Extract features
        rms, peak, k_val, cf_val, dom_freq, freq_energy = (
            self.extract_features(signal)
        )

        features = np.array([[rms, peak, k_val, cf_val, dom_freq, freq_energy]])

        # Isolation Forest decision score (negative score = more anomalous)
        anomaly_score = self.model.decision_function(features)[0]

        # ----------------------------------------------------
        # Multi-Stage Fault Severity Decision Tree
        # ----------------------------------------------------
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

        # Update KPI UI display
        self.rms_label.config(text=f"{rms:.3f}")
        self.peak_label.config(text=f"{peak:.3f}")
        self.kurtosis_label.config(text=f"{k_val:.2f}")
        self.freq_label.config(text=f"{dom_freq:.1f}")

        self.severity_label.config(text=severity, fg=color)
        self.status_label.config(text=status_text, fg=color)
        self.telltale.config(fg=color)

        self.update_graphs(signal, condition, anomaly_score, color)

    # ========================================================
    # Multi-Panel Matplotlib Rendering Pipeline
    # ========================================================

    def update_graphs(self, signal, condition, anomaly_score, status_color):
        # Clear previous plots
        for ax in (self.time_ax, self.stft_ax, self.freq_ax, self.score_ax):
            ax.clear()
            ax.set_facecolor("#0F172A")
            ax.tick_params(colors="#94A3B8", labelsize=7)
            ax.grid(True, color="#334155", linestyle="--", alpha=0.5)
            for spine in ax.spines.values():
                spine.set_color("#334155")

        # 1. Time Signal Trace
        display_samples = 2000
        self.time_ax.plot(
            self.t[:display_samples],
            signal[:display_samples],
            color=status_color,
            linewidth=1,
        )
        self.time_ax.set_title(
            f"Time-Domain Acceleration ({condition})",
            color="#F8FAFC",
            fontsize=9,
            pad=6,
        )
        self.time_ax.set_xlabel("Time (s)", color="#94A3B8", fontsize=7)
        self.time_ax.set_ylabel("Accel (g)", color="#94A3B8", fontsize=7)

        # 2. STFT Spectrogram (Time-Frequency Analysis)
        f_stft, t_stft, Sxx = spectrogram(signal, self.fs, nperseg=128)
        self.stft_ax.pcolormesh(
            t_stft, f_stft, 10 * np.log10(Sxx + 1e-10), cmap="magma", shading="gouraud"
        )
        self.stft_ax.set_ylim(0, 200)
        self.stft_ax.set_title(
            "STFT Spectrogram (Time-Frequency)",
            color="#F8FAFC",
            fontsize=9,
            pad=6,
        )
        self.stft_ax.set_xlabel("Time (s)", color="#94A3B8", fontsize=7)
        self.stft_ax.set_ylabel("Freq (Hz)", color="#94A3B8", fontsize=7)

        # 3. FFT Frequency Spectrum
        N = len(signal)
        spectrum = np.abs(fft(signal))
        frequencies = fftfreq(N, 1 / self.fs)
        mask = frequencies >= 0

        self.freq_ax.plot(
            frequencies[mask], spectrum[mask], color="#38BDF8", linewidth=1
        )
        self.freq_ax.set_xlim(0, 200)
        self.freq_ax.set_title(
            "Frequency Spectrum (FFT)", color="#F8FAFC", fontsize=9, pad=6
        )
        self.freq_ax.set_xlabel("Frequency (Hz)", color="#94A3B8", fontsize=7)
        self.freq_ax.set_ylabel("Amplitude", color="#94A3B8", fontsize=7)

        # 4. Multi-Stage Anomaly Risk Gauge
        categories = ["Normal Level", "Warning Threshold", "Current Run Score"]
        scores = [0.05, -0.02, anomaly_score]
        bar_colors = ["#10B981", "#F59E0B", status_color]

        self.score_ax.barh(categories, scores, color=bar_colors, height=0.5)
        self.score_ax.set_xlim(-0.25, 0.15)
        self.score_ax.set_title(
            "ML Anomaly Score Index", color="#F8FAFC", fontsize=9, pad=6
        )
        self.score_ax.set_xlabel("Decision Score", color="#94A3B8", fontsize=7)

        self.figure.tight_layout(pad=1.8)
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = SmartChassisApp(root)
    root.mainloop()