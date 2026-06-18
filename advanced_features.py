# =============================================================================
# OPTIONAL ADVANCED FEATURES
# SAFE TO ADD / REMOVE
# =============================================================================
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run_advanced_features(
    t,
    pwm,
    output,
    dt,
    device,
    frequency,
    duty_cycle,
    time_window,
    metrics,
    VMAX,
    generate_pwm_signal,
    get_device_response
):
st.markdown("---")
st.header("🚀 Advanced Optional Features")

# =============================================================================
# FEATURE TOGGLES
# =============================================================================

enable_live_monitor = st.checkbox(
    "📡 Live PWM Monitor"
)

enable_frequency_sweep = st.checkbox(
    "📈 Frequency Sweep Analysis"
)

enable_harmonics = st.checkbox(
    "🎵 Harmonic Spectrum"
)

enable_efficiency = st.checkbox(
    "⚡ Power Efficiency Estimator"
)

enable_comparison = st.checkbox(
    "🔬 Compare Multiple Duty Cycles"
)

enable_noise = st.checkbox(
    "🌊 Add Signal Noise"
)

enable_3d_view = st.checkbox(
    "🧊 3D Surface View"
)

enable_dark_panel = st.checkbox(
    "🖥 Engineering Dashboard"
)

# =============================================================================
# LIVE PWM MONITOR
# =============================================================================

if enable_live_monitor:

    st.subheader("📡 Live PWM Monitor")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Frequency",
        f"{frequency} Hz"
    )

    col2.metric(
        "Duty Cycle",
        f"{duty_cycle}%"
    )

    col3.metric(
        "Average Voltage",
        f"{np.mean(pwm):.2f} V"
    )

    progress = duty_cycle / 100

    st.progress(progress)

# =============================================================================
# FREQUENCY SWEEP ANALYSIS
# =============================================================================

if enable_frequency_sweep:

    st.subheader("📈 Frequency Sweep")

    sweep_freqs = np.logspace(1, 5, 60)

    sweep_outputs = []

    for f in sweep_freqs:

        _, pwm_sweep, dt_sweep = generate_pwm_signal(
            duty_cycle,
            f,
            0.02
        )

        out = get_device_response(
            device,
            pwm_sweep,
            dt_sweep
        )

        ripple = np.std(out)

        sweep_outputs.append(ripple)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.semilogx(
        sweep_freqs,
        sweep_outputs,
        linewidth=2
    )

    ax.set_title(
        f"{device.capitalize()} Ripple vs Frequency"
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Ripple")

    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

# =============================================================================
# HARMONIC ANALYSIS
# =============================================================================

if enable_harmonics:

    st.subheader("🎵 Harmonic Spectrum")

    fft = np.fft.fft(output)

    freqs = np.fft.fftfreq(
        len(output),
        d=dt
    )

    positive = freqs > 0

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        freqs[positive],
        np.abs(fft[positive]),
        linewidth=1.5
    )

    ax.set_xlim(0, frequency * 10)

    ax.set_title(
        "FFT Spectrum"
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")

    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

# =============================================================================
# POWER EFFICIENCY ESTIMATION
# =============================================================================

if enable_efficiency:

    st.subheader("⚡ Power Efficiency")

    avg_voltage = np.mean(output)

    efficiency = (
        avg_voltage / VMAX
    ) * 100

    heat_loss = 100 - efficiency

    col1, col2 = st.columns(2)

    col1.metric(
        "Estimated Efficiency",
        f"{efficiency:.1f}%"
    )

    col2.metric(
        "Estimated Loss",
        f"{heat_loss:.1f}%"
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(
        ["Useful Power", "Loss"],
        [efficiency, heat_loss]
    )

    ax.set_ylim(0, 100)

    ax.set_ylabel("Percent")

    st.pyplot(fig)

# =============================================================================
# MULTI DUTY CYCLE COMPARISON
# =============================================================================

if enable_comparison:

    st.subheader("🔬 Duty Cycle Comparison")

    compare_values = [20, 50, 80]

    fig, ax = plt.subplots(figsize=(12, 5))

    for d in compare_values:

        t_cmp, pwm_cmp, dt_cmp = generate_pwm_signal(
            d,
            frequency,
            time_window
        )

        out_cmp = get_device_response(
            device,
            pwm_cmp,
            dt_cmp
        )

        ax.plot(
            t_cmp,
            out_cmp,
            label=f"{d}% Duty"
        )

    ax.set_title(
        f"{device.capitalize()} Comparison"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Output")

    ax.grid(True, alpha=0.3)

    ax.legend()

    st.pyplot(fig)

# =============================================================================
# SIGNAL NOISE INJECTION
# =============================================================================

if enable_noise:

    st.subheader("🌊 PWM with Noise")

    noise_strength = st.slider(
        "Noise Strength",
        0.0,
        2.0,
        0.3
    )

    noisy_pwm = pwm + np.random.normal(
        0,
        noise_strength,
        len(pwm)
    )

    noisy_output = get_device_response(
        device,
        noisy_pwm,
        dt
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        t,
        noisy_pwm,
        alpha=0.5,
        label="Noisy PWM"
    )

    ax.plot(
        t,
        noisy_output,
        linewidth=2,
        label="Device Response"
    )

    ax.set_title(
        "Noise Response Analysis"
    )

    ax.grid(True, alpha=0.3)

    ax.legend()

    st.pyplot(fig)

# =============================================================================
# 3D SURFACE VIEW
# =============================================================================

if enable_3d_view:

    from mpl_toolkits.mplot3d import Axes3D

    st.subheader("🧊 3D PWM Surface")

    duty_vals = np.linspace(0, 100, 30)
    freq_vals = np.linspace(10, 5000, 30)

    D, F = np.meshgrid(
        duty_vals,
        freq_vals
    )

    Z = (D / 100) * np.log10(F)

    fig = plt.figure(figsize=(10, 6))

    ax = fig.add_subplot(
        111,
        projection='3d'
    )

    ax.plot_surface(
        D,
        F,
        Z
    )

    ax.set_xlabel("Duty Cycle")
    ax.set_ylabel("Frequency")
    ax.set_zlabel("Response")

    st.pyplot(fig)

# =============================================================================
# ENGINEERING DASHBOARD PANEL
# =============================================================================

if enable_dark_panel:

    st.subheader("🖥 Engineering Status Panel")

    st.code(f"""
SYSTEM STATUS REPORT
---------------------
DEVICE            : {device.upper()}
PWM FREQUENCY     : {frequency} Hz
DUTY CYCLE        : {duty_cycle} %
TIME WINDOW       : {time_window} s
MEAN OUTPUT       : {metrics['mean']:.2f}
RMS OUTPUT        : {metrics['rms']:.2f}
MAX OUTPUT        : {metrics['max']:.2f}
MIN OUTPUT        : {metrics['min']:.2f}

SYSTEM MODE       : ACTIVE
PWM ENGINE        : RUNNING
GRAPH ENGINE      : ONLINE
AI ASSISTANT      : READY
""")

# =============================================================================
# END OF OPTIONAL FEATURES
# =============================================================================
