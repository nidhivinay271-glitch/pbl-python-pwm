"""
PWM Signal Simulator Dashboard
A web-based dashboard to simulate PWM signals and visualize their effects on real-world physical systems.
"""

import streamlit as st
import numpy as np
import time

# Attempt to import Matplotlib, handle missing case
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

# Optional AI Imports
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None
    cosine_similarity = None
    _HAS_SENTENCE_TRANSFORMERS = False

# Fallback values
VMAX = 5.0
DEFAULT_FREQUENCY = 1000
DEFAULT_DUTY_CYCLE = 50
DEFAULT_TIME_WINDOW = 0.05

# ============================================================================
# APP CONFIG
# ============================================================================
st.set_page_config(
    page_title="⚡ Real-Time PWM WorkBench",
    page_icon="⚡",
    layout="wide"
)

# Initialize Session State
if "duty_cycle" not in st.session_state:
    st.session_state.duty_cycle = DEFAULT_DUTY_CYCLE
if "frequency" not in st.session_state:
    st.session_state.frequency = DEFAULT_FREQUENCY
if "device_params" not in st.session_state:
    st.session_state.device_params = {}

# ============================================================================
# UNIFIED CSS DESIGN
# ============================================================================
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .dashboard-panel, .waveform-panel, .actions-panel, .metrics-panel {
        background: #1a1a1a;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #333;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .panel-header { font-size: 20px; font-weight: bold; margin-bottom: 15px; border-bottom: 1px solid #333; padding-bottom: 5px; }
    .infographic-container { text-align: center; padding: 10px; }
    .led-orb { width: 100px; height: 100px; border-radius: 50%; margin: 0 auto; transition: box-shadow 0.1s; }
    .gear-spin { font-size: 100px; transform-origin: center; animation: spin linear infinite; }
    .heater-element { font-size: 100px; font-weight: bold; transition: color 0.1s; }
    .sound-pulse { font-size: 100px; animation: pulse linear infinite; }
    @keyframes spin { 100% { transform: rotate(360deg); } }
    @keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.1); } }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PWM GENERATION ENGINE
# ============================================================================
@st.cache_data
def generate_pwm_signal(duty_cycle, frequency, time_duration_s):
    frequency = max(1.0, float(frequency))
    time_duration_s = max(0.001, float(time_duration_s))
    
    duty = np.clip(duty_cycle / 100.0, 0.0, 1.0)
    period = 1.0 / frequency
    
    samples_per_cycle = 150
    total_periods = time_duration_s / period
    total_samples = int(np.ceil(total_periods * samples_per_cycle))
    total_samples = np.clip(total_samples, 200, 50000)
    
    t = np.linspace(0, time_duration_s, total_samples, endpoint=False)
    phase = np.mod(t, period)
    pwm = np.where(phase < duty * period, VMAX, 0.0)
    
    dt = t[1] - t[0] if len(t) > 1 else time_duration_s
    return t, pwm, dt

# ============================================================================
# DEVICE MODELS (PHYSICS-BASED)
# ============================================================================
def first_order_filter(vin, dt, tau):
    y = np.zeros_like(vin, dtype=float)
    alpha = 1.0 - np.exp(-dt / tau) if tau > 0 else 1.0
    for i in range(1, len(vin)):
        y[i] = y[i - 1] + alpha * (vin[i] - y[i - 1])
    return y

def get_device_response(device, vin, dt, params):
    dt_float = float(dt)
    y = np.zeros_like(vin, dtype=float)

    if device == "Capacitor (RC Filter)":
        R, C = params.get("R", 1000.0), params.get("C", 10e-6)
        y = first_order_filter(vin, dt_float, R*C)
        
    elif device == "Inductor (RL Circuit)":
        R, L = params.get("R", 3.0), params.get("L", 10e-3)
        current = np.zeros_like(vin)
        for i in range(1, len(vin)):
            di = (vin[i-1] - R * current[i-1]) / L * dt_float
            current[i] = max(current[i-1] + di, 0.0)
        y = current * R

    elif device == "LED":
        Vf = params.get("Vf", 2.0)
        y = np.where(vin > Vf, ((vin - Vf) / (VMAX - Vf)) * VMAX, 0.0)

    elif device == "Motor":
        Ke, Kt, B, J = params.get("Ke", 0.01), params.get("Kt", 0.01), params.get("B", 0.00001), params.get("J", 0.0001)
        current, speed = np.zeros_like(vin), np.zeros_like(vin)
        for i in range(1, len(vin)):
            di = (vin[i-1] - 2.0 * current[i-1] - Ke*speed[i-1]) / 0.001 * dt_float
            current[i] = max(current[i-1] + di, 0.0)
            domega = (Kt*current[i] - B*speed[i-1]) / J * dt_float
            speed[i] = max(speed[i-1] + domega, 0.0)
        y = (speed / (VMAX / Ke)) * VMAX

    elif device == "Heater":
        tau_th = params.get("tau_th", 10.0)
        temp = np.full_like(vin, params.get("ambient_temp", 25.0))
        alpha = 1.0 - np.exp(-dt_float / tau_th)
        for i in range(1, len(vin)):
            power = (vin[i] ** 2) / 12.0
            temp[i] = temp[i-1] + alpha * (power * 5.0 + temp[i-1] - temp[i-1])
        y = temp

    elif device == "Buzzer":
        threshold = params.get("threshold", 2.5)
        tone = np.where(vin > threshold, 1.0, 0.0)
        y = first_order_filter(tone, dt_float, 0.05) * VMAX

    elif device == "Diode":
        y = np.where(vin > params.get("Vf", 0.7), vin - params.get("Vf", 0.7), 0.0)

    elif device == "Transistor":
        target = np.where(vin > params.get("Vth", 1.2), VMAX, 0.0)
        y = first_order_filter(target, dt_float, 0.00008)

    return np.clip(y, 0, VMAX)

# ============================================================================
# SIDEBAR: WorkBench Controls
# ============================================================================
st.sidebar.header("⚡ PWM Controls")

with st.sidebar.expander("Main Signal Parameters", expanded=True):
    frequency = st.slider("Frequency (Hz)", 1, 20000, DEFAULT_FREQUENCY)
    duty_cycle = st.slider("Duty Cycle (%)", 0, 100, DEFAULT_DUTY_CYCLE)
    selected_device = st.selectbox("Simulate Device", ["LED", "Capacitor (RC Filter)", "Inductor (RL Circuit)", "Motor", "Heater", "Buzzer", "Diode", "Transistor"])
    
with st.sidebar.expander("Advanced Simulation Settings", expanded=False):
    time_window = st.slider("Time Window (s)", 0.01, 30.0, 0.05, step=0.01)
    graph_mode = st.selectbox("Advanced Graph View", ["Both (Input vs Output)", "PWM Only", "Device Response Only", "Separate Subplots"])
    
if time_window < 0.1 and selected_device == "Heater": st.sidebar.warning("Increase time window to see thermal response.")
elif time_window < 0.05 and selected_device == "Motor": st.sidebar.warning("Increase time window to see motor inertia.")

with st.sidebar.expander("🛠️ Device Parameters", expanded=False):
    params = st.session_state.device_params
    if selected_device == "LED": params["Vf"] = st.slider("Forward Voltage Vf (V)", 1.2, 3.6, 2.0)
    elif selected_device == "Capacitor (RC Filter)":
        params["R"] = st.slider("Resistance (Ohm)", 100.0, 10000.0, 1000.0)
        params["C"] = st.slider("Capacitance (uF)", 0.1, 1000.0, 10.0) * 1e-6
    elif selected_device == "Inductor (RL Circuit)":
        params["R"] = st.slider("Resistance (Ohm)", 1.0, 50.0, 3.0)
        params["L"] = st.slider("Inductance (mH)", 1.0, 200.0, 10.0) * 1e-3
    elif selected_device == "Motor":
        params["Ke"] = st.slider("Ke (Back EMF)", 0.005, 0.05, 0.01)
        params["B"] = st.slider("B (Viscous Fric.)", 0.000001, 0.0001, 0.00001)
    elif selected_device == "Heater":
        params["tau_th"] = st.slider("Thermal Time Constant (s)", 1.0, 120.0, 10.0)
    elif selected_device == "Buzzer": params["threshold"] = st.slider("Buzzer Vth (V)", 0.5, 4.5, 2.5)

st.session_state.device_params = params

if st.sidebar.button("System Reset"): st.experimental_rerun()

# ============================================================================
# MAIN PAGE
# ============================================================================
st.title("⚡ PWM WorkBench: Direct Dynamics")

col_main, col_visual = st.columns([2, 1], gap="medium")

# Run Simulation
t, pwm, dt = generate_pwm_signal(duty_cycle, frequency, time_window)
output = get_device_response(selected_device, pwm, dt, params)
normalized_output = (output - np.min(output)) / (np.max(output) - np.min(output) + 1e-9)

# ----------------------------------------------------------------------------
# COLUMN 1: WAVEFORM AND METRICS
# ----------------------------------------------------------------------------
with col_main:
    metrics_block = st.empty()
    
    with st.container():
        st.markdown(f'<div class="waveform-panel"><div class="panel-header">{selected_device} vs PWM</div>', unsafe_allow_html=True)
        if plt and matplotlib:
            fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
            if graph_mode == "Both (Input vs Output)":
                ax.plot(t, pwm, linestyle="--", alpha=0.6, label="PWM Input (V)", color="#666")
                ax.set_ylabel("PWM Logic Level (V)")
                ax_twin = ax.twinx()
                color = "#ff7f0e"
                label = f"{selected_device.split('(')[0].strip()} Output"
                if selected_device == "Heater": label += " (°C)"; color="#d62728"
                elif selected_device == "Motor": label += " (Scaled RPM)"; color="#1f77b4"
                ax_twin.plot(t, output, linewidth=2, color=color, label=label)
                ax_twin.set_ylabel(label, color=color)
                fig.legend(loc="upper right")
            elif graph_mode == "PWM Only":
                ax.plot(t, pwm, linestyle="--", linewidth=2, color="blue", label="PWM Input")
            elif graph_mode == "Device Response Only":
                ax.plot(t, output, linewidth=2.5, color="orange", label="Output")
            ax.set_xlabel("Time (s)")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    metrics = { "Mean": np.mean(output), "RMS": np.sqrt(np.mean(output**2)), "Min": np.min(output), "Max": np.max(output) }
    units = "V" if selected_device not in ["Heater", "Motor"] else ("°C" if selected_device == "Heater" else "RPM Level")
    metrics_html = f'<div class="metrics-panel"><div class="panel-header">{selected_device} Unified Metrics</div><div style="display:flex; justify-content:space-around;">'
    for k, v in metrics.items(): metrics_html += f'<div><strong>{k}:</strong> {v:.2f}{units}</div>'
    metrics_block.markdown(metrics_html + "</div></div>", unsafe_allow_html=True)
        
# ----------------------------------------------------------------------------
# COLUMN 2: LIVE DEVICE DYNAMICS & INFOGRAPHICS
# ----------------------------------------------------------------------------
with col_visual:
    st.markdown('<div class="dashboard-panel"><div class="panel-header">Device Dynamics Dashboard</div>', unsafe_allow_html=True)
    anim_placeholder = st.empty()
    v = normalized_output[-1] 
    
    state_color = "f1c40f" if selected_device != "Heater" else "d62728"
    state_class = "state-on" if v > 0.05 else "state-off"
    glow_param = int(10 + v * 40) if v > 0.1 else 0
    
    if selected_device == "LED":
        html = f"""<div class="infographic-container">
            <div style="width:100px; height:100px; margin:0 auto; padding:10px;">
                <div class="led-orb" style="background-color:#{state_color}; box-shadow: 0 0 {glow_param}px {glow_param//2}px #{state_color};"></div>
            </div>
            <div style="font-weight:bold; color:#{state_color}; font-size:24px; margin-top:10px;">Glow: {int(v*100)}%</div></div>"""
    elif selected_device == "Motor":
        duration = 5.0 / (v + 0.1)
        html = f"""<div class="infographic-container">
            <div class="gear-spin" style="animation-duration:{duration:.2f}s;">⚙️</div>
            <div style="font-weight:bold; font-size:24px;">SPEED: {int(v*100)}%</div></div>"""
    elif selected_device == "Heater":
        hue = int(12 * (1.0 - v))
        html = f"""<div class="infographic-container">
            <div class="heater-element" style="color:rgb({int(100+v*155)}, 30, {hue});">🔥</div>
            <div style="font-weight:bold; font-size:24px;">TEMP: {output[-1]:.1f}°C</div></div>"""
    elif selected_device == "Capacitor (RC Filter)":
        charge_dots = "".join([f'<div style="width:{max(1, int(v*10))}px; height:10px; background:#00ff87; border-radius:50%; margin:3px;"></div>' for _ in range(10)])
        html = f"""<div class="infographic-container">
            <div style="width:150px; height:80px; margin:0 auto; display:flex; flex-direction:column; justify-content:center;">
                <div style="width:100%; height:20px; background:#444;"></div>
                <div style="width:100%; height:40px; background:#222; display:flex; flex-wrap:wrap; justify-content:center; padding:2px;">{charge_dots}</div>
                <div style="width:100%; height:20px; background:#444;"></div>
            </div>
            <div style="font-weight:bold; font-size:24px;">CHARGE: {int(v*100)}%</div></div>"""
    elif selected_device == "Inductor (RL Circuit)":
        flux_lines = "".join([f'<div style="font-size:{max(20, int(30+v*10))}px; color:#b224ef;">➰</div>' for _ in range(5)])
        html = f"""<div class="infographic-container">
            <div style="width:150px; height:80px; margin:0 auto; display:flex; justify-content:center; align-items:center;">{flux_lines}</div>
            <div style="font-weight:bold; font-size:24px;">FLUX: {int(v*100)}%</div></div>"""
    elif selected_device == "Buzzer":
        html = f"""<div class="infographic-container">
            <div style="position:relative; width:150px; height:100px; margin:0 auto;">
                <div class="sound-pulse" style="font-size:80px;">🔊</div>
            </div>
            <div style="font-weight:bold; font-size:24px;">SOUND: {int(v*100)}%</div></div>"""
    elif selected_device == "Diode":
        state = "CONDUCTING ✅" if v > 0.2 else "BLOCKING ❌"
        color = "#00ff87" if v > 0.2 else "#ff003c"
        html = f"""<div class="infographic-container">
            <div style="font-size:80px;">➡️</div>
            <div style="font-weight:bold; color:{color}; font-size:24px;">{state}</div></div>"""
    elif selected_device == "Transistor":
        state = "ON 🟢" if v > 0.5 else "OFF 🔴"
        color = "#00ff87" if v > 0.5 else "#ff003c"
        html = f"""<div class="infographic-container">
            <div style="font-size:80px;">🔀</div>
            <div style="font-weight:bold; color:{color}; font-size:24px;">STATE: {state}</div></div>"""
    
    anim_placeholder.markdown(html, unsafe_allow_html=True)
    if st.button("Update Infographic"): pass
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# OPTIONAL DIAGNOSTICS BLOCK 
# =============================================================================
st.markdown("---")
with st.expander("🚀 WorkBench Advanced Diagnostics Panel", expanded=False):
    diagnostic_tool = st.selectbox("Select Diagnostic Tool", ["None", "FFT (Frequency Analyzer)", "Efficiency Calculator", "Circuit Diagram Panel"])
    if diagnostic_tool == "FFT (Frequency Analyzer)":
        if plt and matplotlib:
            fft = np.fft.fft(output - np.mean(output))
            freqs = np.fft.fftfreq(len(output), d=dt)
            positive = freqs > 0
            fig_fft, ax_fft = plt.subplots(figsize=(10, 4), dpi=100)
            ax_fft.plot(freqs[positive], np.abs(fft[positive]), linewidth=1.5, color="#b224ef")
            ax_fft.set_xlim(0, frequency * 10)
            ax_fft.set_title(f"Frequency Spectrum: {selected_device} Response")
            ax_fft.set_xlabel("Frequency (Hz)")
            ax_fft.set_ylabel("Normalized Magnitude")
            st.pyplot(fig_fft, use_container_width=True)
    elif diagnostic_tool == "Efficiency Calculator":
        efficiency = (np.mean(output) / VMAX) * 100
        ripple = ((np.max(output) - np.min(output)) / VMAX) * 100
        c1, c2 = st.columns(2)
        c1.metric("Logic conversion Efficiency", f"{efficiency:.1f}%")
        c2.metric("Response Ripple (Peak-to-Peak)", f"{ripple:.1f}%")
    elif diagnostic_tool == "Circuit Diagram Panel":
        diagrams = {
            "LED": "PWM Pin ---- Resistor ---- LED ---- GND",
            "Motor": "PWM Pin ---- MOSFET ---- Motor ---- Supply\n                     |\n                    Diode",
            "Capacitor (RC Filter)": "PWM ---- RC Filter ---- Output\n           |\n       Capacitor\n           |\n          GND",
            "Inductor (RL Circuit)": "PWM ---- Inductor ---- Load",
            "Heater": "PWM ---- MOSFET ---- Heater",
            "Transistor": "PWM ---- Base Resistor ---- Transistor"
        }
        st.code(diagrams.get(selected_device, "Circuit not available"))

# =============================================================================
# AI CHATBOT & KNOWLEDGE BASE
# =============================================================================

KNOWLEDGE_BASE = [
    {
        "topic": "pwm general",
        "text": """### ⚡ Pulse Width Modulation (PWM) Theory
**Pulse Width Modulation (PWM)** is a technique used to encode a continuous analog value into a pulsing digital signal. Because microcontrollers (like Arduino) cannot output varying voltages natively (they only output 0V or 5V), they rapidly switch the pin ON and OFF.
* **How it works:** By changing how long the signal stays HIGH versus LOW, the *average* voltage delivered to the load changes. 
* **Formula:** $V_{avg} = Duty\\_Cycle \\times V_{max}$
* **Why use it?** It is highly efficient. A transistor driving a PWM signal is either fully ON (near zero resistance) or fully OFF (zero current), meaning very little power is wasted as heat compared to using a linear resistor to drop voltage."""
    },
    {
        "topic": "duty cycle",
        "text": """### ⏱️ Duty Cycle Explained
The **Duty Cycle** is the percentage of one period in which a signal is active (HIGH).
* **0% Duty Cycle:** The signal is completely flat (0V). The device is OFF.
* **50% Duty Cycle:** The signal is HIGH half the time and LOW half the time. The average voltage is half of the maximum (e.g., 2.5V on a 5V system).
* **100% Duty Cycle:** The signal is a constant DC voltage (5V). The device is fully ON.
**In this simulator:** Adjusting the duty cycle slider directly alters the area under the curve of the blue PWM input graph, scaling the final output state of every device."""
    },
    {
        "topic": "frequency and time window",
        "text": """### 🌊 Frequency & Simulation Time
**Frequency (Hz)** dictates how many times per second the PWM cycle repeats. 
* $Frequency = 1 / Period$. A 1000 Hz signal repeats 1,000 times a second.
* **Device Reaction to Frequency:** Electrical devices (Capacitors, Inductors) react to frequency aggressively. High frequencies result in smooth DC-like outputs. Low frequencies result in heavy "ripple."
* **Mechanical/Thermal Devices:** Motors and heaters are so slow (high inertia) that even a low frequency (10 Hz) is smoothed out naturally by the physical mass of the device.

**Time Window:** This slider is critical for viewing different physics. Electrical transients happen in milliseconds (set to 0.05s to see them). Thermal changes take minutes (set to 10.0s to see a heater warm up)."""
    },
    {
        "topic": "motor",
        "text": """### ⚙️ DC Motor Physics
A DC Motor converts electrical PWM energy into mechanical rotation. 
* **The Physics:** The motor has **electrical inertia** (Inductance, $L$) which smooths the current spikes, and **mechanical inertia** (Rotor Mass, $J$) which smooths the physical speed.
* **Back-EMF:** As the motor spins, it acts like a generator, creating a reverse voltage ($K_e \\omega$) that opposes the supply voltage. 
* **Why PWM?** PWM is ideal for motors because it provides full 5V/12V "kicks" of torque, preventing the motor from stalling at low speeds, which often happens if you just lower a pure analog voltage."""
    },
    {
        "topic": "capacitor rc filter",
        "text": """### 🔋 Capacitor & RC Filters
A Capacitor stores electrical energy in an electric field. When paired with a Resistor, it forms an **RC Low-Pass Filter**.
* **The Physics:** Capacitors oppose sudden changes in voltage. When the PWM goes HIGH, the capacitor charges exponentially. When it goes LOW, it discharges. 
* **Time Constant ($\\tau = R \\times C$):** This dictates how fast the capacitor charges. It takes roughly $5\\tau$ to fully charge.
* **Smoothing:** If the PWM frequency is fast enough (Period $\\ll \\tau$), the capacitor never fully charges or discharges. Instead, it averages the PWM into a clean, flat DC voltage with a tiny "ripple" riding on top."""
    },
    {
        "topic": "inductor rl circuit",
        "text": """### 🌀 Inductor & RL Circuits
An Inductor stores energy in a magnetic field. 
* **The Physics:** Inductors strictly oppose sudden changes in **current** (unlike capacitors, which oppose voltage changes). 
* **Behavior:** When the PWM turns ON, the inductor initially blocks the current, allowing it to ramp up slowly. When the PWM turns OFF, the magnetic field collapses, forcing current to keep flowing.
* **Practical Use:** This property is the foundational principle of "Buck Converters" and switch-mode power supplies. It smooths chopping PWM current into a steady flow."""
    },
    {
        "topic": "heater",
        "text": """### 🔥 Resistive Heaters
A Heater converts electrical power into thermal energy via Joule Heating ($P = V^2 / R$).
* **Thermal Inertia:** Heaters possess massive thermal mass. The time constant ($\\tau_{th}$) is usually measured in tens of seconds or minutes. 
* **PWM Effect:** Because heaters are so slow to react, the PWM frequency doesn't need to be fast. Industrial heaters are often controlled by Solid State Relays (SSRs) switching ON and OFF just once every few seconds (e.g., 0.5 Hz). The ambient environment acts as a natural low-pass filter, resulting in a perfectly stable target temperature."""
    },
    {
        "topic": "led brightness",
        "text": """### 💡 LED (Light Emitting Diode)
LEDs are semiconductors that emit light when forward-biased.
* **Forward Voltage ($V_f$):** An LED will not turn on until the voltage exceeds its threshold (usually ~2.0V for Red/Green, ~3.0V for Blue/White).
* **Current Limiting:** A series resistor is mandatory to prevent the LED from drawing infinite current and burning out once $V_f$ is crossed.
* **PWM Dimming:** LEDs react to electricity almost instantly (microseconds). If you apply a 50% PWM, the LED is actually flashing ON and OFF at full brightness. However, due to human **Persistence of Vision**, any frequency above ~100 Hz blends together, and our brains perceive it as the LED being exactly 50% as bright!"""
    },
    {
        "topic": "diode",
        "text": """### ➡️ Standard Diode
A Diode is the electrical equivalent of a one-way check valve.
* **Rectification:** It only allows current to flow from Anode to Cathode. If the voltage reverses, it blocks the current.
* **Voltage Drop:** Nothing is free. Passing through the P-N junction "costs" energy, resulting in a continuous voltage drop of about $0.7V$ across the diode. In the simulator, you will notice the output peak is exactly 0.7V lower than the input PWM peak."""
    },
    {
        "topic": "zener diode",
        "text": """### ⚡ Zener Diode & Clamping
While standard diodes block reverse current, Zener diodes are specifically engineered to safely break down and conduct backwards when a specific **Zener Voltage ($V_z$)** is reached.
* **Voltage Regulation:** If you apply a 5V PWM signal to a 3.3V Zener diode in reverse bias, it will "clamp" the top of the waveform, cleanly chopping the voltage off at 3.3V. This is heavily used to protect sensitive 3.3V microcontrollers from 5V logic signals."""
    },
    {
        "topic": "transistor switch",
        "text": """### 🔀 Transistor (BJT/MOSFET)
Transistors act as digital amplifiers or electronic switches.
* **Threshold Voltage:** The transistor will remain OFF until the base/gate voltage crosses a specific threshold (e.g., 1.2V). 
* **PWM Amplification:** Microcontrollers can only output tiny amounts of current (~20mA). By feeding the microcontroller's PWM signal into the Gate of a MOSFET, the transistor acts as a massive electronic switch, perfectly mirroring the PWM signal but allowing massive currents (like 10 Amps) to flow from a heavy power supply to a motor."""
    },
    {
        "topic": "buzzer piezo",
        "text": """### 🔊 Piezo Buzzer
A buzzer converts PWM signals into acoustic pressure waves.
* **Frequency = Pitch:** The frequency of the PWM directly controls the pitch (note) of the sound. 1000 Hz sounds like a high-pitched beep, while 200 Hz sounds like a low buzz.
* **Duty Cycle = Volume:** To get the maximum volume out of a buzzer, you apply a 50% duty cycle. This allows the internal piezo crystal to flex outward for exactly half the cycle and snap inward for the other half, moving the maximum amount of air."""
    }
]

_kb_texts = [x["text"] for x in KNOWLEDGE_BASE]

@st.cache_resource
def load_model():
    if not _HAS_SENTENCE_TRANSFORMERS: return None
    return SentenceTransformer("all-MiniLM-L6-v2")

_model = load_model()

@st.cache_resource
def load_embeddings():
    if _model is None: return None
    return _model.encode(_kb_texts, normalize_embeddings=True)

_kb_embeddings = load_embeddings()

def get_chat_response(query):
    if _model is None or _kb_embeddings is None or cosine_similarity is None:
        return ("⚠️ **Semantic AI features are currently offline.** \n\n"
                "To enable the AI tutor, ensure you have installed `sentence-transformers` and `scikit-learn` "
                "in your requirements.txt file.")
    q_emb = _model.encode([query], normalize_embeddings=True)
    scores = cosine_similarity(q_emb, _kb_embeddings)[0]
    idx = int(np.argmax(scores))
    if scores[idx] < 0.2:
        return ("🤔 I'm not entirely sure about that. Try asking me about Duty cycle, frequency, or specific device physics!")
    return KNOWLEDGE_BASE[idx]["text"]

st.markdown("---")
st.subheader("🤖 PWM AI Engineering Assistant")
query = st.text_input("Ask a question:", placeholder="e.g., 'How does duty cycle affect average voltage?'")
if query:
    with st.spinner("Analyzing engineering database..."):
        response = get_chat_response(query)
        st.markdown(f'<div style="background-color:#1E1E2E; padding:20px; border-radius:10px; border-left: 5px solid #667eea; margin-top:10px;">{response}</div>', unsafe_allow_html=True)
