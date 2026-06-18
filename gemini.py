# =============================================================================
# PWM SIGNAL SIMULATOR DASHBOARD
# =============================================================================

# ==============================
# Imports
# ==============================

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import base64
from io import StringIO
import time

# ==============================
# OPTIONAL AI IMPORTS
# ==============================

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None
    cosine_similarity = None
    _HAS_SENTENCE_TRANSFORMERS = False


# =============================================================================
# FALLBACK CONFIG
# =============================================================================

VMAX = 5.0
DEFAULT_FREQUENCY = 1000
DEFAULT_DUTY_CYCLE = 50
DEFAULT_TIME_WINDOW = 0.05


# =============================================================================
# APP CONFIG
# =============================================================================

st.set_page_config(
    page_title="PWM Simulator",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ PWM Signal Simulator")
st.markdown("Real-time PWM simulation with realistic device modeling.")


# =============================================================================
# PWM GENERATION
# =============================================================================

def generate_pwm_signal(duty_cycle, frequency, time_duration_s):
    frequency = max(1, float(frequency))
    time_duration_s = max(0.001, float(time_duration_s))

    duty = np.clip(duty_cycle / 100.0, 0.0, 1.0)
    period = 1.0 / frequency

    samples_per_cycle = 100
    total_samples = int(frequency * samples_per_cycle * time_duration_s)
    total_samples = max(200, min(total_samples, 50000))

    t = np.linspace(0, time_duration_s, total_samples)
    phase = np.mod(t, period)

    pwm = np.where(phase < duty * period, VMAX, 0.0)
    dt = t[1] - t[0] if len(t) > 1 else time_duration_s

    return t, pwm, dt


# =============================================================================
# DEVICE MODELS (REAL WORLD PHYSICS)
# =============================================================================

def first_order_filter(vin, dt, tau):
    y = np.zeros_like(vin, dtype=float)
    alpha = 1.0 - np.exp(-dt / tau) if tau > 0 else 1.0
    for i in range(1, len(vin)):
        y[i] = y[i - 1] + alpha * (vin[i] - y[i - 1])
    return y


def simulate_rc(vin, dt, R=1000, C=10e-6):
    tau = R * C
    vout = np.zeros_like(vin, dtype=float)
    alpha = 1.0 - np.exp(-dt / tau)
    for i in range(1, len(vin)):
        vout[i] = vout[i - 1] + alpha * (vin[i] - vout[i - 1])
    return vout


def simulate_rl(vin, dt, R=3.0, L=10e-3):
    current = np.zeros_like(vin, dtype=float)
    for i in range(1, len(vin)):
        di_dt = (vin[i-1] - R * current[i - 1]) / L
        current[i] = current[i - 1] + di_dt * dt
    current_scaled = current * R
    return np.clip(current_scaled, 0, VMAX)


def simulate_led(vin, Vf=2.0):
    R_series = 220.0
    current = np.where(vin > Vf, (vin - Vf) / R_series, 0.0)
    brightness = current * 100  
    return np.clip(brightness, 0, VMAX)


def simulate_diode(vin, dt, Vf=0.7):
    vout = np.where(vin > Vf, vin - Vf, 0.0)
    return vout


def simulate_zener(vin, dt, Vz=3.3):
    vout = np.zeros_like(vin)
    for i in range(len(vin)):
        if vin[i] < 0.7:
            vout[i] = 0.0
        elif vin[i] < Vz:
            vout[i] = vin[i] - 0.7
        else:
            vout[i] = Vz
    return first_order_filter(vout, dt, tau=0.0002)


def simulate_transistor(vin, dt, Vth=1.2):
    target = np.where(vin > Vth, VMAX, 0.0)
    return first_order_filter(target, dt, tau=0.00008)


def simulate_motor(vin, dt):
    R = 2.0; L = 0.001
    Ke = 0.01; Kt = 0.01; J = 0.0001; B = 0.00001
    
    current = np.zeros_like(vin)
    speed = np.zeros_like(vin)

    for i in range(1, len(vin)):
        di = (vin[i-1] - current[i-1]*R - Ke*speed[i-1]) / L * dt
        current[i] = max(current[i-1] + di, 0.0)
        
        domega = (Kt*current[i] - B*speed[i-1]) / J * dt
        speed[i] = max(speed[i-1] + domega, 0.0)

    max_theoretical_speed = VMAX / Ke
    speed_scaled = (speed / max_theoretical_speed) * VMAX
    return np.clip(speed_scaled, 0, VMAX)


def simulate_heater(vin, dt):
    ambient_temp = 25.0
    R_heater = 12.0
    R_th = 5.0 
    tau_th = 10.0 

    temp = np.full_like(vin, ambient_temp)
    alpha = 1.0 - np.exp(-dt / tau_th)

    for i in range(1, len(vin)):
        power = (vin[i] ** 2) / R_heater
        t_ss = ambient_temp + (power * R_th)
        temp[i] = temp[i - 1] + alpha * (t_ss - temp[i - 1])

    return temp


def simulate_buzzer(vin, dt, threshold=2.5):
    tone = np.where(vin > threshold, 1.0, 0.0)
    output = np.zeros_like(vin)
    tau_s = 0.05
    alpha = 1.0 - np.exp(-dt / tau_s)
    
    for i in range(1, len(vin)):
        output[i] = output[i - 1] + alpha * (tone[i] - output[i - 1])
    return output * VMAX

# =============================================================================
# DEVICE RESPONSE ROUTER
# =============================================================================

def get_device_response(device, vin, dt):
    if device == "capacitor":
        return simulate_rc(vin, dt)
    elif device == "inductor":
        return simulate_rl(vin, dt)
    elif device == "led":
        return simulate_led(vin)
    elif device == "diode":
        return simulate_diode(vin, dt)
    elif device == "zener":
        return simulate_zener(vin, dt)
    elif device == "transistor":
        return simulate_transistor(vin, dt)
    elif device == "motor":
        return simulate_motor(vin, dt)
    elif device == "heater":
        return simulate_heater(vin, dt)
    elif device == "buzzer":
        return simulate_buzzer(vin, dt)
    else:
        raise ValueError("Unknown device")


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(signal):
    signal = np.array(signal, dtype=float)
    return {
        "mean": float(np.mean(signal)),
        "rms": float(np.sqrt(np.mean(signal ** 2))),
        "min": float(np.min(signal)),
        "max": float(np.max(signal))
    }


# =============================================================================
# EXPORT CSV
# =============================================================================

def export_csv(t, y, filename="pwm_output.csv"):
    buffer = StringIO()
    buffer.write("time,signal\n")

    for ti, yi in zip(t, y):
        buffer.write(f"{ti},{yi}\n")

    b64 = base64.b64encode(buffer.getvalue().encode()).decode()
    return f'''
    <a href="data:file/csv;base64,{b64}"
       download="{filename}">
       Download CSV
    </a>
    '''


# =============================================================================
# ARDUINO CODE GENERATOR
# =============================================================================

def generate_arduino_code(duty_cycle, pin):
    pwm_value = int(np.clip(duty_cycle, 0, 100) / 100 * 255)

    return f"""
int pwmPin = {pin};

void setup()
{{
    pinMode(pwmPin, OUTPUT);
}}

void loop()
{{
    analogWrite(pwmPin, {pwm_value});
}}
"""


# =============================================================================
# PLOT
# =============================================================================

def plot_waveforms(t, pwm, output, mode, device):

    if mode == "Both":
        fig, ax = plt.subplots(figsize=(13, 5))

        ax.plot(
            t, pwm, linestyle="--", linewidth=1.5, alpha=0.7, label="PWM Input"
        )
        
        if device in ["heater", "motor", "led"]:
            ax_twin = ax.twinx()
            ax_twin.plot(t, output, linewidth=2.5, color="orange", label=f"{device.capitalize()} Output")
            ax_twin.set_ylabel(f"Response ({'°C' if device == 'heater' else 'Scaled'})")
            ax_twin.legend(loc="upper right")
        else:
            ax.plot(t, output, linewidth=2.5, color="orange", label=f"{device.capitalize()} Output")
            ax.set_ylabel("Amplitude")
            ax.legend(loc="upper right")

        ax.set_title(f"{device.capitalize()} vs PWM")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("PWM Voltage (V)")
        ax.grid(True, alpha=0.3)
        return fig

    elif mode == "PWM Only":
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(t, pwm, linestyle="--", linewidth=2, color="blue")
        ax.set_title("PWM Input Signal")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage")
        ax.grid(True, alpha=0.3)
        return fig

    elif mode == "Device Only":
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(t, output, linewidth=2.5, color="orange")
        ax.set_title(f"{device.capitalize()} Output")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Response")
        ax.grid(True, alpha=0.3)
        return fig

    elif mode == "Separate Subplots":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        ax1.plot(t, pwm, linestyle="--", linewidth=1.5)
        ax1.set_title("PWM Input")
        ax1.set_ylabel("Voltage")
        ax1.grid(True, alpha=0.3)

        ax2.plot(t, output, linewidth=2.5, color="orange")
        ax2.set_title(f"{device.capitalize()} Response")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Output")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

# =============================================================================
# SMART INSIGHTS
# =============================================================================

def generate_insights(device, frequency, duty_cycle, metrics):
    insights = []
    recommendations = []

    if duty_cycle < 30:
        duty_level = "🟢 LOW"
    elif duty_cycle < 70:
        duty_level = "🟡 MEDIUM"
    else:
        duty_level = "🔴 HIGH"

    insights.append(f"Duty Cycle Level: {duty_level}")
    insights.append(f"Operating Frequency: {frequency} Hz")
    insights.append(f"Mean Output: {metrics['mean']:.2f}")
    insights.append(f"RMS Output: {metrics['rms']:.2f}")

    if device == "led":
        insights.append(f"LED Brightness ≈ {duty_cycle:.0f}%")
        if duty_cycle < 20: recommendations.append("🟢 Dim LED operation")
        elif duty_cycle < 80: recommendations.append("🟡 Normal LED brightness")
        else: recommendations.append("🔴 Very high brightness → heating possible")

        if frequency < 100: recommendations.append("🔴 Visible LED flicker likely")
        else: recommendations.append("🟢 Smooth LED brightness")

    elif device == "motor":
        insights.append(f"Estimated Motor Speed ≈ {duty_cycle:.0f}%")
        if duty_cycle < 25: recommendations.append("🟢 Low speed operation")
        elif duty_cycle < 75: recommendations.append("🟡 Moderate motor speed")
        else: recommendations.append("🔴 High speed → increased current draw")

        if frequency < 50: recommendations.append("🔴 Motor may jerk or vibrate")
        else: recommendations.append("🟢 Smooth motor rotation expected")

    elif device == "heater":
        insights.append(f"Estimated Heating Power ≈ {duty_cycle:.0f}%")
        if duty_cycle < 30: recommendations.append("🟢 Low heating")
        elif duty_cycle < 70: recommendations.append("🟡 Moderate heating")
        else: recommendations.append("🔴 High temperature operation")

        if frequency < 50: recommendations.append("🟡 Heater response is slow but visible")
        else: recommendations.append("🟢 Thermal averaging is strong")

    elif device == "capacitor":
        insights.append("Capacitor smooths PWM into analog-like voltage")
        if frequency < 100: recommendations.append("🔴 Ripple voltage may be high")
        elif frequency < 1000: recommendations.append("🟡 Moderate filtering")
        else: recommendations.append("🟢 Strong smoothing effect")

    elif device == "inductor":
        insights.append("Inductor resists sudden current changes")
        if frequency < 100: recommendations.append("🔴 Current ripple may be large")
        elif frequency < 1000: recommendations.append("🟡 Moderate ripple current")
        else: recommendations.append("🟢 Smooth inductor current")

    elif device == "diode":
        insights.append("Diode allows one-direction current flow")
        if duty_cycle < 20: recommendations.append("🟢 Low conduction interval")
        elif duty_cycle < 80: recommendations.append("🟡 Normal rectification")
        else: recommendations.append("🔴 High average diode current")

    elif device == "zener":
        insights.append("Zener regulates voltage near breakdown level")
        if duty_cycle < 30: recommendations.append("🟢 Light regulation load")
        elif duty_cycle < 70: recommendations.append("🟡 Stable regulation")
        else: recommendations.append("🔴 High zener power dissipation")

    elif device == "transistor":
        insights.append("Transistor operates as PWM electronic switch")
        if duty_cycle < 20: recommendations.append("🟢 Low switching activity")
        elif duty_cycle < 80: recommendations.append("🟡 Efficient switching region")
        else: recommendations.append("🔴 High conduction time → heating possible")
        if frequency > 10000: recommendations.append("🟡 Switching losses may increase")
        else: recommendations.append("🟢 Switching stress remains moderate")

    elif device == "buzzer":
        insights.append("Buzzer converts PWM into audible sound")
        if frequency < 100: recommendations.append("🔴 Clicking sound likely")
        elif frequency < 5000: recommendations.append("🟢 Audible tone region")
        else: recommendations.append("🟡 Frequency may exceed hearing range")

        if duty_cycle < 20: recommendations.append("🟢 Low sound intensity")
        elif duty_cycle < 80: recommendations.append("🟡 Moderate sound level")
        else: recommendations.append("🔴 Very loud buzzer operation")

    final_output = []
    final_output.extend(insights)
    final_output.extend(recommendations)
    return final_output


# =============================================================================
# KNOWLEDGE BASE
# =============================================================================

KNOWLEDGE_BASE = [
    {"topic": "pwm", "text": "PWM controls average power using ON/OFF switching."},
    {"topic": "motor", "text": "Motor speed depends on average voltage and inertia."},
    {"topic": "capacitor", "text": "Capacitors smooth PWM into DC-like voltage."},
    {"topic": "inductor", "text": "Inductors resist sudden current changes."},
    {"topic": "heater", "text": "Heaters respond slowly due to thermal mass."},
    {"topic": "led", "text": "LED brightness changes with duty cycle and average current."},
    {"topic": "diode", "text": "Diodes conduct only when forward biased above threshold."},
    {"topic": "zener", "text": "Zener diodes clamp voltage at a breakdown level."},
    {"topic": "transistor", "text": "Transistors can act like PWM-controlled switches."},
    {"topic": "buzzer", "text": "Buzzers respond with sound depending on PWM switching."}
]


# =============================================================================
# AI CHATBOT
# =============================================================================

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
        return ("Semantic AI features are unavailable because sentence_transformers is not installed.")
    q_emb = _model.encode([query], normalize_embeddings=True)
    scores = cosine_similarity(q_emb, _kb_embeddings)[0]
    idx = int(np.argmax(scores))
    return KNOWLEDGE_BASE[idx]["text"]


# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================

st.sidebar.header("PWM Controls")

frequency = st.sidebar.slider("Frequency (Hz)", 1, 20000, DEFAULT_FREQUENCY)
duty_cycle = st.sidebar.slider("Duty Cycle (%)", 0, 100, DEFAULT_DUTY_CYCLE)

device = st.sidebar.selectbox(
    "Device",
    ["capacitor", "inductor", "led", "diode", "zener", "transistor", "motor", "heater", "buzzer"]
)
graph_mode = st.sidebar.selectbox(
    "Advanced Graph View",
    ["Both", "PWM Only", "Device Only", "Separate Subplots"]
)

time_window = st.sidebar.slider(
    "Time Window (s)",
    0.01, 30.0, 0.10, step=0.01
)
pin = st.sidebar.selectbox("PWM Pin", [3, 5, 6, 9, 10, 11])

if device == "heater" and time_window < 5.0:
    st.sidebar.warning("Heater needs a larger time window (10.0s+) for visible thermal response.")

if device == "motor" and time_window < 0.5:
    st.sidebar.warning("Motor inertia is easier to see with a larger time window (1.0s+).")


# =============================================================================
# SIMULATION
# =============================================================================

t, pwm, dt = generate_pwm_signal(duty_cycle, frequency, time_window)
output = get_device_response(device, pwm, dt)
metrics = compute_metrics(output)


# =============================================================================
# OUTPUT DISPLAY
# =============================================================================

st.subheader("📈 Waveform Output")
st.pyplot(plot_waveforms(t, pwm, output, graph_mode, device))
st.markdown(export_csv(t, output), unsafe_allow_html=True)


# =============================================================================
# METRICS DISPLAY
# =============================================================================

st.subheader("📊 Metrics")
col1, col2, col3, col4 = st.columns(4)

if device == "heater":
    col1.metric("Mean Temp", f"{metrics['mean']:.1f} °C")
    col4.metric("Max Temp", f"{metrics['max']:.1f} °C")
else:
    col1.metric("Mean", f"{metrics['mean']:.2f}")
    col2.metric("RMS", f"{metrics['rms']:.2f}")
    col3.metric("Min", f"{metrics['min']:.2f}")
    col4.metric("Max", f"{metrics['max']:.2f}")


# =============================================================================
# INSIGHTS
# =============================================================================

st.subheader("🧠 Smart Insights")
for insight in generate_insights(device, frequency, duty_cycle, metrics):
    st.write(insight)


# =============================================================================
# ARDUINO CODE
# =============================================================================

st.subheader("🔌 Arduino PWM Code")
st.code(generate_arduino_code(duty_cycle, pin), language="cpp")


# =============================================================================
# CHATBOT UI
# =============================================================================

st.subheader("🤖 PWM AI Assistant")
query = st.text_input("Ask about PWM/devices:")
if query:
    response = get_chat_response(query)
    st.success(response)


# =============================================================================
# LIVE DEVICE DYNAMICS (UPGRADED ANIMATION BLOCK)
# =============================================================================

st.subheader("🎞 Live Device Dynamics")

# Helper to normalize device output for animations
norm = np.clip((output - np.min(output)) / (np.max(output) - np.min(output) + 1e-9), 0, 1)

if st.button("▶ Start/Refresh Animation"):
    anim_placeholder = st.empty()
    
    # Calculate a safe step size to run at roughly 30-60 FPS based on array size
    step_size = max(1, len(norm) // 60)
    
    for v in norm[::step_size]:
        if device == "led":
            html = f"""
            <div style="text-align:center; padding:20px; border-radius:15px; background:#1a1a1a;">
                <div style="font-size:80px; filter: drop-shadow(0 0 {20*v}px gold);">💡</div>
                <div style="font-weight:bold; color:gold; margin-top:10px;">INTENSITY: {int(v*100)}%</div>
            </div>"""
        elif device == "motor":
            html = f"""
            <div style="text-align:center; padding:20px; border-radius:15px; background:#1a1a1a;">
                <div style="font-size:80px; display:inline-block; transform: rotate({v*360}deg);">⚙️</div>
                <div style="font-weight:bold; color:#4facfe; margin-top:10px;">RPM: {int(v*100)}%</div>
            </div>"""
        elif device == "heater":
            red = int(100 + v * 155)
            html = f"""
            <div style="text-align:center; padding:20px; border-radius:15px; background:#1a1a1a;">
                <div style="font-size:80px;">🔥</div>
                <div style="color:rgb({red}, 50, 0); font-weight:bold; font-size:20px; margin-top:10px;">{int(25 + v*100)}°C</div>
            </div>"""
        elif device == "buzzer":
            state = "🔊" if v > 0.5 else "🔈"
            html = f"""
            <div style="text-align:center; padding:20px; border-radius:15px; background:#1a1a1a;">
                <div style="font-size:{60 + int(v * 40)}px;">{state}</div>
                <div style="font-weight:bold; color:#00f2fe; margin-top:10px;">VOLUME: {int(v*100)}%</div>
            </div>"""
        elif device == "capacitor":
            fill = int(v * 100)
            html = f"""
            <div style="text-align:center; padding:20px; border-radius:15px; background:#1a1a1a;">
                <div style="font-size:60px;">🔋</div>
                <div style="width:200px; height:20px; margin:15px auto; border:2px solid white; border-radius:5px; overflow:hidden;">
                    <div style="width:{fill}%; height:100%; background:#00ff87;"></div>
                </div>
                <div style="font-weight:bold; color:#00ff87;">CHARGE: {fill}%</div>
            </div>"""
        elif device == "inductor":
            waves = int(v * 6) + 1
            html = f"""
            <div style="text-align:center; padding:20px; border-radius:15px; background:#1a1a1a;">
                <div style="font-size:60px;">🌀</div>
                <h3 style="color:#b224ef; letter-spacing: 5px;">{'➰' * waves}</h3>
                <div style="font-weight:bold; color:#b224ef;">CURRENT FLUX: {int(v*100)}%</div>
            </div>"""
        elif device == "diode":
            state = "CONDUCTING ✅" if v > 0.2 else "BLOCKING ❌"
            color = "#00ff87" if v > 0.2 else "#ff003c"
            html = f"""
            <div style="text-align:center; padding:20px; border-radius:15px; background:#1a1a1a;">
                <div style="font-size:80px;">➡️</div>
                <div style="font-weight:bold; color:{color}; margin-top:10px;">{state}</div>
            </div>"""
        elif device == "zener":
            state = "CLAMPED ⚡" if v > 0.7 else "NORMAL 🟢"
            color = "#f6d365" if v > 0.7 else "#00ff87"
            html = f"""
            <div style="text-align:center; padding:20px; border-radius:15px; background:#1a1a1a;">
                <div style="font-size:80px; filter: drop-shadow(0 0 {10*v}px {color});">⚡</div>
                <div style="font-weight:bold; color:{color}; margin-top:10px;">{state}</div>
            </div>"""
        elif device == "transistor":
            state = "ON 🟢" if v > 0.5 else "OFF 🔴"
            color = "#00ff87" if v > 0.5 else "#ff003c"
            html = f"""
            <div style="text-align:center; padding:20px; border-radius:15px; background:#1a1a1a;">
                <div style="font-size:80px;">🔀</div>
                <div style="font-weight:bold; color:{color}; margin-top:10px;">STATE: {state}</div>
            </div>"""
            
        anim_placeholder.markdown(html, unsafe_allow_html=True)
        time.sleep(0.03)


# =============================================================================
# MODULAR UI BLOCK: COLLAPSIBLE ADVANCED FEATURES
# =============================================================================

st.markdown("---")
with st.expander("🚀 Advanced Engineering Features (Optional)", expanded=False):
    st.info("Toggle this block to hide/show advanced diagnostic tools. Selecting 'None' clears the space.")
    
    advanced_feature = st.selectbox(
        "Select Advanced Feature",
        [
            "None",
            "FFT Analyzer",
            "Device Theory Panel",
            "Oscilloscope Theme",
            "Comparison Mode",
            "Efficiency Calculator",
            "Circuit Diagram Panel",
            "Real Component Sliders"
        ]
    )

    # -------------------------------------------------------------------------
    # FFT ANALYZER
    # -------------------------------------------------------------------------
    if advanced_feature == "FFT Analyzer":
        st.subheader("🎵 FFT Frequency Spectrum")
        fft = np.fft.fft(output)
        freqs = np.fft.fftfreq(len(output), d=dt)
        positive = freqs > 0
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(freqs[positive], np.abs(fft[positive]), linewidth=1.5)
        ax.set_xlim(0, frequency * 10)
        ax.set_title(f"{device.capitalize()} Frequency Spectrum")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # -------------------------------------------------------------------------
    # DEVICE THEORY PANEL
    # -------------------------------------------------------------------------
    elif advanced_feature == "Device Theory Panel":
        st.subheader("📘 Device Theory")
        theory = {
            "capacitor": "Capacitors oppose sudden voltage change.\n\nPWM Result:\n- Smooth charging/discharging curve\n- Converts PWM into analog-like voltage\n- Higher frequency = smoother output",
            "inductor": "Inductors oppose sudden current change.\n\nPWM Result:\n- Current ramps gradually\n- Triangular ripple waveform possible\n- Used in buck converters and motors",
            "led": "LEDs respond to average current.\n\nPWM Result:\n- Brightness controlled by duty cycle\n- High PWM frequency removes flicker",
            "diode": "Diodes conduct only in forward bias.\n\nPWM Result:\n- Acts like rectifier\n- Blocks reverse conduction",
            "zener": "Zener diode regulates voltage.\n\nPWM Result:\n- Output clamps near breakdown voltage\n- Used for voltage protection",
            "transistor": "Transistor acts as electronic switch.\n\nPWM Result:\n- ON/OFF switching waveform\n- Used for motor and LED control",
            "motor": "Motors have inertia.\n\nPWM Result:\n- Speed changes gradually\n- Mechanical lag smooths PWM",
            "heater": "Heaters have thermal inertia.\n\nPWM Result:\n- Temperature changes slowly\n- PWM controls average heating power",
            "buzzer": "Buzzers convert PWM into sound.\n\nPWM Result:\n- Frequency controls tone\n- Duty cycle affects loudness"
        }
        st.info(theory.get(device, "No theory available."))

    # -------------------------------------------------------------------------
    # OSCILLOSCOPE THEME
    # -------------------------------------------------------------------------
    elif advanced_feature == "Oscilloscope Theme":
        st.subheader("🖥 Oscilloscope Display")
        fig, ax = plt.subplots(figsize=(12, 5), facecolor="black")
        ax.set_facecolor("black")
        ax.plot(t, output, linewidth=2, color="lime")
        ax.plot(t, pwm, linestyle="--", alpha=0.4, color="cyan")
        ax.set_title("Oscilloscope View", color="white")
        ax.set_xlabel("Time (s)", color="white")
        ax.set_ylabel("Amplitude", color="white")
        ax.tick_params(colors="white")
        ax.grid(True, color="green", alpha=0.2)
        st.pyplot(fig)

    # -------------------------------------------------------------------------
    # COMPARISON MODE
    # -------------------------------------------------------------------------
    elif advanced_feature == "Comparison Mode":
        st.subheader("🔬 Duty Cycle Comparison")
        compare_values = [20, 50, 80]
        fig, ax = plt.subplots(figsize=(12, 5))
        for d in compare_values:
            t_cmp, pwm_cmp, dt_cmp = generate_pwm_signal(d, frequency, time_window)
            out_cmp = get_device_response(device, pwm_cmp, dt_cmp)
            ax.plot(t_cmp, out_cmp, linewidth=2, label=f"{d}% Duty")
        ax.set_title(f"{device.capitalize()} Comparison")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Output")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    # -------------------------------------------------------------------------
    # EFFICIENCY CALCULATOR
    # -------------------------------------------------------------------------
    elif advanced_feature == "Efficiency Calculator":
        st.subheader("⚡ Efficiency Calculator")
        avg_voltage = np.mean(output)
        
        if device == "heater":
            efficiency = (duty_cycle / 100) * 100 
        else:
            efficiency = (avg_voltage / VMAX) * 100
            
        loss = 100 - efficiency
        col1, col2 = st.columns(2)
        col1.metric("Estimated Efficiency", f"{efficiency:.1f}%")
        col2.metric("Estimated Loss", f"{loss:.1f}%")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(["Useful Output", "Loss"], [efficiency, loss])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percent")
        ax.set_title("PWM Efficiency")
        st.pyplot(fig)

    # -------------------------------------------------------------------------
    # CIRCUIT DIAGRAM PANEL
    # -------------------------------------------------------------------------
    elif advanced_feature == "Circuit Diagram Panel":
        st.subheader("🔌 Basic Circuit Diagram")
        diagrams = {
            "led": "PWM Pin ---- Resistor ---- LED ---- GND",
            "motor": "PWM Pin ---- MOSFET ---- Motor ---- Supply\n                     |\n                    Diode",
            "capacitor": "PWM ---- RC Filter ---- Output\n           |\n       Capacitor\n           |\n          GND",
            "inductor": "PWM ---- Inductor ---- Load",
            "heater": "PWM ---- MOSFET ---- Heater",
            "transistor": "PWM ---- Base Resistor ---- Transistor"
        }
        st.code(diagrams.get(device, "Circuit not available"))

    # -------------------------------------------------------------------------
    # REAL COMPONENT SLIDERS
    # -------------------------------------------------------------------------
    elif advanced_feature == "Real Component Sliders":
        st.subheader("🎛 Real Component Controls")
        if device == "capacitor":
            R = st.slider("Resistance (Ohm)", 100, 10000, 1000)
            C = st.slider("Capacitance (uF)", 1, 1000, 100) * 1e-6
            modified_output = simulate_rc(pwm, dt, R=R, C=C)
        elif device == "inductor":
            R = st.slider("Resistance (Ohm)", 1, 20, 2)
            L = st.slider("Inductance (mH)", 1, 100, 5) * 1e-3
            modified_output = simulate_rl(pwm, dt, R=R, L=L)
        else:
            st.info("Real component sliders available mainly for RC/RL devices.")
            modified_output = output

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(t, pwm, linestyle="--", alpha=0.5, label="PWM")
        ax.plot(t, modified_output, linewidth=2.5, label="Modified Output")
        ax.set_title("Real Component Simulation")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

# =============================================================================
# END
# =============================================================================
