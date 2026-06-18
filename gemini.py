# =============================================================================
# PWM SIGNAL SIMULATOR DASHBOARD - ULTIMATE PHYSICS EDITION
# =============================================================================

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
import pandas as pd
from io import StringIO
import time

# --- Global Config ---
VMAX = 5.0
st.set_page_config(page_title="⚡ Real-Time PWM WorkBench", layout="wide")

# =============================================================================
# PHYSICS ENGINE
# =============================================================================

@st.cache_data
def generate_pwm_signal(duty_cycle, frequency, time_duration_s):
    frequency = max(1.0, float(frequency))
    time_duration_s = max(0.001, float(time_duration_s))
    duty = np.clip(duty_cycle / 100.0, 0.0, 1.0)
    period = 1.0 / frequency
    total_samples = np.clip(int(frequency * 100 * time_duration_s), 200, 50000)
    
    t = np.linspace(0, time_duration_s, total_samples, endpoint=False)
    phase = np.mod(t, period)
    pwm = np.where(phase < duty * period, VMAX, 0.0)
    dt = t[1] - t[0] if len(t) > 1 else time_duration_s
    return t, pwm, dt

def first_order_filter(vin, dt, tau):
    y = np.zeros_like(vin, dtype=float)
    alpha = 1.0 - np.exp(-dt / tau) if tau > 0 else 1.0
    for i in range(1, len(vin)):
        y[i] = y[i - 1] + alpha * (vin[i] - y[i - 1])
    return y

def get_device_response(device, vin, dt, time_duration_s, R_cap=1000, C_cap=10e-6, R_ind=3.0, L_ind=10e-3):
    dt_float = float(dt)
    
    if device == "capacitor":
        return first_order_filter(vin, dt_float, R_cap * C_cap)
        
    elif device == "inductor":
        current = np.zeros_like(vin)
        for i in range(1, len(vin)):
            current[i] = current[i-1] + ((vin[i-1] - R_ind * current[i-1]) / L_ind) * dt_float
        return np.clip(current * R_ind, 0, VMAX)
        
    elif device == "led":
        return np.clip(np.where(vin > 2.0, ((vin - 2.0) / (VMAX - 2.0)) * VMAX, 0.0), 0, VMAX)
        
    elif device == "diode":
        return np.where(vin > 0.7, vin - 0.7, 0.0)
        
    elif device == "zener":
        vout = np.where(vin < 0.7, 0.0, np.where(vin < 3.3, vin - 0.7, 3.3))
        return first_order_filter(vout, dt_float, tau=0.0002)
        
    elif device == "transistor":
        return first_order_filter(np.where(vin > 1.2, VMAX, 0.0), dt_float, tau=0.00008)
        
    elif device == "motor":
        Ke, Kt, B, J = 0.01, 0.01, 0.00001, 0.0001
        current, speed = np.zeros_like(vin), np.zeros_like(vin)
        for i in range(1, len(vin)):
            current[i] = max(current[i-1] + ((vin[i-1] - 2.0 * current[i-1] - Ke*speed[i-1]) / 0.001 * dt_float), 0.0)
            speed[i] = max(speed[i-1] + ((Kt*current[i] - B*speed[i-1]) / J * dt_float), 0.0)
        return np.clip((speed / (VMAX / Ke)) * VMAX, 0, VMAX)
        
    elif device == "heater":
        temp = np.full_like(vin, 25.0)
        tau_th = max(0.01, time_duration_s / 4.0) 
        alpha = 1.0 - np.exp(-dt_float / tau_th) 
        for i in range(1, len(vin)):
            power_factor = vin[i] / VMAX
            target_temp = 25.0 + (power_factor * 200.0)
            temp[i] = temp[i-1] + alpha * (target_temp - temp[i-1])
        return temp
        
    elif device == "buzzer":
        return first_order_filter(np.where(vin > 2.5, 1.0, 0.0), dt_float, 0.05) * VMAX

# =============================================================================
# SMART INSIGHTS GENERATOR (RED/YELLOW/GREEN)
# =============================================================================
def get_smart_insights(device, dc, freq):
    insights = []
    if dc <= 30: insights.append("🌱 **Mode: ECO** - Low power consumption, minimal heat generation.")
    elif dc <= 70: insights.append("⚖️ **Mode: NEUTRAL** - Balanced performance and energy use.")
    else: insights.append("🚀 **Mode: PERFORMANCE** - Maximum output, highest current draw.")

    if device == "motor":
        if freq < 100: insights.append("🔴 **Poor:** Frequency too low. Motor will physically vibrate and jerk.")
        elif freq <= 500: insights.append("🟡 **Acceptable:** Motor turns, but may emit a loud, audible coil whine.")
        else: insights.append("🟢 **Optimal:** High frequency provides smooth, silent rotation.")
        if dc < 15: insights.append("🔴 **Warning:** Duty cycle may be too low to overcome mechanical stiction.")
        elif dc > 85: insights.append("🟡 **Caution:** Approaching max continuous current; monitor thermals.")
        else: insights.append("🟢 **Good:** Operating in the highly linear speed-control region.")
    elif device == "led":
        if freq < 60: insights.append("🔴 **Poor:** Frequency is below human persistence of vision. LED will visibly flicker.")
        else: insights.append("🟢 **Optimal:** Frequency is high enough for smooth, flicker-free dimming.")
    elif device == "heater":
        if freq > 100: insights.append("🟡 **Overkill:** High frequencies waste switching energy. Heaters have massive thermal inertia.")
        else: insights.append("🟢 **Optimal:** Low frequency switching is perfect for high-current heating elements.")
    elif device == "capacitor":
        if freq < 200: insights.append("🔴 **Poor:** Frequency is too slow for the RC time constant. Massive voltage ripple.")
        elif freq < 1000: insights.append("🟡 **Acceptable:** Moderate ripple; good enough for basic analog signals.")
        else: insights.append("🟢 **Optimal:** High frequency completely smoothed into a flat DC voltage.")
    elif device == "buzzer":
        if freq < 100: insights.append("🔴 **Poor:** Will sound like an annoying clicking noise rather than a tone.")
        elif freq > 10000: insights.append("🟡 **Caution:** Approaching the limits of typical piezo acoustic response.")
        else: insights.append("🟢 **Optimal:** Operating in the primary audible pitch range.")
    else:
        insights.append("🟢 **System Stable:** General solid-state switching parameters look acceptable.")
        
    return insights

# =============================================================================
# MAIN APP & SIDEBAR
# =============================================================================
st.title("⚡ PWM Signal Simulator")

st.sidebar.header("PWM Controls")
preset_mode = st.sidebar.radio("Operating Mode Preset", ["Manual", "🌱 Eco (25%)", "⚖️ Neutral (50%)", "🚀 Performance (85%)"])
if preset_mode == "🌱 Eco (25%)": st.session_state.dc = 25
elif preset_mode == "⚖️ Neutral (50%)": st.session_state.dc = 50
elif preset_mode == "🚀 Performance (85%)": st.session_state.dc = 85
elif "dc" not in st.session_state: st.session_state.dc = 50

duty_cycle = st.sidebar.slider("Duty Cycle (%)", 0, 100, st.session_state.dc)
frequency = st.sidebar.slider("Frequency (Hz)", 1, 20000, 1000)

device = st.sidebar.selectbox("Device", ["capacitor", "inductor", "led", "diode", "zener", "transistor", "motor", "heater", "buzzer"])
time_window = st.sidebar.slider("Time Window (s)", 0.005, 10.0, 0.05, step=0.005)
graph_mode = st.sidebar.selectbox("Advanced Graph View", ["Both", "PWM Only", "Device Only", "Separate Subplots"])
pin = st.sidebar.selectbox("PWM Pin", [3, 5, 6, 9, 10, 11])

t, pwm, dt = generate_pwm_signal(duty_cycle, frequency, time_window)
output = get_device_response(device, pwm, dt, time_window)
metrics = {"mean": float(np.mean(output)), "rms": float(np.sqrt(np.mean(output ** 2))), "min": float(np.min(output)), "max": float(np.max(output))}

# =============================================================================
# OUTPUT & GRAPHS (FIXED SCALING DUAL Y-AXIS)
# =============================================================================
st.subheader("📈 Waveform Output")

if graph_mode == "Separate Subplots":
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax1.plot(t, pwm, linestyle="--", linewidth=1.5, color="#1f77b4", label="PWM Input (V)")
    ax1.set_title("PWM Input Signal")
    ax1.set_ylabel("Voltage (V)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    
    ax2.plot(t, output, linewidth=2.5, color="#ff7f0e", label=f"{device.capitalize()} Output")
    ax2.set_title(f"{device.capitalize()} Response")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    plt.tight_layout()
    st.pyplot(fig)

else:
    fig, ax = plt.subplots(figsize=(13, 5))
    if graph_mode == "Both":
        ax.plot(t, pwm, linestyle="--", linewidth=1.5, alpha=0.7, color="#1f77b4", label="PWM Input (V)")
        ax.set_ylabel("PWM Logic Level (V)", color="#1f77b4")
        ax.tick_params(axis='y', labelcolor="#1f77b4")
        
        # Dual Y-Axis for devices with completely different scales
        if device in ["heater", "motor"]:
            ax2 = ax.twinx()
            ax2.plot(t, output, linewidth=2.5, color="#ff7f0e", label=f"{device.capitalize()} Output")
            ax2.set_ylabel("Temperature (°C)" if device == "heater" else "Speed (RPM)", color="#ff7f0e")
            ax2.tick_params(axis='y', labelcolor="#ff7f0e")
            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")
        else:
            ax.plot(t, output, linewidth=2.5, color="#ff7f0e", label=f"{device.capitalize()} Output")
            ax.legend(loc="upper right")

    elif graph_mode == "PWM Only":
        ax.plot(t, pwm, linestyle="--", linewidth=2, color="#1f77b4", label="PWM Input")
        ax.set_ylabel("Voltage (V)")
        ax.legend(loc="upper right")

    elif graph_mode == "Device Only":
        ax.plot(t, output, linewidth=2.5, color="#ff7f0e", label=f"{device.capitalize()} Output")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")

    ax.set_title(f"{device.capitalize()} Response" if graph_mode != "PWM Only" else "PWM Input Signal")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

buffer = StringIO(); buffer.write("time,signal\n")
for ti, yi in zip(t, output): buffer.write(f"{ti},{yi}\n")
b64 = base64.b64encode(buffer.getvalue().encode()).decode()
st.markdown(f'<a href="data:file/csv;base64,{b64}" download="pwm_output.csv">Download CSV Data</a>', unsafe_allow_html=True)

# =============================================================================
# METRICS & SMART INSIGHTS
# =============================================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Metrics")
    c1, c2 = st.columns(2)
    c1.metric("Mean", f"{metrics['mean']:.2f}")
    c2.metric("RMS", f"{metrics['rms']:.2f}")
    c3, c4 = st.columns(2)
    c3.metric("Min", f"{metrics['min']:.2f}")
    c4.metric("Max", f"{metrics['max']:.2f}")

with col2:
    st.subheader("🧠 Smart Insights")
    insights = get_smart_insights(device, duty_cycle, frequency)
    for insight in insights:
        st.markdown(insight)

# =============================================================================
# DYNAMIC ARDUINO CODE BLOCK
# =============================================================================
st.markdown("---")
st.subheader("🔌 Dynamic Arduino Code")
st.markdown(f"This code automatically updates based on your selected **PWM Pin ({pin})** and **Duty Cycle ({duty_cycle}%)**.")

# Map 0-100% duty cycle to Arduino's 8-bit analogWrite scale (0-255)
pwm_8bit = int((duty_cycle / 100.0) * 255)

arduino_code = f"""int pwmPin = {pin};

void setup() {{
    // Set the selected pin as an output
    pinMode(pwmPin, OUTPUT);
}}

void loop() {{
    // {duty_cycle}% Duty Cycle equates to {pwm_8bit} on an 8-bit scale (0-255)
    analogWrite(pwmPin, {pwm_8bit});
}}
"""
st.code(arduino_code, language="cpp")

# =============================================================================
# LIVE DEVICE DYNAMICS (PERFECT SYNC)
# =============================================================================
st.markdown("---")
st.subheader("🎞 Device Animation")

if st.button("▶ Run Physics Animation", key="run_anim"):
    placeholder = st.empty()
    
    # Absolute physical scaling matching the graph Y-axis
    if device == "heater":
        v_anim = np.clip((output - 25.0) / (250.0 - 25.0), 0.0, 1.0)
    else:
        v_anim = np.clip(output / VMAX, 0.0, 1.0)

    step = max(1, len(v_anim) // 120)

    for i in range(0, len(v_anim), step):
        v = v_anim[i]
        current_time = t[i]

        if device == "led":
            glow, size = int(50 + v * 205), 80 + int(v * 40)
            html = f'<div style="text-align:center; font-size:{size}px; filter: drop-shadow(0 0 {20*v}px rgb(255,255,0));">💡<br><h3 style="color:rgb({glow},{glow},0);">Brightness: {int(v * 100)}%</h3><p>Time: {current_time:.3f} s</p></div>'
        elif device == "motor":
            html = f'<div style="text-align:center;"><div style="font-size:{80 + int(v*40)}px; transform: rotate({v*360*5}deg);">⚙️</div><br><h3>Speed: {int(v * 100)}%</h3><p>Time: {current_time:.3f} s</p></div>'
        elif device == "heater":
            html = f'<div style="text-align:center;"><div style="font-size:{80 + int(v*20)}px;">🔥</div><br><h3 style="color:rgb({int(100+v*155)},50,0);">Temp: {int(25 + v * 225)} °C</h3><p>Time: {current_time:.3f} s</p></div>'
        elif device == "capacitor":
            fill = int(v * 100)
            html = f'<div style="text-align:center; font-size:{70 + int(v*20)}px;">🔋<br><div style="width:300px; height:30px; margin:auto; border:2px solid white; border-radius:10px;"><div style="width:{fill}%; height:100%; background:lime; border-radius:8px;"></div></div><h3>Charge: {fill}%</h3><p>Time: {current_time:.3f} s</p></div>'
        elif device == "inductor":
            html = f'<div style="text-align:center; font-size:{80 + int(v*20)}px;">🌀<br><h3>{"➰" * int(v*8)}</h3><p>Flux: {int(v * 100)}% | Time: {current_time:.3f} s</p></div>'
        elif device == "buzzer":
            state = "🔊" if v > 0.5 else "🔈"
            html = f'<div style="text-align:center;"><div style="font-size:{80 + int(v*30)}px;">{state}</div><h2>Sound Level</h2><h3>{int(v * 100)}%</h3><p>Time: {current_time:.3f} s</p></div>'
        elif device == "diode":
            state = "Conducting ✅" if v > 0.2 else "Blocking ❌"
            html = f'<div style="text-align:center;"><div style="font-size:90px;">➡️</div><h2>{state}</h2><p>Time: {current_time:.3f} s</p></div>'
        elif device == "zener":
            state = "Voltage Clamped ⚡" if v > 0.7 else "Normal"
            html = f'<div style="text-align:center;"><div style="font-size:{80 + int(v*20)}px;">⚡</div><h2>{state}</h2><p>Regulation: {int(v * 100)}%</p><p>Time: {current_time:.3f} s</p></div>'
        elif device == "transistor":
            state = "ON 🟢" if v > 0.5 else "OFF 🔴"
            html = f'<div style="text-align:center;"><div style="font-size:{80 + int(v*15)}px;">🔀</div><h2>{state}</h2><p>Switching Level: {int(v * 100)}%</p><p>Time: {current_time:.3f} s</p></div>'
        else:
            html = f'<div style="text-align:center; font-size:{80 + int(v*20)}px;">⚡<br><h3>State: {int(v * 100)}%</h3><p>Time: {current_time:.3f} s</p></div>'
            
        placeholder.markdown(html, unsafe_allow_html=True)
        time.sleep(0.02)

# =============================================================================
# ADVANCED FEATURES 
# =============================================================================
st.markdown("---")
st.header("🚀 Advanced Engineering Features")

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
        "Real Component Sliders",
        "Data Table (Real-Time)",
        "Safety Limits & Stress Analysis"
    ]
)

if advanced_feature == "FFT Analyzer":
    st.subheader("🎵 FFT Frequency Spectrum")
    fft = np.fft.fft(output - np.mean(output))
    freqs = np.fft.fftfreq(len(output), d=dt)
    positive = freqs > 0
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs[positive], np.abs(fft[positive]), linewidth=1.5)
    ax.set_xlim(0, frequency * 10)
    ax.set_title("Frequency Spectrum")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

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

elif advanced_feature == "Comparison Mode":
    st.subheader("🔬 Duty Cycle Comparison")
    compare_values = [20, 50, 80]
    fig, ax = plt.subplots(figsize=(12, 5))
    for d in compare_values:
        t_cmp, pwm_cmp, dt_cmp = generate_pwm_signal(d, frequency, time_window)
        out_cmp = get_device_response(device, pwm_cmp, dt_cmp, time_window)
        ax.plot(t_cmp, out_cmp, linewidth=2, label=f"{d}% Duty")
    ax.set_title(f"{device.capitalize()} Comparison")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Output")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

elif advanced_feature == "Efficiency Calculator":
    st.subheader("⚡ Power & Efficiency Calculator")
    eff = (duty_cycle / 100.0) * 100 if device == "heater" else (np.mean(output) / VMAX) * 100
    c1, c2 = st.columns(2)
    c1.metric("Logic Conversion Efficiency", f"{eff:.1f}%")
    c2.metric("Heat/Switching Loss", f"{100-eff:.1f}%")

elif advanced_feature == "Circuit Diagram Panel":
    st.subheader("🔌 Basic Circuit Diagram")
    diagrams = {
        "led": "PWM Pin ---- Resistor ---- LED ---- GND", 
        "motor": "PWM Pin ---- MOSFET ---- Motor ---- Supply\n                     |\n                    Diode", 
        "capacitor": "PWM ---- RC Filter ---- GND",
        "inductor": "PWM ---- Inductor ---- Load",
        "heater": "PWM ---- MOSFET ---- Heater",
        "transistor": "PWM ---- Base Resistor ---- Transistor"
    }
    st.code(diagrams.get(device, "Circuit not available"))

elif advanced_feature == "Real Component Sliders":
    st.subheader("🎛 Real Component Controls")
    if device in ["capacitor", "inductor"]:
        R = st.slider("Resistance (Ohm)", 1, 1000, 100)
        mod_out = get_device_response(device, pwm, dt, time_window, R_cap=R, R_ind=R)
        fig, ax = plt.subplots()
        ax.plot(t, mod_out, label="Modified Output")
        st.pyplot(fig)
    else: st.info("Sliders apply to RC/RL components.")

elif advanced_feature == "Data Table (Real-Time)":
    st.subheader("📋 Raw Simulation Data")
    df = pd.DataFrame({"Time (s)": t, "PWM Input (V)": pwm, "Device Output": output})
    st.dataframe(df.head(100), use_container_width=True) 

elif advanced_feature == "Safety Limits & Stress Analysis":
    st.subheader("⚠️ Safety & Thermal Analysis")
    if duty_cycle > 90 and device in ["motor", "heater", "transistor", "led"]:
        st.error(f"DANGER: {duty_cycle}% duty cycle places extreme thermal stress on the {device}.")
    elif frequency > 15000 and device in ["transistor", "motor"]:
        st.warning(f"CAUTION: {frequency}Hz switching frequency induces high dynamic switching losses.")
    else:
        st.success(f"System operating within safe thermal and switching limits for a generic {device}.")

# =============================================================================
# PURE PYTHON NLP AI CHATBOT (NO DEPENDENCIES REQUIRED - WILL NEVER CRASH)
# =============================================================================
st.markdown("---")
st.subheader("🤖 PWM AI Engineering Assistant")
st.markdown("Ask the AI about the physics, math, or how any device reacts to PWM in the real world.")

def get_smart_ai_response(user_query):
    q = user_query.lower()
    
    # EXACT intent matching to differentiate "what is" vs "why use"
    is_what = any(k in q for k in ["what", "define", "meaning", "how does it work", "explain pwm"])
    is_why = any(k in q for k in ["why", "advantage", "benefit", "reason", "purpose"])
    
    if is_what and "pwm" in q:
        return "### ⚡ What is PWM?\n**Pulse Width Modulation (PWM)** is a digital technique used to mimic analog results. Because microcontrollers (like Arduino) cannot output varying voltages natively (they only output 0V or 5V), they rapidly switch the pin ON and OFF to create a specific 'average' voltage.\n\n* **Formula:** $V_{avg} = Duty\\_Cycle \\times V_{max}$"
        
    if is_why and "pwm" in q:
        return "### 🤔 Why use PWM?\nEfficiency! If you use a resistor to drop 5V down to 2.5V, the resistor burns the extra energy as heat. With PWM, a transistor is either fully ON (near zero resistance) or fully OFF (zero current). This means very little power is wasted, making it ideal for high-current loads like motors and heaters."
        
    # Device physics matching
    if "motor" in q: return "### ⚙️ DC Motor Physics\nA motor converts electrical PWM energy into mechanical rotation. It has **electrical inertia** (Inductance) which smooths current spikes, and **mechanical inertia** (Rotor Mass) which smooths physical speed. PWM provides full 5V 'kicks' of torque, preventing stalls at low speeds."
    if "capacitor" in q or "rc" in q: return "### 🔋 Capacitor & RC Filters\nA Capacitor stores electrical energy. Paired with a Resistor, it forms an **RC Low-Pass Filter**. It opposes sudden changes in voltage. If the PWM frequency is fast enough, the capacitor averages the PWM into a clean, flat DC voltage."
    if "inductor" in q or "rl" in q: return "### 🌀 Inductor & RL Circuits\nAn Inductor stores energy in a magnetic field and opposes sudden changes in **current**. When PWM turns ON, it forces current to ramp up slowly. This is the foundational principle of Buck Converters."
    if "heater" in q or "thermal" in q: return "### 🔥 Resistive Heaters\nDue to massive **thermal inertia**, heaters react very slowly. Therefore, PWM frequencies for heaters can be extremely low (e.g., 1 Hz). The ambient environment acts as a natural low-pass filter."
    if "led" in q or "light" in q: return "### 💡 Light Emitting Diode (LED)\nLEDs react to electricity instantly. If you apply a 50% PWM, the LED flashes ON and OFF at full brightness. Frequencies above ~100 Hz blend together due to human **Persistence of Vision**, making it appear 50% bright."
    if "diode" in q: return "### ➡️ Standard Diode\nA Diode is a one-way check valve for current. Passing through the P-N junction results in a continuous voltage drop of about 0.7V."
    if "zener" in q: return "### ⚡ Zener Diode\nZener diodes safely break down and conduct backwards when a specific **Zener Voltage (Vz)** is reached. They clamp and protect sensitive microcontrollers from high logic signals."
    if "transistor" in q or "mosfet" in q: return "### 🔀 Transistor\nTransistors act as massive electronic switches. By feeding a microcontroller's tiny PWM signal into the Gate of a MOSFET, it mirrors the PWM signal but allows massive currents to flow to heavy loads."
    if "buzzer" in q or "sound" in q: return "### 🔊 Piezo Buzzer\nA buzzer converts PWM into acoustic waves. **Frequency** controls the pitch (note), and **Duty Cycle** controls the volume (50% is the loudest because it allows maximum crystal flex)."
    if "duty" in q or "cycle" in q: return "### ⏱️ Duty Cycle Explained\nThe Duty Cycle is the percentage of one period in which a signal is active (HIGH).\n\n* **0%:** Completely flat (0V). Device is OFF.\n* **50%:** HIGH half the time, LOW half the time. Average voltage is half of max.\n* **100%:** Constant DC voltage. Device is fully ON."
    if "freq" in q or "time" in q: return "### 🌊 Frequency & Simulation Time\n**Frequency (Hz)** dictates how many times per second the PWM cycle repeats. Electrical devices need high frequencies for smooth outputs. Mechanical/Thermal devices are slow and naturally smooth out even low frequencies."

    return "🤔 I didn't catch a specific concept. Try asking 'What is PWM?', 'Why use PWM?', or ask about a device like 'How do motors react to PWM?'"

query = st.text_input("Ask a question:", placeholder="e.g., 'What is PWM?', 'Why use PWM?', or 'Explain motor physics'")

if query:
    with st.spinner("Analyzing Engineering Database..."):
        time.sleep(0.4) # UI loading feel
        response = get_smart_ai_response(query)
        st.markdown(f'<div style="background-color:#1E1E2E; padding:20px; border-radius:10px; border-left: 5px solid #00ff87;">{response}</div>', unsafe_allow_html=True)
