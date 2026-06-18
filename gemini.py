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

# --- Optional AI Imports ---
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None
    cosine_similarity = None
    _HAS_SENTENCE_TRANSFORMERS = False

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
# OUTPUT & GRAPHS
# =============================================================================
st.subheader("📈
