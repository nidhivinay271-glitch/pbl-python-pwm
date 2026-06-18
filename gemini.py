import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Global Config ---
VMAX = 5.0
st.set_page_config(page_title="PWM Diagnostic Workbench", layout="wide")
st.title("⚡ PWM Core Physics Diagnostic")

# =============================================================================
# PHYSICS ENGINE (ISOLATED)
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

def get_device_response(device, vin, dt):
    dt_float = float(dt)
    
    if device == "capacitor":
        tau = 1000 * 10e-6 # R * C
        y = np.zeros_like(vin, dtype=float)
        alpha = 1.0 - np.exp(-dt_float / tau)
        for i in range(1, len(vin)):
            y[i] = y[i - 1] + alpha * (vin[i] - y[i - 1])
        return y
        
    elif device == "motor":
        Ke, Kt, B, J = 0.01, 0.01, 0.00001, 0.0001
        current, speed = np.zeros_like(vin), np.zeros_like(vin)
        for i in range(1, len(vin)):
            current[i] = max(current[i-1] + ((vin[i-1] - 2.0 * current[i-1] - Ke*speed[i-1]) / 0.001 * dt_float), 0.0)
            speed[i] = max(speed[i-1] + ((Kt*current[i] - B*speed[i-1]) / J * dt_float), 0.0)
        return np.clip((speed / (VMAX / Ke)) * VMAX, 0, VMAX)
        
    elif device == "heater":
        temp = np.full_like(vin, 25.0)
        # We use a short thermal tau (2.0s) so it fits on a standard graph window
        alpha = 1.0 - np.exp(-dt_float / 2.0) 
        for i in range(1, len(vin)):
            temp[i] = temp[i-1] + alpha * ((25.0 + ((vin[i]**2) / 12.0) * 5.0) - temp[i-1])
        return temp

# =============================================================================
# UI & CONTROLS
# =============================================================================

st.sidebar.header("Test Parameters")
device = st.sidebar.selectbox("Test Device", ["heater", "motor", "capacitor"])
duty_cycle = st.sidebar.slider("Duty Cycle (%)", 0, 100, 50)
frequency = st.sidebar.slider("Frequency (Hz)", 1, 2000, 100)

# ⚠️ CRITICAL: The Heater requires a long time window to show a curve!
default_time = 5.0 if device == "heater" else 0.05
time_window = st.sidebar.slider("Time Window (s)", 0.005, 10.0, default_time, step=0.01)

# Run Simulation
t, pwm, dt = generate_pwm_signal(duty_cycle, frequency, time_window)
output = get_device_response(device, pwm, dt)

# =============================================================================
# GRAPH OUTPUT
# =============================================================================

st.subheader(f"Diagnostic Graph: {device.capitalize()}")

fig, ax = plt.subplots(
