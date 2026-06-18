# =============================================================================
# DEVICE ANIMATION (FIXED GRAPH SYNC)
# =============================================================================

st.subheader("🎞 Device Animation")

if st.button("Run Animation", key="run_animation_button"):

    placeholder = st.empty()

    # FIX: Calculate animation absolute percentage to perfectly sync with the graph!
    if device == "heater":
        # Heater scales from 25 ambient to 250 max
        v_anim = np.clip((output - 25.0) / (250.0 - 25.0), 0.0, 1.0)
    else:
        # All other devices scale 0 to VMAX (5V)
        v_anim = np.clip(output / VMAX, 0.0, 1.0)

    max_frames = 120
    step = max(1, len(v_anim) // max_frames)

    for i in range(0, len(v_anim), step):
        v = v_anim[i]
        current_time = t[i]

        if device == "led":
            glow = int(50 + v * 205)
            size = 80 + int(v * 40)
            html = f"""
            <div style="text-align:center; font-size:{size}px; filter: drop-shadow(0 0 {20*v}px rgb(255,255,0));">💡</div>
            <h3 style="text-align:center; color:rgb({glow},{glow},0);">Brightness: {int(v * 100)}%</h3>
            <p style="text-align:center;">Simulation Time: {current_time:.3f} s</p>
            """
            placeholder.markdown(html, unsafe_allow_html=True)

        elif device == "motor":
            html = f"""
            <div style="text-align:center;">
                <div style="font-size:{80 + int(v * 40)}px; transform: rotate({v * 360}deg); transition: 0.05s linear;">⚙️</div>
                <h3>Speed: {int(v * 100)}%</h3>
                <p>Simulation Time: {current_time:.3f} s</p>
            </div>
            """
            placeholder.markdown(html, unsafe_allow_html=True)

        elif device == "heater":
            red = int(100 + v * 155)
            html = f"""
            <div style="text-align:center;">
                <div style="font-size:{80 + int(v * 20)}px;">🔥</div>
                <h2 style="color:rgb({red},50,0);">Temperature</h2>
                <h3>{int(25 + v * 225)} °C</h3>
                <p>Simulation Time: {current_time:.3f} s</p>
            </div>
            """
            placeholder.markdown(html, unsafe_allow_html=True)

        elif device == "buzzer":
            state = "🔊" if v > 0.5 else "🔈"
            html = f"""
            <div style="text-align:center;">
                <div style="font-size:{80 + int(v * 30)}px;">{state}</div>
                <h2>Sound Level</h2>
                <h3>{int(v * 100)}%</h3>
                <p>Simulation Time: {current_time:.3f} s</p>
            </div>
            """
            placeholder.markdown(html, unsafe_allow_html=True)

        elif device == "capacitor":
            fill = int(v * 100)
            html = f"""
            <div style="text-align:center;">
                <div style="font-size:{70 + int(v * 20)}px;">🔋</div>
                <h2>Charge Level</h2>
                <div style="width:300px; height:30px; margin:auto; border:2px solid white; border-radius:10px; overflow:hidden;">
                    <div style="width:{fill}%; height:100%; background:lime; border-radius:8px;"></div>
                </div>
                <h3>{fill}%</h3>
                <p>Simulation Time: {current_time:.3f} s</p>
            </div>
            """
            placeholder.markdown(html, unsafe_allow_html=True)

        elif device == "inductor":
            waves = int(v * 8)
            html = f"""
            <div style="text-align:center;">
                <div style="font-size:{80 + int(v * 20)}px;">🌀</div>
                <h2>Magnetic Field</h2>
                <h3>{'➰' * waves}</h3>
                <p>Current: {int(v * 100)}%</p>
                <p>Simulation Time: {current_time:.3f} s</p>
            </div>
            """
            placeholder.markdown(html, unsafe_allow_html=True)

        elif device == "diode":
            state = "Conducting ✅" if v > 0.2 else "Blocking ❌"
            html = f"""
            <div style="text-align:center;">
                <div style="font-size:90px;">➡️</div>
                <h2>{state}</h2>
                <p>Simulation Time: {current_time:.3f} s</p>
            </div>
            """
            placeholder.markdown(html, unsafe_allow_html=True)

        elif device == "zener":
            state = "Voltage Clamped ⚡" if v > 0.7 else "Normal"
            html = f"""
            <div style="text-align:center;">
                <div style="font-size:{80 + int(v * 20)}px;">⚡</div>
                <h2>{state}</h2>
                <p>Regulation: {int(v * 100)}%</p>
                <p>Simulation Time: {current_time:.3f} s</p>
            </div>
            """
            placeholder.markdown(html, unsafe_allow_html=True)

        elif device == "transistor":
            state = "ON 🟢" if v > 0.5 else "OFF 🔴"
            html = f"""
            <div style="text-align:center;">
                <div style="font-size:{80 + int(v * 15)}px;">🔀</div>
                <h2>{state}</h2>
                <p>Switching Level: {int(v * 100)}%</p>
                <p>Simulation Time: {current_time:.3f} s</p>
            </div>
            """
            placeholder.markdown(html, unsafe_allow_html=True)

        time.sleep(0.02)
