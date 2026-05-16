# ESP32 CSI Board Verification Checklist

Work through this list **one board at a time**.  Each section is a gate:
do not advance until the current section passes.

---

## What is CSI and why does it matter?

**Channel State Information** is a snapshot of how the Wi-Fi radio channel
looks right now.  The ESP32's Wi-Fi hardware uses OFDM — it splits the 20 MHz
band into **52 parallel sub-carriers** (narrow frequency slots).  For every
Wi-Fi packet it receives it measures how each sub-carrier was attenuated and
phase-shifted by the air, walls, furniture, and anything moving.

When a person walks between a transmitter (your router) and the receiver
(ESP32), the reflections they cause change which sub-carriers are boosted or
attenuated.  That pattern change is detectable without a camera.

**Practical ESP32 CSI limits you must know:**

| Limit | Detail |
|---|---|
| Receive-only | CSI is measured on RECEIVED packets. The ESP32 cannot measure its own transmissions. You always need a Wi-Fi transmitter nearby. |
| Passive rate | Without active traffic the only source is router beacon frames (~10/sec). That is enough for slow motion detection. |
| Active rate | With UDP/ping traffic between two ESP32s you can push 50-100 CSI samples/sec. |
| Sub-carriers | 52 (LLTF) to 56 (HT-LTF) usable sub-carriers per 20 MHz channel. |
| Phase drift | Raw phase values are uncalibrated and drift with temperature. Use **amplitude** (sqrt(imag²+real²)) as the stable signal. |
| Range | Reliable motion detection ≈ 5-10 m through 1-2 walls at 2.4 GHz. |
| Antenna | ESP32-WROOM has a PCB antenna — works fine; directional patch antennas would improve range. |

---

## Prerequisites (do once, not per board)

- [ ] PlatformIO extension installed in VS Code
  - Open VS Code → Extensions → search "PlatformIO IDE" → Install
- [ ] Python virtualenv active with pyserial
  ```powershell
  # From the project root (defence_2026)
  .venv\Scripts\Activate.ps1
  pip install pyserial
  ```
- [ ] Edit `firmware/src/main.cpp` — set `WIFI_SSID` and `WIFI_PASS`
  - Use a 2.4 GHz network (ESP32 does not support 5 GHz)
  - If you have no router handy, leave them blank and use the fallback AP mode
    (you will need a second device connected to `ESP32_CSI_TEST` to get CSI)

---

## Per-Board Checklist

### Board ID: ________  COM Port: ________  Date: ________

---

### Stage 1 — Physical connection

- [ ] **USB cable plugged in** — use a data cable, not a charge-only cable
  - Charge-only cables have no D+/D− wires; the device will not enumerate
- [ ] **Device Manager shows a new COM port** (or run the detect script)

  ```powershell
  # From the project root
  .venv\Scripts\Activate.ps1
  python port_test.py
  ```

  Expected output example:
  ```
  COM4: Silicon Labs CP210x USB to UART Bridge (COM4) | hwid=USB VID:PID=10C4:EA60
  ```

  Common chip names: `CP210x`, `CH340`, `FTDI FT232`.
  If nothing appears → try a different USB cable → try a different USB port.

- [ ] Note the COM port: **COM____**

---

### Stage 2 — Build and flash firmware

1. Open the firmware folder in VS Code:
   ```powershell
   code csi_test\firmware
   ```
   VS Code should detect `platformio.ini` and activate PlatformIO.

2. Tell PlatformIO which port to use — open `platformio.ini` and uncomment:
   ```ini
   upload_port  = COM4   ; ← your port
   monitor_port = COM4
   ```

3. **Build the firmware** — press the tick (✓) icon in the PlatformIO toolbar
   (bottom-left of VS Code) or open the PlatformIO panel → Build.
   - First build downloads the ESP32 toolchain (~300 MB). Be patient.
   - Expected: `SUCCESS` with a RAM/Flash usage summary.

- [ ] Firmware compiles without errors

4. **Flash the firmware** — press the right-arrow (→) icon or PlatformIO → Upload.
   - If the upload hangs at "Connecting..." hold the BOOT button on the ESP32
     while the dots appear, then release.  Some boards need this.
   - Expected: `Hard resetting via RTS pin... done`

- [ ] Firmware flashed successfully

---

### Stage 3 — Serial communication

Open the serial monitor in one of these ways:
- PlatformIO toolbar → plug icon (Serial Monitor)
- VS Code command palette: `PlatformIO: Serial Monitor`
- Or from the terminal:
  ```powershell
  # This avoids needing the PlatformIO monitor
  .venv\Scripts\Activate.ps1
  python -m serial.tools.miniterm COM4 115200
  ```

Within 5 seconds of the monitor opening you should see:
```
INFO: === ESP32 CSI Tester ===
INFO: Connecting to YOUR_WIFI_NAME
```

- [ ] Serial output appears (board is alive and printing)

---

### Stage 4 — Wi-Fi connection

After the connection attempt (up to 15 seconds of dots) you should see either:

**Success:**
```
INFO: Connected! IP=192.168.1.42 GW=192.168.1.1 CH=6 RSSI=-52dBm
INFO: CSI enabled. Waiting for Wi-Fi packets...
```

**Fallback AP mode (no router available):**
```
INFO: Could not connect to YOUR_WIFI_NAME
INFO: Starting fallback AP: ESP32_CSI_TEST
INFO: AP IP: 192.168.4.1
```

- [ ] Wi-Fi connected (STA mode) **OR** AP mode started (fallback)

If you see repeated `......` without connecting:
  - Double-check `WIFI_SSID` and `WIFI_PASS` spelling (case-sensitive)
  - Confirm the network is 2.4 GHz — ESP32 cannot see 5 GHz networks
  - Move the board closer to the router

---

### Stage 5 — CSI data stream

Within a few seconds of connecting you should see lines like:
```
CSI:-52,-95,6,128,3,-1,2,0,-4,2,5,-3,...
CSI:-53,-95,6,128,1,0,3,-1,-5,3,4,-2,...
STATUS: uptime=5s csi_packets=48 rssi=-52dBm
```

**What the numbers mean:**
```
CSI: <rssi>, <noise>, <channel>, <len>, <bytes...>
       -52     -95       6        128    raw sub-carrier data
```

- `rssi` — signal strength from the AP (dBm, typically −40 to −80)
- `noise` — ambient noise floor (dBm, typically −90 to −100)
- `channel` — Wi-Fi channel (6, 11, etc.)
- `len` — number of raw bytes (128 = 64 sub-carrier pairs for LLTF+HTLTF)
- bytes — alternating int8_t: [imag0, real0, imag1, real1, ...]

**Typical rates:**
- Connected to router, no extra traffic: ~10 lines/sec (beacon frames)
- With active pinging between two ESP32s: 50–100 lines/sec

- [ ] CSI lines appear in serial monitor
- [ ] `STATUS` heartbeat appears every 5 seconds
- [ ] Packet count increases (csi_packets > 0)

If no CSI lines appear but Wi-Fi connected:
  - The CSI API may have returned an error — look for `ERROR:` lines
  - Some cheap clone boards have a different Wi-Fi chip; check with the
    seller that the board uses the standard Espressif chip

---

### Stage 6 — Python logger

Close the serial monitor (only one program can own the COM port at a time), then:

```powershell
# From the project root, with venv active
.venv\Scripts\Activate.ps1
python csi_test\python\csi_logger.py --port COM4
```

Expected output:
```
Saving CSI data to: csi_test\data\csi_20260516_143022.csv
Opening serial port COM4 at 115200 baud...
Press Ctrl+C to stop.

[ESP32] INFO: CSI enabled. Waiting for Wi-Fi packets...
  pkts=    11  rate=  9.8/s  rssi= -52dBm  ch= 6  subcarriers=64  mean_amp= 18.43  snr=43.0dB
  pkts=    21  rate= 10.1/s  rssi= -52dBm  ch= 6  subcarriers=64  mean_amp= 18.51  snr=43.0dB
```

Press **Ctrl+C** to stop.

- [ ] CSV file created in `csi_test/data/`
- [ ] CSV has rows with timestamp, rssi, amplitudes per sub-carrier
- [ ] Packet rate is non-zero and consistent

Verify the CSV is readable:
```powershell
# Print first 3 rows
python -c "import csv; rows=list(csv.reader(open('csi_test/data/csi_YOURTIMESTAMP.csv'))); [print(r[:10]) for r in rows[:3]]"
```

---

### Stage 7 — Motion sensitivity spot check (optional but recommended)

With the logger running and CSI flowing:

1. Stand still for 10 seconds — watch `mean_amplitude` values in the terminal
2. Walk slowly between the ESP32 and the router — the `mean_amplitude` should
   fluctuate noticeably (even ±2–5 units indicates good sensitivity)
3. Wave your hand between the devices — you should see faster fluctuations

- [ ] `mean_amplitude` is stable when still
- [ ] `mean_amplitude` fluctuates when moving

If amplitude barely changes during motion: the person is not in the propagation
path between router and ESP32.  Reposition so the board is line-of-sight or
near-line-of-sight to the router, with you walking between them.

---

### Board result

| Check | Result |
|---|---|
| Detected on COM port | Pass / Fail |
| Firmware compiled & flashed | Pass / Fail |
| Serial output OK | Pass / Fail |
| Wi-Fi connected | Pass / Fail (STA / AP fallback) |
| CSI lines visible | Pass / Fail |
| Python logger saves CSV | Pass / Fail |
| Motion changes amplitude | Pass / Fail |

Notes: _______________________________________________________________

---

## Next steps after all boards pass

1. Set up one ESP32 as a **dedicated transmitter** (connect to the AP and send
   UDP packets at 50 Hz) and a second as the **CSI receiver**.  More predictable
   traffic = cleaner CSI signal.
2. Wire the mean-amplitude stream into `PersonTracker.update_rf_estimate()` in
   `main.py` — when amplitude variance exceeds a threshold, nudge the ghost
   figure in the direction of motion.
3. Try 5 GHz if you get an ESP32-S2 or ESP32-C6 (they support 5 GHz) for
   higher sub-carrier count and better multipath resolution.
