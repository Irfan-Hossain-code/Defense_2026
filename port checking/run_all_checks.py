"""
Run all ESP32 port diagnostics in sequence and produce a summary report.
"""
import subprocess
import sys
import os

SCRIPTS = [
    ("1_list_all_ports.py",  "List all COM ports"),
    ("2_find_esp32.py",      "Find ESP32 by VID/PID"),
    ("3_check_drivers.py",   "Check USB-Serial drivers"),
    ("5_usb_devices.py",     "Enumerate all USB devices"),
]

here = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("ESP32 FULL DIAGNOSTIC REPORT")
print("=" * 60)

for script, label in SCRIPTS:
    print(f"\n>>> {label} ({script})")
    print("-" * 60)
    result = subprocess.run(
        [sys.executable, os.path.join(here, script)],
        capture_output=False
    )

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
print("""
Quick fix checklist:
  [ ] Is the USB cable a data cable? (try a different cable)
  [ ] Is the ESP32 light on AND does the PC make a sound when plugged in?
  [ ] Does Device Manager show a yellow ! for any device?
  [ ] Is the correct driver installed? (check step 3 output above)
  [ ] Is Arduino IDE / another serial monitor holding the port open?
  [ ] Try a different physical USB port on your PC
  [ ] On some ESP32-S3/C3 boards, hold BOOT and press EN while plugging in
""")
