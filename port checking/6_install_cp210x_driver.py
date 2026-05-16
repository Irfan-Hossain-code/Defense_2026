"""
Step 6: Install the CP210x (Silicon Labs) driver for the ESP32's CP2102N chip.
Tries Windows Update first, then falls back to downloading the official installer.
"""
import subprocess
import sys
import os
import urllib.request
import tempfile

DRIVER_URL = "https://www.silabs.com/documents/public/software/CP210x_Windows_Drivers.zip"

def try_windows_update():
    print("Trying Windows Update to find CP210x driver ...")
    result = subprocess.run(
        ["pnputil", "/scan-devices"],
        capture_output=True, text=True, timeout=60
    )
    print(result.stdout)
    return result.returncode == 0

def check_if_fixed():
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if p.vid == 0x10C4 and p.pid == 0xEA60:
            return p.device
    return None

print("=" * 60)
print("CP210x DRIVER INSTALLATION")
print("=" * 60)

print("\nStep 1: Triggering Windows Update driver scan ...")
try_windows_update()

import time
print("Waiting 5 seconds for driver to apply ...")
time.sleep(5)

port = check_if_fixed()
if port:
    print(f"\nSUCCESS: CP210x driver installed. ESP32 is now on {port}")
    sys.exit(0)

print("\nWindows Update did not install the driver automatically.")
print("\nStep 2: Attempting PowerShell driver update ...")

ps_cmd = (
    'Get-PnpDevice | Where-Object { $_.InstanceId -like "*VID_10C4*" } | '
    'ForEach-Object { Update-PnpDevice -InstanceId $_.InstanceId -Confirm:$false }'
)
result = subprocess.run(
    ["powershell", "-Command", ps_cmd],
    capture_output=True, text=True, timeout=60
)
print(result.stdout or "(no output)")
if result.stderr:
    print("STDERR:", result.stderr[:500])

time.sleep(5)
port = check_if_fixed()
if port:
    print(f"\nSUCCESS: Driver installed. ESP32 is now on {port}")
    sys.exit(0)

print("\nAutomatic install did not work.")
print("\n" + "=" * 60)
print("MANUAL INSTALL REQUIRED")
print("=" * 60)
print("""
Your ESP32 uses a Silicon Labs CP2102N chip.
The driver is NOT included in Windows by default.

Download and install from Silicon Labs:
  1. Go to: www.silabs.com
  2. Search: "CP210x VCP drivers"
  3. Download "CP210x Windows Drivers" (ZIP)
  4. Extract and run: CP210xVCPInstaller_x64.exe
  5. Restart if prompted, then replug the ESP32

After installing, run this again or run 2_find_esp32.py to confirm.
""")
