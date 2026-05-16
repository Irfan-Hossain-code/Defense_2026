"""
Step 3: Check Windows registry for installed USB-Serial drivers.
Missing drivers are the #1 reason an ESP32 doesn't show up as a COM port.
"""
import winreg
import subprocess

DRIVER_KEYS = {
    "CP210x (Silicon Labs)": [
        r"SYSTEM\CurrentControlSet\Services\CP210x",
        r"SYSTEM\CurrentControlSet\Services\silabser",
    ],
    "CH340 / CH341": [
        r"SYSTEM\CurrentControlSet\Services\ch341ser",
        r"SYSTEM\CurrentControlSet\Services\CH341SER_A64",
    ],
    "FTDI": [
        r"SYSTEM\CurrentControlSet\Services\FTDIBUS",
        r"SYSTEM\CurrentControlSet\Services\ftdibus",
    ],
}

print("=" * 60)
print("DRIVER CHECK (Windows Registry)")
print("=" * 60)

any_found = False
for driver_name, keys in DRIVER_KEYS.items():
    found = False
    for key_path in keys:
        try:
            winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
            found = True
            break
        except FileNotFoundError:
            pass
    status = "INSTALLED" if found else "NOT FOUND"
    marker = "OK " if found else "!!!"
    print(f"  [{marker}] {driver_name}: {status}")
    if found:
        any_found = True

print()
if not any_found:
    print("No ESP32 USB-Serial drivers detected.")
    print("\nInstall the right driver for your board's bridge chip:")
    print("  CP2102 (most ESP32 DevKits):")
    print("    https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers")
    print("  CH340 (cheap clones):")
    print("    https://www.wch-ic.com/downloads/CH341SER_EXE.html")
    print("  FTDI:")
    print("    https://ftdichip.com/drivers/vcp-drivers/")
else:
    print("At least one driver is present.")
    print("If the ESP32 still isn't visible, try a different USB cable or port.")

print()
print("=" * 60)
print("DEVICE MANAGER: Unknown / Problem Devices")
print("=" * 60)

try:
    result = subprocess.run(
        ["pnputil", "/enum-devices", "/problem"],
        capture_output=True, text=True, timeout=15
    )
    output = result.stdout.strip()
    if output:
        print(output)
    else:
        print("No problem devices found — good sign.")
except Exception as e:
    print(f"Could not run pnputil: {e}")
    print("Open Device Manager manually and look for yellow ! icons.")
