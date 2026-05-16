"""
Step 5: Dump all USB devices Windows sees, regardless of driver status.
This catches the ESP32 even when it has no driver and shows up as 'Unknown device'.
"""
import subprocess

print("=" * 60)
print("ALL USB DEVICES (via pnputil)")
print("=" * 60)

try:
    result = subprocess.run(
        ["pnputil", "/enum-devices", "/class", "USB", "/connected"],
        capture_output=True, text=True, timeout=15
    )
    print(result.stdout or "(no output)")
except FileNotFoundError:
    print("pnputil not found — falling back to WMI query")
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-WmiObject Win32_USBControllerDevice | "
             "ForEach-Object { [wmi]($_.Dependent) } | "
             "Select-Object Name, DeviceID, Status | Format-Table -AutoSize"],
            capture_output=True, text=True, timeout=20
        )
        print(result.stdout or "(no output)")
    except Exception as e:
        print(f"WMI query also failed: {e}")

print()
print("=" * 60)
print("PORTS CLASS DEVICES (includes COM ports with no driver)")
print("=" * 60)

try:
    result = subprocess.run(
        ["pnputil", "/enum-devices", "/class", "Ports", "/connected"],
        capture_output=True, text=True, timeout=15
    )
    print(result.stdout or "(no Ports-class devices found)")
except Exception as e:
    print(f"Error: {e}")

print()
print("=" * 60)
print("UNKNOWN / UNRECOGNISED DEVICES")
print("=" * 60)

try:
    result = subprocess.run(
        ["powershell", "-Command",
         "Get-WmiObject Win32_PnPEntity | "
         "Where-Object { $_.ConfigManagerErrorCode -ne 0 } | "
         "Select-Object Name, DeviceID, ConfigManagerErrorCode | "
         "Format-Table -AutoSize"],
        capture_output=True, text=True, timeout=20
    )
    output = result.stdout.strip()
    print(output if output else "No problem devices found.")
except Exception as e:
    print(f"Error: {e}")
