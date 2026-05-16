"""
Step 2: Scan all ports for known ESP32 USB-Serial chip VID/PID combos.
ESP32 boards ship with one of three bridge chips — this script identifies which.
"""
import serial.tools.list_ports

# (VID, PID) -> chip name
ESP32_CHIPS = {
    (0x10C4, 0xEA60): "Silicon Labs CP2102/CP2104 (very common on ESP32 DevKit)",
    (0x10C4, 0xEA70): "Silicon Labs CP2105",
    (0x1A86, 0x7523): "CH340  (common on cheap ESP32 clones)",
    (0x1A86, 0x7522): "CH340B",
    (0x1A86, 0x55D4): "CH343P",
    (0x0403, 0x6001): "FTDI FT232R",
    (0x0403, 0x6010): "FTDI FT2232",
    (0x0403, 0x6014): "FTDI FT232H",
    (0x303A, 0x1001): "Espressif USB-JTAG/Serial (ESP32-S3 / ESP32-C3 native USB)",
    (0x303A, 0x0002): "Espressif ESP32-S2 native USB",
}

ports = list(serial.tools.list_ports.comports())

print("=" * 60)
print("ESP32 PORT SCAN")
print("=" * 60)

found = []
for p in ports:
    if p.vid is None:
        continue
    key = (p.vid, p.pid)
    if key in ESP32_CHIPS:
        found.append((p, ESP32_CHIPS[key]))

if found:
    print(f"\nFound {len(found)} ESP32-compatible port(s):\n")
    for p, chip in found:
        print(f"  PORT : {p.device}")
        print(f"  CHIP : {chip}")
        print(f"  HWID : {p.hwid}")
        print()
    print("Use the port name above in your code, e.g.:")
    print(f'  serial.Serial("{found[0][0].device}", 115200)')
else:
    print("\nNo ESP32 found on any port.")
    print("\nAll ports seen (for reference):")
    if ports:
        for p in ports:
            vid = hex(p.vid) if p.vid else "no VID"
            pid = hex(p.pid) if p.pid else "no PID"
            print(f"  {p.device}: {p.description} | VID={vid} PID={pid}")
    else:
        print("  (none)")

    print("\nIf the ESP32 is plugged in but not listed, check:")
    print("  1. Cable — must be a data cable, not charge-only")
    print("  2. Driver — see 3_check_drivers.py")
    print("  3. Device Manager — look for yellow ! under 'Other devices'")
