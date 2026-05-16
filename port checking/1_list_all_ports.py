"""
Step 1: List every COM port Windows can see, with full details.
Run this first to get a baseline of what's visible.
"""
import serial.tools.list_ports

ports = list(serial.tools.list_ports.comports())

print("=" * 60)
print("ALL VISIBLE COM PORTS")
print("=" * 60)

if not ports:
    print("No COM ports found at all.")
    print("\nPossible causes:")
    print("  - USB cable is charge-only (no data lines)")
    print("  - Driver not installed for the USB-Serial chip")
    print("  - Device not recognised by Windows (check Device Manager)")
else:
    for p in ports:
        print(f"\nPort   : {p.device}")
        print(f"Name   : {p.name}")
        print(f"Desc   : {p.description}")
        print(f"HWID   : {p.hwid}")
        print(f"VID    : {hex(p.vid) if p.vid else 'N/A'}")
        print(f"PID    : {hex(p.pid) if p.pid else 'N/A'}")
        print(f"Mfr    : {p.manufacturer or 'N/A'}")
        print(f"Product: {p.product or 'N/A'}")
        print(f"Serial : {p.serial_number or 'N/A'}")
        print("-" * 40)

print(f"\nTotal ports found: {len(ports)}")
