"""
Step 4: Try to open a serial connection to an ESP32 port and read output.
Run this after steps 1-3 tell you which port to use.

Usage:
  python 4_connect_test.py           <- auto-detect first ESP32 port
  python 4_connect_test.py COM5      <- test a specific port
"""
import sys
import time
import serial
import serial.tools.list_ports

ESP32_VIDS = {0x10C4, 0x1A86, 0x0403, 0x303A}

def find_esp32_port():
    for p in serial.tools.list_ports.comports():
        if p.vid in ESP32_VIDS:
            return p.device
    return None

def test_port(port, baud=115200, duration=5):
    print(f"Attempting to open {port} at {baud} baud ...")
    try:
        with serial.Serial(port, baud, timeout=1) as ser:
            print(f"  Connection opened successfully.")
            print(f"  Reading for {duration} seconds (press Ctrl+C to stop early):\n")
            end_time = time.time() + duration
            lines_read = 0
            while time.time() < end_time:
                line = ser.readline()
                if line:
                    print("  >>", line.decode("utf-8", errors="replace").rstrip())
                    lines_read += 1
            if lines_read == 0:
                print("  (no data received — device may be idle or baud rate wrong)")
            print(f"\nConnection test complete. Lines received: {lines_read}")
    except serial.SerialException as e:
        print(f"  FAILED to open port: {e}")
        print("\nPossible causes:")
        print("  - Port is already open in another program (Arduino IDE, PlatformIO, etc.)")
        print("  - Wrong port name")
        print("  - Insufficient permissions")

if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else None

    if port is None:
        port = find_esp32_port()
        if port:
            print(f"Auto-detected ESP32 port: {port}")
        else:
            print("No ESP32 port auto-detected.")
            print("Run 1_list_all_ports.py to see what's available, then pass the port as an argument:")
            print("  python 4_connect_test.py COM5")
            sys.exit(1)

    test_port(port)
