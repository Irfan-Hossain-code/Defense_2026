"""
bridge_middle.py — run this on the Mac to stream MIDDLE node data to Windows.

The Mac reads the MIDDLE ESP32's serial output and forwards every line to
the Windows laptop over a TCP socket on port 5555.  Both machines must be
on the same Wi-Fi network (the OnePlus hotspot).

Usage (on Mac):
    python3 bridge_middle.py

Then on Windows, run_all_three.py with the Mac's hotspot IP:
    python run_all_three.py --middle-host 10.75.241.XXX

Find the Mac's IP:  ifconfig en0  (look for inet in the 10.75.241.x range)
Or check the OnePlus hotspot connected devices list.
"""

import socket
import sys
import threading
import time

import serial
import serial.tools.list_ports

SERIAL_PORT = "/dev/cu.usbserial-1130"
BAUD        = 115200
BRIDGE_PORT = 5555   # TCP port Windows connects to


def find_hotspot_ip() -> str:
    """Return this machine's IP on the 10.75.241.x hotspot network."""
    import socket as _s
    for iface_ip in _get_all_ips():
        if iface_ip.startswith("10.75.241."):
            return iface_ip
    return "0.0.0.0"   # bind to all interfaces as fallback


def _get_all_ips() -> list[str]:
    try:
        import subprocess
        out = subprocess.check_output(["ifconfig"], text=True)
        import re
        return re.findall(r"inet (\d+\.\d+\.\d+\.\d+)", out)
    except Exception:
        return []


def main() -> None:
    # ── Open serial ──────────────────────────────────────────────────────────
    print(f"Opening serial: {SERIAL_PORT} @ {BAUD} baud")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD, timeout=2)
    except serial.SerialException as exc:
        print(f"ERROR: cannot open {SERIAL_PORT}: {exc}")
        print()
        print("Available ports:")
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}: {p.description}")
        sys.exit(1)

    print("Serial OK.")

    # ── Start TCP server ──────────────────────────────────────────────────────
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", BRIDGE_PORT))
    server.listen(1)

    my_ip = find_hotspot_ip()
    print()
    print(f"Bridge listening on port {BRIDGE_PORT}.")
    print(f"This Mac's hotspot IP appears to be: {my_ip}")
    print()
    print(f"On the Windows laptop run:")
    print(f"  python combining_three_esps/run_all_three.py --middle-host {my_ip}")
    print()
    print("Waiting for Windows to connect...")

    while True:
        try:
            conn, addr = server.accept()
        except KeyboardInterrupt:
            break

        print(f"Windows connected from {addr[0]}:{addr[1]}")

        # Forward serial → socket until the connection drops
        try:
            while True:
                line = ser.readline()
                if not line:
                    continue
                try:
                    conn.sendall(line)
                except (BrokenPipeError, ConnectionResetError):
                    break
        except Exception as exc:
            print(f"Connection error: {exc}")
        finally:
            conn.close()
            print(f"Windows disconnected. Waiting for reconnect...")

    ser.close()
    server.close()


if __name__ == "__main__":
    main()
