"""
csi_logger.py — reads ESP32 CSI data over serial, saves to timestamped CSV.

Usage:
    python csi_logger.py                   # auto-detect port
    python csi_logger.py --port COM4       # specify port manually
    python csi_logger.py --port COM4 --baud 115200 --out ../data

The script prints a live summary to the terminal and writes a CSV file
to the data/ folder.  Press Ctrl+C to stop cleanly.

Requirements:
    pip install pyserial
    (numpy is optional — only needed if you add FFT analysis later)
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys
import time

import serial
import serial.tools.list_ports

# Make sure csi_parser is importable from the same directory
sys.path.insert(0, os.path.dirname(__file__))
import csi_parser


# ── port detection ────────────────────────────────────────────────────────────

# USB-to-serial chip names that appear on typical ESP32 dev boards.
# CP2102 = most ESP32-WROOM breakout boards
# CH340  = cheap Chinese clones
# FTDI   = some older boards
_ESP32_USB_CHIPS = ["cp210", "ch340", "ftdi", "uart", "esp32", "silicon labs"]


def find_esp32_port() -> str | None:
    """Return the first COM port that looks like an ESP32 board."""
    for port in serial.tools.list_ports.comports():
        desc = (port.description or "").lower()
        hwid = (port.hwid or "").lower()
        if any(chip in desc or chip in hwid for chip in _ESP32_USB_CHIPS):
            return port.device
    return None


def list_all_ports() -> None:
    """Print every available serial port (for debugging)."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("  No COM ports found at all. Is the USB cable plugged in?")
        return
    for p in ports:
        print(f"  {p.device:8s}  {p.description}  [{p.hwid}]")


# ── CSV logging ───────────────────────────────────────────────────────────────

MAX_SUBCARRIERS = 64  # pad every row to this width so the CSV stays rectangular


def make_output_path(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"csi_{stamp}.csv")


# ── main loop ─────────────────────────────────────────────────────────────────

def run(port: str, baud: int, out_dir: str) -> None:
    out_path = make_output_path(out_dir)
    print(f"Saving CSI data to: {out_path}")
    print(f"Opening serial port {port} at {baud} baud...")
    print("Press Ctrl+C to stop.\n")

    total_lines   = 0   # all serial lines received
    csi_packets   = 0   # lines successfully parsed as CSI
    bad_lines     = 0   # CSI-prefixed lines that failed to parse
    last_print_t  = time.time()
    packets_since_last_print = 0

    try:
        ser = serial.Serial(port, baud, timeout=2)
    except serial.SerialException as exc:
        print(f"ERROR: Could not open {port}: {exc}")
        sys.exit(1)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csi_parser.csv_header(MAX_SUBCARRIERS))

        try:
            while True:
                try:
                    raw = ser.readline()
                except serial.SerialException as exc:
                    print(f"\nSerial error: {exc}")
                    break

                if not raw:
                    continue  # timeout with no data

                try:
                    line = raw.decode("ascii", errors="replace").strip()
                except Exception:
                    continue

                total_lines += 1

                # Pass through STATUS and INFO lines so the user sees them
                if line.startswith(("STATUS:", "INFO:", "ERROR:")):
                    print(f"[ESP32] {line}")
                    continue

                if not line.startswith("CSI:"):
                    continue  # ignore unrecognised lines

                timestamp = datetime.datetime.now().isoformat(timespec="milliseconds")
                pkt = csi_parser.parse_line(line)

                if pkt is None:
                    bad_lines += 1
                    print(f"  PARSE FAIL: {line[:80]}")
                    continue

                writer.writerow(csi_parser.packet_to_row(timestamp, pkt, MAX_SUBCARRIERS))
                f.flush()  # write immediately so data isn't lost on Ctrl+C

                csi_packets += 1
                packets_since_last_print += 1

                # Print a live summary every second
                now = time.time()
                if now - last_print_t >= 1.0:
                    rate = packets_since_last_print / (now - last_print_t)
                    print(
                        f"  pkts={csi_packets:6d}  "
                        f"rate={rate:5.1f}/s  "
                        f"rssi={pkt.rssi:4d}dBm  "
                        f"ch={pkt.channel:2d}  "
                        f"subcarriers={pkt.num_subcarriers:2d}  "
                        f"mean_amp={pkt.mean_amplitude:6.2f}  "
                        f"snr={pkt.snr:.1f}dB"
                    )
                    last_print_t = now
                    packets_since_last_print = 0

        except KeyboardInterrupt:
            pass

    ser.close()
    print(f"\nDone. {csi_packets} CSI packets saved to {out_path}")
    if bad_lines:
        print(f"  ({bad_lines} lines failed to parse — check firmware output format)")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log ESP32 CSI data to CSV over serial."
    )
    parser.add_argument(
        "--port", "-p",
        default=None,
        help="COM port (e.g. COM4).  Auto-detected if omitted.",
    )
    parser.add_argument(
        "--baud", "-b",
        type=int,
        default=115200,
        help="Baud rate (default: 115200, must match firmware)",
    )
    parser.add_argument(
        "--out", "-o",
        default=os.path.join(os.path.dirname(__file__), "..", "data"),
        help="Output directory for CSV files (default: ../data)",
    )
    args = parser.parse_args()

    port = args.port
    if port is None:
        print("No --port specified.  Scanning for ESP32...")
        port = find_esp32_port()
        if port is None:
            print("\nNo ESP32-like port found.  Available ports:")
            list_all_ports()
            print("\nTip: specify the port manually with --port COM4")
            sys.exit(1)
        print(f"Auto-detected: {port}")

    run(port, args.baud, args.out)


if __name__ == "__main__":
    main()
