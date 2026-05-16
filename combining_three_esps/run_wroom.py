"""
run_wroom.py — build and flash LEFT or RIGHT WROOM nodes.

Usage:
    python combining_three_esps/run_wroom.py left    # flash LEFT  (COM9)
    python combining_three_esps/run_wroom.py right   # flash RIGHT (COM8)
    python combining_three_esps/run_wroom.py left  --build    # build only
    python combining_three_esps/run_wroom.py right --monitor  # flash + monitor

Tip: both boards are buggy and may need the BOOT button held during upload.
Hold BOOT when you see 'Connecting......' dots appear, then release.
"""

import argparse
import os
import subprocess
import sys

PIO_EXE      = os.path.join(os.path.expanduser("~"), ".platformio", "penv", "Scripts", "pio.exe")
FIRMWARE_DIR = os.path.join(os.path.dirname(__file__), "firmware", "wroom")


def find_pio() -> str:
    if os.path.isfile(PIO_EXE):
        return PIO_EXE
    from shutil import which
    found = which("pio") or which("platformio")
    if found:
        return found
    print("ERROR: PlatformIO CLI not found.")
    sys.exit(1)


def run(cmd: list[str], cwd: str) -> int:
    print(f"\n> {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=cwd).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Flash LEFT or RIGHT WROOM node.")
    parser.add_argument("node", choices=["left", "right"],
                        help="Which node to flash: left (COM9) or right (COM8)")
    parser.add_argument("--build",   action="store_true")
    parser.add_argument("--monitor", action="store_true")
    args = parser.parse_args()

    env  = args.node.upper()   # LEFT or RIGHT
    port = "COM9" if args.node == "left" else "COM8"
    pio  = find_pio()

    print(f"PlatformIO : {pio}")
    print(f"Firmware   : {FIRMWARE_DIR}")
    print(f"Environment: {env}  ({port}, ESP32-WROOM)")

    print(f"\n=== BUILDING {env} firmware ===")
    rc = run([pio, "run", "-e", env], FIRMWARE_DIR)
    if rc != 0:
        print("\nBUILD FAILED.")
        sys.exit(rc)
    print("\nBUILD OK")

    if args.build:
        sys.exit(0)

    print(f"\n=== FLASHING to {port} ({env}) ===")
    print("Tip: hold BOOT button while 'Connecting......' appears if it hangs.")
    rc = run([pio, "run", "-e", env, "--target", "upload"], FIRMWARE_DIR)
    if rc != 0:
        print(f"\nFLASH FAILED. Check {port} is the right port and try BOOT button.")
        sys.exit(rc)
    print(f"\nFLASH OK — {env} node ready on {port}.")

    if args.monitor:
        run([pio, "device", "monitor", "-e", env], FIRMWARE_DIR)


if __name__ == "__main__":
    main()
