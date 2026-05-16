"""
run_s3.py — build and flash the ESP32-S3 MIDDLE node on COM7.

Usage:
    python combining_three_esps/run_s3.py           # build + flash
    python combining_three_esps/run_s3.py --build   # build only (no flash)
    python combining_three_esps/run_s3.py --monitor # flash then open serial monitor

After flashing, the S3 can be unplugged from the laptop and powered from
a USB power bank.  It will auto-connect to Wi-Fi on boot and start
self-probing without any laptop connection needed.
"""

import argparse
import os
import subprocess
import sys

PIO_EXE     = os.path.join(os.path.expanduser("~"), ".platformio", "penv", "Scripts", "pio.exe")
FIRMWARE_DIR = os.path.join(os.path.dirname(__file__), "firmware", "s3")
ENV_NAME    = "MIDDLE"


def find_pio() -> str:
    if os.path.isfile(PIO_EXE):
        return PIO_EXE
    from shutil import which
    found = which("pio") or which("platformio")
    if found:
        return found
    print("ERROR: PlatformIO CLI not found.")
    print("  Install the PlatformIO IDE extension in VS Code, then rerun.")
    sys.exit(1)


def run(cmd: list[str], cwd: str) -> int:
    print(f"\n> {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=cwd).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Flash ESP32-S3 MIDDLE node.")
    parser.add_argument("--build",   action="store_true", help="Build only, no flash.")
    parser.add_argument("--monitor", action="store_true", help="Open serial monitor after flash.")
    args = parser.parse_args()

    pio = find_pio()
    print(f"PlatformIO : {pio}")
    print(f"Firmware   : {FIRMWARE_DIR}")
    print(f"Environment: {ENV_NAME}  (COM7, ESP32-S3 MIDDLE node)")

    # ── Build ─────────────────────────────────────────────────────────────────
    print("\n=== BUILDING S3 firmware ===")
    rc = run([pio, "run", "-e", ENV_NAME], FIRMWARE_DIR)
    if rc != 0:
        print("\nBUILD FAILED.")
        print("If you see 'MissingPackageManifestError' run the root run_platformio.py")
        print("once first so it downloads the ESP32 toolchain, then retry.")
        sys.exit(rc)
    print("\nBUILD OK")

    if args.build:
        print("--build flag set, skipping flash.")
        sys.exit(0)

    # ── Flash ─────────────────────────────────────────────────────────────────
    print("\n=== FLASHING to COM7 (ESP32-S3 MIDDLE) ===")
    print("Tip: if it hangs at 'Connecting...', hold BOOT on the S3 while dots appear.")
    rc = run([pio, "run", "-e", ENV_NAME, "--target", "upload"], FIRMWARE_DIR)
    if rc != 0:
        print("\nFLASH FAILED.")
        print("  - Check COM7 is the right port (run python port_test.py)")
        print("  - Hold BOOT button during upload if it times out")
        print("  - If board is a different S3 variant, change 'board' in firmware/s3/platformio.ini")
        sys.exit(rc)
    print("\nFLASH OK — S3 is ready.")
    print("You can now unplug COM7 and power it from a USB power bank.")
    print("It will auto-start and connect to Wi-Fi on boot.")

    # ── Monitor (optional) ────────────────────────────────────────────────────
    if args.monitor:
        print("\n=== SERIAL MONITOR (Ctrl+C to exit) ===")
        run([pio, "device", "monitor", "-e", ENV_NAME], FIRMWARE_DIR)


if __name__ == "__main__":
    main()
