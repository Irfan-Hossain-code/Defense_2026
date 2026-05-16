"""
run_platformio.py — build, flash, and optionally monitor the ESP32 CSI firmware.

Usage:
    python run_platformio.py           # build + flash
    python run_platformio.py --build   # build only (no flash)
    python run_platformio.py --monitor # flash then open serial monitor

The script finds the PlatformIO CLI that VS Code installs automatically,
so you do NOT need to install anything extra.
"""

import argparse
import os
import subprocess
import sys

# ── PlatformIO CLI location ───────────────────────────────────────────────────
# VS Code's PlatformIO extension installs pio here on Windows.
PIO_EXE = os.path.join(os.path.expanduser("~"), ".platformio", "penv", "Scripts", "pio.exe")

# The PlatformIO project to build/flash
FIRMWARE_DIR = os.path.join(os.path.dirname(__file__), "csi_test", "firmware")


def find_pio() -> str:
    """Return path to pio.exe, or exit with a helpful message."""
    if os.path.isfile(PIO_EXE):
        return PIO_EXE
    # Fallback: maybe pio is on PATH
    from shutil import which
    on_path = which("pio") or which("platformio")
    if on_path:
        return on_path
    print("ERROR: Cannot find the PlatformIO CLI (pio.exe).")
    print()
    print("Fix options:")
    print("  1. Open VS Code, install the 'PlatformIO IDE' extension, then rerun.")
    print("  2. Or install it standalone:")
    print("       pip install platformio")
    sys.exit(1)


def run(cmd: list[str], cwd: str) -> tuple[int, str]:
    """Run a command, stream its output live, return (exit_code, combined_output)."""
    print(f"\n> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False,
                            stdout=None, stderr=None)
    return result.returncode, ""


def run_captured(cmd: list[str], cwd: str) -> tuple[int, str]:
    """Run a command silently, return (exit_code, combined_output)."""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def fix_missing_packages(pio: str) -> None:
    """
    Detect and repair the MissingPackageManifestError that happens when
    a toolchain package downloaded but its package.json is absent.
    Deletes the broken package directory and reinstalls the platform.
    """
    import shutil
    packages_dir = os.path.join(os.path.expanduser("~"), ".platformio", "packages")
    platforms_dir = os.path.join(os.path.expanduser("~"), ".platformio", "platforms")
    repaired = False

    if os.path.isdir(packages_dir):
        for pkg in os.listdir(packages_dir):
            pkg_path = os.path.join(packages_dir, pkg)
            manifest = os.path.join(pkg_path, "package.json")
            if os.path.isdir(pkg_path) and not os.path.isfile(manifest):
                print(f"  Removing corrupted package (missing package.json): {pkg}")
                shutil.rmtree(pkg_path, ignore_errors=True)
                repaired = True

    esp_platform = os.path.join(platforms_dir, "espressif32")
    if repaired and os.path.isdir(esp_platform):
        print("  Removing espressif32 platform for clean reinstall...")
        shutil.rmtree(esp_platform, ignore_errors=True)

    if repaired:
        print("  Running: pio platform install espressif32  (downloads ~300 MB, be patient)...")
        rc, _ = run_captured([pio, "platform", "install", "espressif32"], os.path.expanduser("~"))
        if rc == 0:
            print("  Platform reinstalled successfully.")
        else:
            print("  Platform install failed. Check your internet connection and try again.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and flash ESP32 CSI firmware.")
    parser.add_argument("--build",   action="store_true", help="Build only, do not flash.")
    parser.add_argument("--monitor", action="store_true", help="Open serial monitor after flashing.")
    args = parser.parse_args()

    pio = find_pio()

    if not os.path.isdir(FIRMWARE_DIR):
        print(f"ERROR: Firmware directory not found: {FIRMWARE_DIR}")
        sys.exit(1)

    print(f"Using PlatformIO: {pio}")
    print(f"Firmware directory: {FIRMWARE_DIR}")

    # ── Step 1: Build (with auto-repair on MissingPackageManifestError) ───────
    print("\n=== BUILDING firmware ===")
    rc, _ = run([pio, "run"], FIRMWARE_DIR)
    if rc != 0:
        # Check if the failure is the known corrupted-package issue and auto-fix it
        print("\nBuild failed. Checking for corrupted packages...")
        fix_missing_packages(pio)
        print("\nRetrying build after repair...")
        rc, _ = run([pio, "run"], FIRMWARE_DIR)

    if rc != 0:
        print(f"\nBUILD FAILED (exit code {rc}).")
        print("If you see 'MissingPackageManifestError', your internet may have")
        print("cut out during a previous download. Delete ~/.platformio/packages")
        print("and ~/.platformio/platforms and run this script again.")
        print("Otherwise, read the compiler error above — it is a code issue.")
        sys.exit(rc)
    print("\nBUILD OK")

    if args.build:
        print("--build flag set, skipping flash.")
        sys.exit(0)

    # ── Step 2: Upload (flash) ────────────────────────────────────────────────
    print("\n=== FLASHING firmware to ESP32 ===")
    print("Tip: if this hangs at 'Connecting...', hold the BOOT button on the")
    print("     ESP32 while dots appear, then release it.")
    rc, _ = run([pio, "run", "--target", "upload"], FIRMWARE_DIR)
    if rc != 0:
        print(f"\nFLASH FAILED (exit code {rc}).")
        print("Common causes:")
        print("  - Wrong COM port in platformio.ini (upload_port = COM8).")
        print("    Run: python port_test.py  to check which port the board is on.")
        print("  - Board not in bootloader mode: hold BOOT button during upload.")
        print("  - USB cable is charge-only (no data lines).")
        sys.exit(rc)
    print("\nFLASH OK")

    # ── Step 3: Serial monitor (optional) ────────────────────────────────────
    if args.monitor:
        print("\n=== OPENING serial monitor (Ctrl+C to exit) ===")
        run([pio, "device", "monitor"], FIRMWARE_DIR)  # noqa: F841


if __name__ == "__main__":
    main()
