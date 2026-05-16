"""
csi_parser.py — helpers for parsing ESP32 CSI serial output.

The ESP32 firmware prints one line per received Wi-Fi packet:
    CSI:<rssi>,<noise_floor>,<channel>,<csi_len>,<byte0>,<byte1>,...

The raw bytes are int8_t pairs: [imaginary, real] per sub-carrier.
We compute amplitude = sqrt(imag^2 + real^2) for each sub-carrier.

Limitations of ESP32 CSI:
  - 52 usable sub-carriers for 20 MHz 802.11n (LLTF gives 52, HT-LTF gives 56)
  - Raw phase is NOT calibrated — phase values drift with temperature and
    board reset.  Amplitude is more stable and is the practical signal for
    motion detection.
  - Callback fires for every received packet; in a busy Wi-Fi environment
    you might see hundreds of lines per second.
  - CSI is only measured on RECEIVED packets.  The ESP32 cannot inject
    packets and measure its own transmission.  You need a Wi-Fi transmitter
    nearby (router beacons, a second ESP32, a phone sending traffic).
  - Range for motion detection is roughly 5-10 metres through 1-2 walls.
    Beyond that, signal-to-noise ratio makes motion hard to distinguish.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CsiPacket:
    rssi: int           # Received Signal Strength Indicator (dBm, negative)
    noise_floor: int    # Ambient noise floor (dBm, negative)
    channel: int        # Wi-Fi channel number (1-13 for 2.4 GHz)
    csi_len: int        # Number of raw bytes in the CSI buffer
    raw: list[int]      # Raw int8_t bytes: [imag0, real0, imag1, real1, ...]
    amplitudes: list[float] = field(default_factory=list)  # per sub-carrier amplitude

    @property
    def snr(self) -> float:
        """Rough signal-to-noise ratio estimate in dB."""
        return float(self.rssi - self.noise_floor)

    @property
    def mean_amplitude(self) -> float:
        if not self.amplitudes:
            return 0.0
        return sum(self.amplitudes) / len(self.amplitudes)

    @property
    def num_subcarriers(self) -> int:
        return len(self.amplitudes)


def parse_line(line: str) -> Optional[CsiPacket]:
    """
    Parse a single serial line from the ESP32 firmware.
    Returns a CsiPacket or None if the line is not a CSI data line.
    """
    line = line.strip()
    if not line.startswith("CSI:"):
        return None

    try:
        parts = line[4:].split(",")
        if len(parts) < 5:
            return None

        rssi        = int(parts[0])
        noise_floor = int(parts[1])
        channel     = int(parts[2])
        csi_len     = int(parts[3])

        # Remaining values are the raw int8_t CSI bytes
        raw = [int(x) for x in parts[4: 4 + csi_len]]
        if len(raw) < csi_len:
            return None  # truncated line

        # Compute per-sub-carrier amplitude from imag/real pairs
        amplitudes: list[float] = []
        for i in range(0, len(raw) - 1, 2):
            imag = raw[i]
            real = raw[i + 1]
            amp = math.sqrt(imag ** 2 + real ** 2)
            amplitudes.append(round(amp, 3))

        return CsiPacket(
            rssi=rssi,
            noise_floor=noise_floor,
            channel=channel,
            csi_len=csi_len,
            raw=raw,
            amplitudes=amplitudes,
        )

    except (ValueError, IndexError):
        return None


def csv_header(max_subcarriers: int = 64) -> list[str]:
    """Return the CSV column names for a given maximum sub-carrier count."""
    base = ["timestamp", "rssi", "noise_floor", "channel", "csi_len",
            "num_subcarriers", "mean_amplitude", "snr"]
    amp_cols = [f"amp_{i}" for i in range(max_subcarriers)]
    return base + amp_cols


def packet_to_row(timestamp: str, pkt: CsiPacket, max_subcarriers: int = 64) -> list:
    """Flatten a CsiPacket into a CSV row (list of values)."""
    padded = pkt.amplitudes + [None] * (max_subcarriers - len(pkt.amplitudes))
    padded = padded[:max_subcarriers]
    base = [
        timestamp,
        pkt.rssi,
        pkt.noise_floor,
        pkt.channel,
        pkt.csi_len,
        pkt.num_subcarriers,
        round(pkt.mean_amplitude, 3),
        round(pkt.snr, 1),
    ]
    return base + padded
