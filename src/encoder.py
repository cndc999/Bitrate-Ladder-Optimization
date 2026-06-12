"""
Module 1 — Encoder
Pipeline step: Source video -> Encode at multiple bitrates.
Encodes the source at every candidate bitrate with FFmpeg (H.264),
then probes the ACTUAL bitrate and file size of each output.
"""

import subprocess
import json
import os

# Candidate bitrates to encode (kbps) — the pool the ladder is built from
CANDIDATE_BITRATES = [200, 400, 800, 1200, 1800, 2500, 3500, 5000]


def resolution_for_bitrate(bitrate_kbps: int) -> tuple[int, str]:
    """Pick a sensible resolution for each bitrate (mirrors real ladders)."""
    if bitrate_kbps < 400:
        return 240, "240p"
    elif bitrate_kbps < 900:
        return 360, "360p"
    elif bitrate_kbps < 1800:
        return 480, "480p"
    elif bitrate_kbps < 4000:
        return 720, "720p"
    else:
        return 1080, "1080p"


def encode_video(src_path: str, bitrate_kbps: int, height: int,
                 output_path: str, max_seconds: int = 5) -> tuple[bool, str]:
    """
    Encode the source to one explicit (bitrate, resolution) pair.
    -maxrate/-bufsize cap the encoder so the actual bitrate stays close
    to the target. Returns (success, ffmpeg_error_message).
    """
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-t", str(max_seconds),
        "-vf", f"scale=-2:{height}",
        "-c:v", "libx264", "-preset", "veryfast",
        "-b:v", f"{bitrate_kbps}k",
        "-maxrate", f"{bitrate_kbps}k",
        "-bufsize", f"{bitrate_kbps}k",
        "-pix_fmt", "yuv420p",
        "-an", output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except FileNotFoundError:
        return False, "FFmpeg not found. Install FFmpeg and add it to PATH."
    except subprocess.TimeoutExpired:
        return False, "FFmpeg timed out. Try a shorter encode duration."
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-4:])
        return False, tail or "FFmpeg failed with an unknown error."
    return True, ""


def probe_actual_bitrate(enc_path: str) -> float:
    """Measure the ACTUAL bitrate (kbps) of an encoded file via ffprobe."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", enc_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        return float(data["format"]["bit_rate"]) / 1000
    except Exception:
        return 0.0


def file_size_kb(path: str) -> float:
    """File size in KB — higher bitrate means a larger file."""
    return os.path.getsize(path) / 1024 if os.path.exists(path) else 0.0
