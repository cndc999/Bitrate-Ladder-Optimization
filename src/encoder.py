"""
encoder.py
──────────
FFmpeg encoding logic: multi-bitrate video encoding and bitrate probing.
"""

import subprocess
import json


def encode_with_ffmpeg(src_path: str, bitrate_kbps: int, height: int,
                       output_path: str, codec: str = "libx264") -> bool:
    """
    Encode a video to a target bitrate and resolution using FFmpeg.

    Args:
        src_path:     Path to the source video file.
        bitrate_kbps: Target video bitrate in kbps.
        height:       Target vertical resolution (width scaled automatically).
        output_path:  Path for the encoded output file.
        codec:        FFmpeg video codec (default: libx264).

    Returns:
        True if encoding succeeded, False otherwise.
    """
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-vf", f"scale=-2:{height}",
        "-c:v", codec, "-preset", "veryfast",
        "-b:v", f"{bitrate_kbps}k",
        "-maxrate", f"{int(bitrate_kbps * 1.5)}k",
        "-bufsize", f"{bitrate_kbps * 2}k",
        "-an", "-f", "mp4", output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    return result.returncode == 0


def get_actual_bitrate(enc_path: str) -> float:
    """
    Probe the actual bitrate of an encoded file using ffprobe.

    Args:
        enc_path: Path to the encoded video file.

    Returns:
        Actual bitrate in kbps, or 0.0 if probing fails.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", enc_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    try:
        data = json.loads(result.stdout)
        return float(data["format"]["bit_rate"]) / 1000
    except Exception:
        return 0.0


def resolution_for_bitrate(bitrate_kbps: int) -> tuple[int, str]:
    """
    Heuristic: pick a sensible resolution for a given bitrate.

    Returns:
        (height_px, label) — e.g. (720, "720p")
    """
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
