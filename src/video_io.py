"""
video_io.py
───────────
Video input/output helpers:
  - Synthetic test video generation
  - Frame extraction from encoded files
  - Pixel-level difference map computation
"""

import numpy as np
import cv2


def make_synthetic_video(path: str, width: int = 1280, height: int = 720,
                         fps: int = 30, duration: int = 3) -> None:
    """
    Generate a synthetic test video designed to stress video codecs:
      - Animated gradient background  (slow-changing, easy to compress)
      - Moving geometric objects      (medium detail)
      - Per-frame random noise        (high-frequency, hard to compress)
      - Burned-in frame counter       (sharp edges)

    Args:
        path:     Output file path (.mp4).
        width:    Frame width in pixels.
        height:   Frame height in pixels.
        fps:      Frames per second.
        duration: Video length in seconds.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for i in range(fps * duration):
        t = i / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Animated gradient background
        for y in range(height):
            r = int(110 + 80 * np.sin(t + y / height * np.pi))   # 110±80 → [30, 190]
            g = int(80  + 60 * np.cos(t * 0.7))                   # 80±60  → [20, 140]
            b = int(140 + 100 * np.sin(t * 1.3 + y / height * 2)) # 140±100 → [40, 240]
            frame[y, :] = [b, g, r]

        # Moving circles
        cx = int(width / 2 + width / 3 * np.sin(t * 2))
        cy = int(height / 2 + height / 4 * np.cos(t * 1.5))
        cv2.circle(frame, (cx, cy), 80,   (0, 200, 255), -1)
        cv2.circle(frame, (cx + 50, cy - 30), 40, (255, 100, 0), -1)

        # Frame counter (sharp edges = compression stress)
        cv2.putText(frame, f"FRAME {i:04d}", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # High-frequency noise
        noise = np.random.randint(0, 15, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)

        out.write(frame)

    out.release()


def extract_frame(video_path: str, frame_index: int = 0) -> np.ndarray | None:
    """
    Extract a single frame from a video file.

    Args:
        video_path:  Path to the video.
        frame_index: Which frame to read (default: first frame).

    Returns:
        BGR numpy array, or None if extraction fails.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def compute_diff_map(frame_orig: np.ndarray, frame_enc: np.ndarray,
                     amplify: float = 5.0) -> np.ndarray:
    """
    Compute an amplified absolute-difference map between two frames.
    Useful for visualising where compression artefacts are concentrated.

    Args:
        frame_orig: Reference frame (BGR, any resolution).
        frame_enc:  Encoded frame (BGR; will be resized to match orig if needed).
        amplify:    Multiplicative amplification factor (default ×5).

    Returns:
        RGB numpy array (uint8) of the amplified difference.
    """
    h, w = frame_enc.shape[:2]
    ref = cv2.resize(frame_orig, (w, h))
    diff = cv2.absdiff(ref, frame_enc)
    diff_amp = np.clip(diff.astype(np.float32) * amplify, 0, 255).astype(np.uint8)
    return cv2.cvtColor(diff_amp, cv2.COLOR_BGR2RGB)
