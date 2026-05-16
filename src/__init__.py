from .encoder import encode_with_ffmpeg, get_actual_bitrate
from .quality_metrics import compute_quality, compute_vmaf_approx
from .ladder_optimizer import build_optimized_ladder, BITRATE_POOL, STANDARD_LADDER
from .stream_simulator import simulate_streaming
from .video_io import make_synthetic_video, extract_frame, compute_diff_map
from .visualizer import (
    plot_quality_curves,
    plot_efficiency_heatmap,
    plot_ladder_comparison,
    plot_streaming_simulation,
)
