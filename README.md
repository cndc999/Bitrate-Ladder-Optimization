Bitrate-Ladder-Optimization 
## Overview
This project focuses on bitrate ladder optimization for adaptive video streaming. It performs multi-bitrate FFmpeg encoding, evaluates video quality using PSNR, SSIM, and VMAF, applies Pareto-based optimization, and simulates ABR streaming performance.
## Key Features

- **Real-time FFmpeg Encoding**: Processing across 12 bitrate levels with PSNR, SSIM, and VMAF quality metrics.
- **Pareto Optimization**: Convex hull-based selection of the most efficient encoding ladder rungs.
- **ABR Streaming Simulation**: Buffer health monitoring with rebuffering event detection.
- **Interactive Quality Estimation**: Instant bitrate and resolution analysis using adjustable sliders.
- **Flexible Video Input**: Support for uploaded videos and built-in synthetic test video generation.

## Folder Structure
```Project
.
├── input/                   # Sample input videos
│
├── src/
│   ├── __init__.py          # Package initializer
│   ├── encoder.py           # FFmpeg encoding logic (multi-bitrate)
│   ├── ladder_optimizer.py  # Convex hull / Pareto frontier algorithm
│   ├── quality_metrics.py   # PSNR, SSIM, VMAF calculation
│   ├── stream_simulator.py  # ABR streaming simulation (buffer-based)
│   ├── video_io.py          # Video input/output (upload, synthetic gen, frame extract)
│   └── visualizer.py        # Plotly charts, heatmaps, diff maps
│
├── app.py                   # Main Streamlit UI (calls src/ only, no logic)
└── requirements.txt         # Project dependencies
```
## Installation & Setup 
### Clone the Repository 

We recommend using a virtual environment (with Python 3.12.3):
```
git clone https://github.com/cndc999/Bitrate-Ladder-Optimization
cd <project-folder>
```
### Create a Virtual Environment
```
python -m venv .venv 
```

### Activate the Virtual Environment
For Linux
```
source .venv/bin/activate
```
For Windows
```
.venv\Scripts\activate
```
### Install Dependencies
```
pip install -r requirements.txt
```
```
pip install streamlit numpy plotly opencv-python-headless scikit-image scipy pandas
``` 

### How to Run 
Launch the interactive dashboard:
```
streamlit run app.py
```



