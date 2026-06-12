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
├── input/                   # Sample input videos (place test clips here)
│
├── src/
│   ├── __init__.py          # Package initializer
│   ├── encoder.py           # FFmpeg multi-bitrate encoding + ffprobe probing of the ACTUAL bitrate and file size
│   │                     
│   ├── quality.py           # PSNR / SSIM, measured at the source resolution for fair cross-resolution comparison
│   │                        
│   ├── ladder.py            # Ladder design: convex hull / Pareto frontier selection vs a fixed standard ladder
│   │                        
│   ├── visualization.py     # Matplotlib charts: quality curves, target-vs-actual, file size, ladder, simulation
│   │                        
│   └── simulator.py         # Buffer-based ABR streaming simulation over configurable bandwidth traces
│                        
│
├── app.py                   # Main Streamlit UI (5 tabs, one per pipeline step)
├── README.md                # This file
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

### How to Run 
Launch the interactive dashboard:
```
streamlit run app.py
```



