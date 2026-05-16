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
├── input/                 # Input audio files

├── src/
│   ├── __init__.py        # Package initializer
│   ├── app.py             # Main Streamlit UI application
│   ├── analytics.py       # Metrics calculation (SNR, compression ratio, quality)
│   ├── audio_io.py        # Audio input/output handling (record, read, write)
│   ├── codec_engine.py    # Encoding/decoding logic (DPCM)
│   ├── vad_handler.py     # Voice Activity Detection processing
│   ├── visualizer.py      # Plotting and visualization tools
├── .venv/                 # Virtual environment
├── requirement.txt        # Project dependencies
├── README.md              # Project documentation
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
cd src
streamlit run app.py
```



