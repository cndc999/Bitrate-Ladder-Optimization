Bitrate-Ladder-Optimization 
## Overview
This project implements a low-bitrate speech communication system using Differential Pulse Code Modulation (DPCM). Our goal is to transmit high-quality voice data over limited bandwidth by reducing data redundancy and utilizing Voice Activity Detection (VAD).

## Key Features

- **Signal-to-Noise Ratio (SNR)**: Quantifying the clarity of reconstructed audio.
- **Compression Ratio**: Measuring bandwidth savings.
- **Spectrogram Analysis**: Visual comparison of frequency loss using librosa.
- **Latency**: Measuring the delay in real-time processing mode.

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
pip install pyaudio sounddevice scipy numpy librosa matplotlib streamlit
``` 

### How to Run 
Launch the interactive dashboard:
```
cd src
streamlit run app.py
```



