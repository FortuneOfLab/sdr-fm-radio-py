# SDR FM Receiver in Python

## Overview

This project implements an **FM receiver system** using **Software-Defined Radio (SDR)** with an **RTL-SDR** device. The program demodulates FM signals and plays audio output using **PyAudio**.

## Features

- Receive FM broadcasts via RTL-SDR
- Demodulation using PLL and various filters
- Support for **stereo and mono** output
- **Automatic Gain Control (AGC) support**
- Real-time **audio output and recording**

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.7+
- RTL-SDR device and its drivers

### Install Dependencies

Run the following command to install required Python packages:

```bash
pip install numpy scipy pyaudio samplerate pyrtlsdr numba
```

## Usage

### Run the FM Receiver

#### Standard mode:

```bash
python fm_receiver.py
```

#### Lightweight mode (optimized for lower CPU usage):

```bash
python fm_receiver.py --light
```

### Interactive Commands (while running)

- `list` → Show available FM stations
- `stereo on` → Enable stereo demodulation
- `stereo off` / `mono` → Enable mono demodulation
- `record start` → Start recording (file is auto-named)
- `record stop` → Stop recording
- `agc on` → Enable Automatic Gain Control
- `agc off` → Enable manual gain control
- `gain <value>` → Set manual gain level
- `<station_num>` or `<freq_MHz>` → Tune to a station
- `q` → Quit the program

## Code Structure

- `fm_receiver.py` → Main script containing all functionality
- `DeemphasisIIRFilter` → Implements FM **de-emphasis filtering**
- `LowpassFilter`, `BandpassFilter` → Filter implementations for processing signals
- `PLL` → Phase-Locked Loop for FM demodulation
- `SDRReceiver` → Handles RTL-SDR data acquisition
- `FMDemodulator` → Standard FM demodulation
- `FMDemodulatorLight` → Optimized version for lightweight processing
- `AudioOutput` → Handles **audio playback and recording**
- `CommandLineInterface` → Allows user input commands
- `FMReceiverController` → Main controller integrating all components

## License

This project is licensed under the **MIT License**.

## Author

[FortuneOfLab]

