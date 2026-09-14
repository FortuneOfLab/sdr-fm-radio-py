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

- Python 3.9+ (3.11+ to use a `stations.toml` of your own)
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

- `list` → Show the preset stations (tunable by number)
- `list all` / `list <area>` → Browse the full nationwide catalogue
- `search <text>` → Find a station by name, transmitter site or frequency
- `stereo on` → Enable stereo demodulation
- `stereo off` / `mono` → Enable mono demodulation
- `record start` → Start recording (file is auto-named)
- `record stop` → Stop recording
- `agc on` → Enable Automatic Gain Control
- `agc off` → Enable manual gain control
- `gain <value>` → Set manual gain level
- `<station_num>` or `<freq_MHz>` → Tune to a station
- `q` → Quit the program

## Station list

The receiver ships with every FM, wide-FM and NHK-FM transmitter in Japan —
983 transmitters from 145 broadcasters, listed under the name each station
is actually known by (`TOKYO FM`, not `エフエム東京`).

`fm_radio/data/stations.json` is a generated snapshot; do not edit it by
hand. Regenerate it when the upstream lists change:

```bash
python tools/fetch_stations.py
```

It merges three primary sources: the
[総務省 list](https://www.soumu.go.jp/menu_seisaku/ictseisaku/housou_suishin/fm-list.html)
of commercial FM and wide-FM transmitters, the JSON behind
[NHK's own frequency page](https://www.nhk.or.jp/radio/info/frequency.html?ch=fm),
and [radiko](https://radiko.jp/)'s station list for the brand names.

### Adding your own stations

Community FM stations are not in the bundled list, and you will want your own
presets. Create `stations.toml` — `%APPDATA%m_radio\stations.toml` on
Windows, `~/.config/fm_radio/stations.toml` elsewhere, or anywhere you like
with `--stations PATH`:

```toml
# Add a station the bundled list does not have
[[station]]
name = "レインボータウンFM"
freq_mhz = 79.2
site = "江東"
area = "関東"
favorite = true

# Correct or hide a bundled entry
[[override]]
match_name = "TOKYO FM"
match_site = "八王子"
hidden = true
```

`favorite = true` puts a station in the preset list that `list` shows and the
numeric tune command indexes into. As soon as you mark any favourite, the
shipped preset list is replaced by yours. Reading this file needs Python 3.11
or newer (`tomllib`); on older versions the bundled list still loads.

## Code Structure

- `fm_receiver.py` → Main script containing all functionality
- `fm_radio/stations.py` → Station catalogue: bundled snapshot + user overrides
- `DeemphasisIIRFilter` → Implements FM **de-emphasis filtering**
- `LowpassFilter`, `BandpassFilter` → Filter implementations for processing signals
- `PLL` → Phase-Locked Loop for FM demodulation
- `SDRReceiver` → Handles RTL-SDR data acquisition
- `FMDemodulator` → Standard FM demodulation
- `FMDemodulatorLight` → Optimized version for lightweight processing
- `AudioOutput` → Handles **audio playback and recording**
- `CommandLineInterface` → Allows user input commands
- `FMReceiverController` → Main controller integrating all components

## Recording files

Auto-named recordings (both audio and raw IQ) are written to the
`recordings/` directory. Each recording session gets a `.json` metadata
sidecar next to the WAV containing the capture parameters (centre
frequency, gain, sample rate), start/stop timestamps, drop counts and —
since long recordings rotate at the WAV 4 GiB limit — the full list of
`.partNNN.wav` files belonging to the session.

## Development

### Running the tests

```bash
pip install -r requirements-dev.txt
pytest            # full suite (~30 s, includes the end-to-end quality gate)
pytest -m "not slow"   # quick iteration without the end-to-end test
```

The test suite injects fakes for `pyaudio` and `rtlsdr` (see
`tests/conftest.py`), so no sound device or RTL-SDR driver is required —
it runs unmodified on CI. GitHub Actions runs the suite on Ubuntu and
Windows for every push to `main` and every pull request.

## License

This project is licensed under the **MIT License**.

## Author

[FortuneOfLab]

