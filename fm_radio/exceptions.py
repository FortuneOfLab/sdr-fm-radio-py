"""Custom exception hierarchy for the FM receiver system."""


class FMReceiverError(Exception):
    """Base exception for the FM receiver system."""


class SDRDeviceError(FMReceiverError):
    """RTL-SDR device errors (init, tuning, gain)."""


class AudioOutputError(FMReceiverError):
    """Audio output errors (PyAudio init, stream)."""


class RecordingError(FMReceiverError):
    """Recording errors (file open, write)."""


class DemodulationError(FMReceiverError):
    """FM demodulation errors (DSP processing)."""
