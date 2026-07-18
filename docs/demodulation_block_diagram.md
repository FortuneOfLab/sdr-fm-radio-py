# FM Stereo Demodulation Block Diagram

Reflects the current pipeline (2026-07). The standard demodulator
defaults to an **arctan discriminator** (the legacy PLL path is
retained behind `MAIN_DEMOD_USE_PLL`), the pilot is extracted by a
**phase-continuous heterodyne + complex lowpass** (no FFT Hilbert),
and every resampling / decimation stage is **stateful across blocks**.

## FMDemodulator (Standard)

```mermaid
flowchart TD
    %% ============================================================
    %% Stage 1: IQ Front-End
    %% ============================================================
    IQ["IQ Samples\n(1.024 MHz complex)"]
    IQ --> DC["DC Offset Removal\n(EMA α=0.01)"]
    DC --> IQLPF["IQ Lowpass Filter\nButterworth N=5, SOS + carried state\nfc = 200 kHz"]
    IQLPF --> DEMOD["FM Demodulation\ndefault: arctan discriminator\nangle(x[n]·conj(x[n-1]))\n(PLL selectable via MAIN_DEMOD_USE_PLL)"]
    DEMOD --> RS1["StatefulResampler\n3 : 16 (Kaiser β=10)\n1.024 MHz → 192 kHz\n(grid-aligned, exact prefix)"]

    RS1 --> COMP["Composite Signal\n192 kHz"]

    %% ============================================================
    %% Stage 2: Stereo Demodulation (at 192 kHz)
    %% ============================================================
    COMP --> MONO_LPF["Mono LPF\nButterworth N=15\nfc = 15 kHz"]
    COMP --> PILOT_HET["Pilot Heterodyne\nmix down by 19 kHz\n(phase-continuous carrier)"]
    COMP --> LR_BPF["L−R BPF\nButterworth N=15\n23 – 53 kHz"]
    COMP --> NB1["Noise Band 1 BPF\n16 – 17.5 kHz"]
    COMP --> NB2["Noise Band 2 BPF\n20.5 – 22 kHz"]

    %% --- Mono path ---
    MONO_LPF --> DELAY["Mono Delay\n(18 samples)"]

    %% --- Pilot path (analytic, no Hilbert) ---
    PILOT_HET --> PILOT_LP["Pilot Complex LPF\nButterworth N=9, SOS + carried state\nfc = 1 kHz (half old BPF width)"]
    PILOT_LP --> PHASE_EST["Pilot Phase\n(residual PLL + mix phase)\nθ, and residual for SNR"]
    PHASE_EST --> SC_GEN["Subcarrier Generation\nI = cos(2θ + φ_offset)\nQ = sin(2θ + φ_offset)\nφ_offset = 316° (disc) / 285° (PLL)"]

    %% --- Pilot SNR ---
    PILOT_LP --> SNR_CALC["Pilot SNR\n2·mean(|residual|²) vs noise bands"]
    NB1 --> SNR_CALC
    NB2 --> SNR_CALC
    SNR_CALC --> BLEND["Adaptive Blend Factor\n(EMA smoothed)\nSNR 7–16.5 dB → 0.0–1.0"]
    SNR_CALC --> HFBLEND["HF Blend Ceiling ramp\nSNR 15–35 dB → MAX_GAIN…1.0\n(neutral by default: MAX_GAIN=1.0)"]

    %% --- L−R synchronous demodulation ---
    LR_BPF --> MUL_I["× I_sub × 2.0\n(DSB-SC demod)"]
    LR_BPF --> MUL_Q["× Q_sub × 2.0"]
    SC_GEN --> MUL_I
    SC_GEN --> MUL_Q

    %% --- 3-band LPF (I/Q parallel) ---
    MUL_I --> LPF15I["LPF 15 kHz (I)"]
    MUL_Q --> LPF15Q["LPF 15 kHz (Q)"]
    MUL_I --> LPF12I["LPF 12 kHz (I)"]
    MUL_Q --> LPF12Q["LPF 12 kHz (Q)"]
    MUL_I --> LPF7I["LPF 7 kHz (I)"]
    MUL_Q --> LPF7Q["LPF 7 kHz (Q)"]

    %% --- IQ Phase Correction ---
    LPF15I --> IQ_CORR["IQ Phase Correction\n4-quadrant tracker\nφ_pa = ½ atan2(2·Cov, Var_I − Var_Q)\nanisotropy-gated, branch by continuity\nout = I·cos(φ) + Q·sin(φ)"]
    LPF15Q --> IQ_CORR
    LPF12I --> IQ_CORR
    LPF12Q --> IQ_CORR
    LPF7I --> IQ_CORR
    LPF7Q --> IQ_CORR

    IQ_CORR --> SPLIT["3-Band Split\nLow: 0 – 7 kHz\nMid-High: 7 – 12 kHz\nSuper-High: 12 – 15 kHz"]

    %% --- Band shaping & blend ---
    SPLIT --> SHAPE["Band Shaping\nlow + G_mid·mid + G_super·super\n× blend_factor"]
    BLEND --> SHAPE
    HFBLEND --> SHAPE

    %% --- Side ratio cap (disabled by default) ---
    SHAPE --> SIDE_CAP["Side Ratio Cap\n(disabled by default)"]

    %% ============================================================
    %% Stage 3: Matrix & Audio Output
    %% ============================================================
    DELAY --> MATRIX["Stereo Matrix\nL = Mono + Side\nR = Mono − Side"]
    SIDE_CAP --> MATRIX

    MATRIX --> NOTCH_L["Pilot Notch ×2\n(19 kHz, Q=30)\nLeft ch"]
    MATRIX --> NOTCH_R["Pilot Notch ×2\n(19 kHz, Q=30)\nRight ch"]

    NOTCH_L --> RS2_L["StatefulResampler 1:4\n192 kHz → 48 kHz (Left)\n(grid-aligned)"]
    NOTCH_R --> RS2_R["StatefulResampler 1:4\n192 kHz → 48 kHz (Right)\n(grid-aligned)"]

    RS2_L --> DEEMP_L["De-emphasis IIR\nτ = 50 μs"]
    RS2_R --> DEEMP_R["De-emphasis IIR\nτ = 50 μs"]

    %% --- Side-channel noise reduction (mid/side) ---
    DEEMP_L --> SIDENR["Side NR (DD-Wiener STFT)\nmid=(L+R)/2 untouched\nside=(L−R)/2 denoised\nframe 1024 / hop 256 / 1.5–15 kHz"]
    DEEMP_R --> SIDENR

    SIDENR --> OUT_L["Left Audio\n48 kHz"]
    SIDENR --> OUT_R["Right Audio\n48 kHz"]

    %% ============================================================
    %% Styling
    %% ============================================================
    classDef iq fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef comp fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef pilot fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef lr fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef audio fill:#fce4ec,stroke:#c62828,stroke-width:2px

    class IQ,DC,IQLPF,DEMOD,RS1 iq
    class COMP comp
    class PILOT_HET,PILOT_LP,PHASE_EST,SC_GEN,NB1,NB2,SNR_CALC,BLEND,HFBLEND pilot
    class LR_BPF,MUL_I,MUL_Q,LPF15I,LPF15Q,LPF12I,LPF12Q,LPF7I,LPF7Q,IQ_CORR,SPLIT,SHAPE,SIDE_CAP lr
    class MONO_LPF,DELAY,MATRIX,NOTCH_L,NOTCH_R,RS2_L,RS2_R,DEEMP_L,DEEMP_R,SIDENR,OUT_L,OUT_R audio
```

## FMDemodulatorLight (Arctan Discriminator)

```mermaid
flowchart TD
    IQ["IQ Samples\n(250 kHz complex)"]
    IQ --> DC["DC Offset Removal\n(EMA α=0.01)"]
    DC --> DISC["FM Demodulation\narctan discriminator\nangle(x[n]·conj(x[n-1]))\n(same as Standard)"]
    DISC --> RS1["StatefulResampler\n96 : 125 (grid-aligned)\n250 kHz → 192 kHz"]
    RS1 --> SCALE["× 0.35\n(LIGHT_COMPOSITE_SCALE)"]
    SCALE --> COMP["Composite Signal\n192 kHz"]

    COMP --> STEREO["Stereo Demodulation\n(same as Standard, order-1 filters)"]

    classDef iq fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef comp fill:#fff3e0,stroke:#e65100,stroke-width:2px

    class IQ,DC,DISC,RS1,SCALE iq
    class COMP,STEREO comp
```

## Signal Rate Summary

| Stage | Rate | Ratio |
|-------|------|-------|
| SDR IQ input (Standard) | 1,024,000 Hz | — |
| SDR IQ input (Light) | 250,000 Hz | — |
| Composite signal | 192,000 Hz | ↓3:16 (Standard) / ↓96:125 (Light) |
| Audio output | 48,000 Hz | ↓1:4 |

All three resampling ratios (3:16, 96:125, 1:4) run through
`StatefulResampler`, which grid-aligns the polyphase phase across
arbitrary block sizes so the streamed output is an exact prefix of a
one-shot `resample_poly`.

## Key Processing Blocks

| Block | Class / Function | File |
|-------|-----------------|------|
| Main FM demod (discriminator / PLL) | `FMDemodulator.process_iq_samples()` | `demodulator.py` |
| Pilot PLL (residual) | `PLL` (return_phase=True) | `pll.py` |
| Resampling (all ratios) | `StatefulResampler` (grid-aligned) | `filters.py` |
| Lowpass / Bandpass / Notch | `LowpassFilter`, `BandpassFilter`, `NotchFilter` | `filters.py` |
| De-emphasis | `DeemphasisIIRFilter` (Numba) | `filters.py` |
| Side-channel NR | `SideNoiseReducer` (DD-Wiener STFT) | `filters.py` |
| Stereo Demod Pipeline | `BaseFMDemodulator._demodulate_stereo()` | `demodulator.py` |
