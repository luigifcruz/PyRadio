# PyRadio

Python scripts for analog radio demodulation (AM/FM) based on the [radio-core] module. Compatible with most SDRs supported by Soapy. Accelerated on the GPU with CUDA by [#cuSignal](https://github.com/rapidsai/cusignal) and on the CPU with [Numba](https://numba.pydata.org/) functions.

## Scripts (CUDA or CPU Accelerated)
- **WBFM**: Demodulate a single Broadcast FM Stations with Stereo Support. Compatible with GPU and CPU
- **MFM**: Demodulate a single Broadcast FM Stations with Mono Support.
- **Multi WBFM**: Demodulate a multiple (25+) Broadcast FM Stations with Stereo Support.
- **Mono WBFM**: Demodulate a multiple (35+) Broadcast FM Stations with Mono Support.

## Validated Radios
- Airspy HF+ Discovery
- LimeSDR Mini/USB
- PlutoSDR
- RTL-SDR

## Roadmap
- [ ] Make scripts more configurable.
- [ ] Add headless audio script.
- [ ] Validate fixed sample-rate SDR.
