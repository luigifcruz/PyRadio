# 🤓 PyRadio
### Accelerated Python scripts for analog radio demodulation. [GTX 1070 Ti Demo](https://twitter.com/luigifcruz/status/1218739374717833216?s=20)
Based on the [radio-core](https://github.com/luigifreitas/radio-core) module. Compatible with most SDRs supported by SoapySDR. Accelerated on the GPU with CUDA by [#cuSignal](https://github.com/rapidsai/cusignal) and on the CPU with [Numba](https://numba.pydata.org/) functions.

## Scripts (CUDA or CPU Accelerated)
- **WBFM**: Demodulate a single Broadcast FM Stations with Stereo Support. Compatible with GPU and CPU
- **MFM**: Demodulate a single Broadcast FM Stations with Mono Support.
- **Multi WBFM**: Demodulate a multiple (25+) Broadcast FM Stations with Stereo Support.
- **Mono WBFM**: Demodulate a multiple (35+) Broadcast FM Stations with Mono Support.

## Validated Radios
- AirSpy HF+ Discovery
- LimeSDR Mini/USB
- PlutoSDR
- RTL-SDR

## Roadmap
This is a list of unfinished tasks that I pretend to pursue soon. Pull requests are more than welcome!
- [ ] Make scripts more configurable.
- [ ] Add headless audio script.
- [ ] Validate fixed sample-rate SDR.
