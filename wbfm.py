import sys
import numpy as np
import sounddevice as sd
from radiocore import WBFM, Buffer, Carrousel, Chopper
from SoapySDR import Device, SOAPY_SDR_RX, SOAPY_SDR_CF32


def process(outdata, *_):
    if not carsl.is_healthy:
        return

    # Load, demod, and play buffer.
    with carsl.dequeue() as buffer:
        demoded = demod.run(buffer)
        outdata[:] = np.dstack(demoded)


def receive(carsl, device_buffer):
    while True:
        # Load, fill, and enqueue buffer.
        with carsl.enqueue() as buffer:
            for chunk in chopr.chop(buffer):
                sdr.readStream(rx, [chunk], device_buffer, timeoutUs=int(1e9))

        # Start audio when buffer reaches N samples.
        if carsl.occupancy >= 4 and not stream.active:
            stream.start()


if __name__ == "__main__":
    enable_cuda: bool = False
    frequency: float = 96.9e6
    deemphasis: float = 75e-6
    input_rate: float = 256e3
    output_rate: float = 32e3
    device_buffer: float = 2048
    device_name: str = "airspyhf"

    # SoapySDR configuration.
    sdr = Device({"driver": device_name})
    sdr.setGainMode(SOAPY_SDR_RX, 0, True)
    sdr.setSampleRate(SOAPY_SDR_RX, 0, input_rate)
    sdr.setFrequency(SOAPY_SDR_RX, 0, frequency)
    rx = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)

    # Queue and shared memory allocation.
    carsl = Carrousel([Buffer(input_rate, cuda=enable_cuda) for _ in range(8)])
    demod = WBFM(deemphasis, input_rate, output_rate, cuda=enable_cuda)
    chopr = Chopper(input_rate, device_buffer)

    # Start collecting data.
    sdr.activateStream(rx)
    stream = sd.OutputStream(blocksize=int(output_rate), callback=process,
                             samplerate=int(output_rate), channels=2)

    try:
        receive(carsl, device_buffer)
    except KeyboardInterrupt:
        sdr.deactivateStream(rx)
        sdr.closeStream(rx)
        sys.exit('\nInterrupted by user')
