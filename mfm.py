#!/usr/bin/env python3

from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
import SoapySDR
import signal
import queue
import numpy as np
from radio.analog import MFM
import sounddevice as sd

# Demodulator Settings
cuda = True
freq = 96.9e6
tau = 75e-6
sfs = int(256e3)
afs = int(32e3)
sdr_buff = 2048

# SoapySDR Configuration
args = dict(driver="lime")
sdr = SoapySDR.Device(args)
sdr.setGainMode(SOAPY_SDR_RX, 0, True)
sdr.setSampleRate(SOAPY_SDR_RX, 0, sfs)
sdr.setFrequency(SOAPY_SDR_RX, 0, freq)

# Queue and Shared Memory Allocation
que = queue.Queue()
demod = MFM(tau, sfs, afs, cuda=cuda)
audio_file = open("FM_{}.if32".format(int(freq)), "bw")

# Declare the memory buffer
if cuda:
    import cusignal as sig
    buff = sig.get_shared_mem(sfs, dtype=np.complex64)
else:
    buff = np.zeros([sfs], dtype=np.complex64)


# Demodulation Function
def process(outdata, f, t, s):
    outdata[:, 0] = demod.run(que.get())
    LPR = outdata[:, 0].astype(np.float32)
    audio_file.write(LPR.tobytes())


# Graceful Exit Handler
def signal_handler(signum, frame):
    sdr.deactivateStream(rx)
    sdr.closeStream(rx)
    exit(-1)


signal.signal(signal.SIGINT, signal_handler)

# Start Collecting Data
plan = [(i*sdr_buff) for i in range(sfs//sdr_buff)]
rx = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rx)

with sd.OutputStream(blocksize=afs, callback=process,
                     samplerate=afs, channels=1):
    while True:
        for i in plan:
            sdr.readStream(rx, [buff[i:]], sdr_buff, timeoutUs=int(1e9))
        que.put_nowait(buff.astype(np.complex64))
