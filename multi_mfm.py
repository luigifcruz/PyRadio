#!/usr/bin/env python3

from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
import SoapySDR
import signal
import queue
import numpy as np
from radio.analog import MFM
from radio.tools import Tuner
import sounddevice as sd

# Demodulator Settings
cuda = True
tau = 75e-6
sfs = int(240e3)
afs = int(48e3)
sdr_buff = 2400

radios = [
    { "freq": 97.5e6, "bw": sfs },
    { "freq": 96.9e6, "bw": sfs },
]

# Queue and Shared Memory Allocation
que = queue.Queue()
tuner = Tuner(radios, sfs, cuda=cuda)
demod = [ MFM(tau, sfs, afs, cuda=cuda) for _ in radios ]
afile = [ open("FM_{}.if32".format(int(f["freq"])), "bw") for f in radios ]

print("# Tuner Settings:")
print("     Bandwidth: {}".format(tuner.bw))
print("     Mean Frequency: {}".format(tuner.mdf))
print("     Offsets: {}".format(tuner.foff))
print("     Stations: {}".format(len(radios)))

# SoapySDR Configuration
args = dict(driver="lime")
sdr = SoapySDR.Device(args)
sdr.setGainMode(SOAPY_SDR_RX, 0, True)
sdr.setSampleRate(SOAPY_SDR_RX, 0, tuner.bw)
sdr.setFrequency(SOAPY_SDR_RX, 0, tuner.mdf)

# Declare the memory buffer
if cuda:
    import cusignal as sig
    buff = sig.get_shared_mem(tuner.size, dtype=np.complex64)
else:
    buff = np.zeros([tuner.size], dtype=np.complex64)

# Demodulation Function
def process(outdata, f, t, s):
    tuner.load(que.get())

    for i, _ in enumerate(radios):
        L = demod[i].run(tuner.run(i))
        L = L.astype(np.float32)
        afile[i].write(L.tobytes())

    outdata[:, 0] = L


# Graceful Exit Handler
def signal_handler(signum, frame):
    sdr.deactivateStream(rx)
    sdr.closeStream(rx)
    exit(-1)


signal.signal(signal.SIGINT, signal_handler)

# Start Collecting Data
plan = [(i*sdr_buff) for i in range(tuner.size//sdr_buff)]
rx = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rx)

with sd.OutputStream(blocksize=afs, callback=process,
                     samplerate=afs, channels=1):
    while True:
        for i in plan:
            sdr.readStream(rx, [buff[i:]], sdr_buff, timeoutUs=int(1e9))
        que.put_nowait(buff.astype(np.complex64))
