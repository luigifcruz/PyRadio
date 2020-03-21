#!/usr/bin/env python3

from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
import SoapySDR
import signal
import queue
import numpy as np
from radio.analog import MFM
from radio.tools import Tuner
import cusignal as sig
import zmq

# Demodulator Settings
tau = 75e-6
sfs = int(256e3)
afs = int(32e3)

radios = [
    {"freq": 97.5e6, "bw": sfs },
    { "freq": 95.5e6, "bw": sfs },
    { "freq": 94.5e6, "bw": sfs },
    { "freq": 96.9e6, "bw": sfs },
]

# Queue and Shared Memory Allocation
que = queue.Queue()

tuner = Tuner(radios, cuda=True)
demod = MFM(tau, sfs, afs, cuda=True)
dsp_out = int(tuner.dfac[0]/(sfs//afs))
sdr_buff = 1024

context = zmq.Context()
context.setsockopt(zmq.IPV6, True)
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")

print("# Tuner Settings:")
print("     Bandwidth: {}".format(tuner.bw))
print("     Mean Frequency: {}".format(tuner.mdf))
print("     Offsets: {}".format(tuner.foff))
print("     Radios: {}".format(len(radios)))

# SoapySDR Configuration
args = dict(driver="lime")
sdr = SoapySDR.Device(args)
sdr.setGainMode(SOAPY_SDR_RX, 0, True)
sdr.setSampleRate(SOAPY_SDR_RX, 0, tuner.bw)
sdr.setFrequency(SOAPY_SDR_RX, 0, tuner.mdf)

# Declare the memory buffer
buff = sig.get_shared_mem(tuner.size, dtype=np.complex64)


# Graceful Exit Handler
def signal_handler(signum, frame):
    sdr.closeStream(rx)
    exit(-1)


signal.signal(signal.SIGINT, signal_handler)

# Start Collecting Data
rx = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rx)

while True:
    for i in range(tuner.size//sdr_buff):
        sdr.readStream(rx, [buff[(i*sdr_buff):]], sdr_buff, timeoutUs=int(1e9))

    tuner.load(buff.copy())
    for i, f in enumerate(radios):
        L = demod.run(tuner.run(i))
        L = L.astype(np.float32)
        address = int(f['freq']).to_bytes(4, byteorder='little')
        socket.send_multipart([address, L.tobytes()])
