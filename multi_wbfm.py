#!/usr/bin/env python3

from SoapySDR import *
import SoapySDR
import pyaudio
import signal
import queue
import numpy as np
from radio.analog import WBFM
from radio.tools import Tuner

#### Demodulator Settings
cuda = True
tau = 75e-6
sfs = int(256e3)
afs = int(32e3)

radios = [
    { "freq": 97.5e6, "bw": sfs },
    { "freq": 91.9e6, "bw": sfs },
    { "freq": 94.5e6, "bw": sfs },
    { "freq": 89.5e6, "bw": sfs },
    { "freq": 101.5e6, "bw": sfs },
    { "freq": 99.9e6, "bw": sfs },
    { "freq": 93.4e6, "bw": sfs },
    { "freq": 98.4e6, "bw": sfs },
    { "freq": 98.9e6, "bw": sfs },
    { "freq": 97.9e6, "bw": sfs },
    { "freq": 92.4e6, "bw": sfs },
    { "freq": 92.9e6, "bw": sfs },
    { "freq": 93.9e6, "bw": sfs },
    { "freq": 94.1e6, "bw": sfs },
    { "freq": 91.1e6, "bw": sfs },
    { "freq": 92.1e6, "bw": sfs },
    { "freq": 93.1e6, "bw": sfs },
    { "freq": 95.9e6, "bw": sfs },
    { "freq": 102.1e6, "bw": sfs },
    { "freq": 103.9e6, "bw": sfs },
    { "freq": 95.5e6, "bw": sfs },
    { "freq": 87.9e6, "bw": sfs },
    { "freq": 91.5e6, "bw": sfs },
    { "freq": 96.9e6, "bw": sfs },
]

#### Queue and Shared Memory Allocation
que = queue.Queue()
p = pyaudio.PyAudio()

tuner = Tuner(radios, cuda=cuda)
demod = WBFM(tau, sfs, afs, sfs, cuda=cuda)
dsp_out = int(tuner.dfac[0]/(sfs//afs))
sdr_buff = 1024

print("#### Tuner Settings:")
print("     Bandwidth: {}".format(tuner.bw))
print("     Mean Frequency: {}".format(tuner.mdf))
print("     Offsets: {}".format(tuner.foff))
print("     Radios: {}".format(len(radios)))

#### Create Recording Files
afile = [ open("FM_{}.if32".format(int(f["freq"])), "bw") for f in radios ]

#### SoapySDR Configuration
args = dict(driver="lime")
sdr = SoapySDR.Device(args)
sdr.setGainMode(SOAPY_SDR_RX, 0, True)
sdr.setSampleRate(SOAPY_SDR_RX, 0, tuner.bw)
sdr.setFrequency(SOAPY_SDR_RX, 0, tuner.mdf)

#### Declare the memory buffer
if cuda:
    import cusignal as sig
    print("#### CUDA Backend Enabled: (cuSignal + Cupy)")
    buff = sig.get_shared_mem(tuner.size, dtype=np.complex64)
else:
    buff = np.zeros([tuner.size], dtype=np.complex64)

#### Demodulation Function 
def process(in_data, frame_count, time_info, status):
    tuner.load(que.get())

    for i, _ in enumerate(radios):
        L, R = demod.run(tuner.run(i))
        I = np.zeros((dsp_out*2), dtype=np.float32)
        I[0::2] = L; I[1::2] = R
        afile[i].write(I.tostring('C'))
    
    return (I, pyaudio.paContinue)

stream = p.open(
    format=pyaudio.paFloat32,
    channels=2,
    frames_per_buffer=dsp_out,
    rate=afs,
    output=True,
    stream_callback=process)

#### Graceful Exit Handler
def signal_handler(signum, frame):
    stream.stop_stream()
    stream.close()
    p.terminate()
    sdr.closeStream(rx)
    exit(-1)
    
signal.signal(signal.SIGINT, signal_handler)

#### Start Collecting Data
stream.start_stream()
rx = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rx)

while True:
    for i in range(tuner.size//sdr_buff):
        sdr.readStream(rx, [buff[(i*sdr_buff):]], sdr_buff, timeoutUs=int(1e9))
    que.put(buff.astype(np.complex64))
