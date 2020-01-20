#/usr/bin/env python3

from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
import SoapySDR
import pyaudio
import signal
import queue
import numpy as np
from radio.analog import WBFM

# Demodulator Settings
cuda = False
freq = 96.9e6
tau = 75e-6
sfs = int(256e3)
afs = int(32e3)

sdr_buff = 2048
dsp_buff = sdr_buff * 4
dsp_out = int(dsp_buff/(sfs/afs))

# SoapySDR Configuration
args = dict(driver="lime")
sdr = SoapySDR.Device(args)
sdr.setGainMode(SOAPY_SDR_RX, 0, True)
sdr.setSampleRate(SOAPY_SDR_RX, 0, sfs)
sdr.setFrequency(SOAPY_SDR_RX, 0, freq)

# Queue and Shared Memory Allocation
que = queue.Queue()
p = pyaudio.PyAudio()
audio_file = open("FM_{}.if32".format(int(freq)), "bw")
demod = WBFM(tau, sfs, afs, dsp_buff, cuda=cuda)

# Declare the memory buffer
if cuda:
    import cusignal as sig
    buff = sig.get_shared_mem(dsp_buff, dtype=np.complex64)
else:
    buff = np.zeros([dsp_buff], dtype=np.complex64)


# Demodulation Function
def process(in_data, frame_count, time_info, status):
    L, R = demod.run(que.get())
    LR = np.zeros((dsp_out*2), dtype=np.float32)
    LR[0::2] = L
    LR[1::2] = R
    audio_file.write(LR.tostring('C'))
    LR = LR.reshape(dsp_out, 2)
    return (LR, pyaudio.paContinue)


stream = p.open(
    format=pyaudio.paFloat32,
    channels=2,
    frames_per_buffer=dsp_out,
    rate=afs,
    output=True,
    stream_callback=process)


# Graceful Exit Handler
def signal_handler(signum, frame):
    stream.stop_stream()
    stream.close()
    p.terminate()
    sdr.closeStream(rx)
    exit(-1)


signal.signal(signal.SIGINT, signal_handler)

# Start Collecting Data
stream.start_stream()
rx = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rx)

while True:
    for i in range(dsp_buff//sdr_buff):
        sdr.readStream(rx, [buff[(i*sdr_buff):]], sdr_buff, timeoutUs=int(1e9))
    que.put(buff.astype(np.complex64))
