#!/usr/bin/env python3

from SoapySDR import *
import SoapySDR
import pyaudio
import signal
import queue
import numpy as np
import collections
import scipy.signal as sig
from pll import PLL

#### Demodulator Settings
fm_freq = 96.9e6 # FM Station Frequency
tau = 75e-6
samp_rate = int(256e3)
audio_fs = int(32e3)
buff_len = 2048*16
samples = int(buff_len/8)

#### SoapySDR Configuration
args = dict(driver="lime")
sdr = SoapySDR.Device(args)
sdr.setSampleRate(SOAPY_SDR_RX, 0, samp_rate)
sdr.setFrequency(SOAPY_SDR_RX, 0, fm_freq)

#### Queue and Shared Memory Allocation
que = queue.Queue()
p = pyaudio.PyAudio()
zi = queue.Queue()
buff = np.zeros([buff_len], dtype=np.complex64)
pll = PLL(samp_rate, buff_len)
audio_file = open("FM_{}.if32".format(int(fm_freq)), "bw")

#### Filter Design
x = np.exp(-1/(audio_fs * tau))
db = [1-x]; da = [1,-x]
pb, pa = sig.butter(4, [19e3-250, 19e3+250], btype='bandpass', fs=samp_rate)
mb, ma = sig.butter(8, 15e3, btype='lowpass', fs=samp_rate)
hb, ha = sig.butter(2, 40, btype='highpass', fs=samp_rate)

z = {
    "mlpr": sig.lfilter_zi(mb, ma), "mlmr": sig.lfilter_zi(mb, ma),
    "dlpr": sig.lfilter_zi(db, da), "dlmr": sig.lfilter_zi(db, da),
    "hlmr": sig.lfilter_zi(hb, ha), "hlpr": sig.lfilter_zi(hb, ha),
    "dc": collections.deque(maxlen=32), "diff": 0.0,
}

#### Demodulation Function 
def demod(in_data, frame_count, time_info, status):
    b = np.array(que.get())
    d = np.concatenate(([z['diff']], np.angle(b)), axis=None)
    b = np.diff(np.unwrap(d))
    z['diff'] = d[-1]
    b /= np.pi

    # Normalize for DC
    dc = np.mean(b)
    z['dc'].append(dc)
    b -= np.mean(z['dc'])

    # Synchronize PLL with Pilot
    P = sig.filtfilt(pb, pa, b)
    pll.step(P)

    # Demod Left + Right (LPR)
    LPR, z['mlpr'] = sig.lfilter(mb, ma, b, zi=z['mlpr'])
    LPR, z['hlpr'] = sig.lfilter(hb, ha, LPR, zi=z['hlpr'])
    LPR = sig.resample(LPR, samples, window='hamming')
    LPR, z['dlpr'] = sig.lfilter(db, da, LPR, zi=z['dlpr'])

    # Demod Left - Right (LMR)
    LMR = pll.mult(2) * b
    LMR, z['mlmr'] = sig.lfilter(mb, ma, LMR, zi=z['mlmr'])
    LMR, z['hlmr'] = sig.lfilter(hb, ha, LMR, zi=z['hlmr'])
    LMR = sig.resample(LMR, samples, window='hamming')
    LMR, z['dlmr'] = sig.lfilter(db, da, LMR, zi=z['dlmr'])
    
    # Mix L+R and L-R to generate L and R
    L = LPR+LMR; R = LPR-LMR
    
    # Generate File Output (IF32)
    I = np.zeros((samples*2), dtype=np.float32)
    I[0::2] = L; I[1::2] = R
    I = np.clip(I, -1.0, 1.0)
    audio_file.write(I.tostring('C'))

    return (I.reshape(samples, 2), pyaudio.paContinue)

stream = p.open(format=pyaudio.paFloat32, channels=2, frames_per_buffer=samples, rate=audio_fs, output=True, stream_callback=demod)

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
    sdr.readStream(rx, [buff], buff_len, timeoutUs=int(1e9))
    que.put(buff.astype(np.complex64))