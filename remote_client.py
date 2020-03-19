#!/usr/bin/env python3

import queue
import numpy as np
import signal
import sounddevice as sd
import zmq

# Demodulator Settings
sfs = int(256e3)
afs = int(32e3)

sdr_buff = 2048
dsp_buff = sdr_buff * 16
dsp_out = int(dsp_buff/(sfs/afs))

# Queue and Shared Memory Allocation
que = queue.Queue()

#  Create Server
print("Creating ZeroMQ Server...")
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.setsockopt(zmq.SUBSCRIBE, b"96900000")

# Demodulation Function
def process(outdata, f, t, s):
    outdata[:, 0] = np.frombuffer(que.get(), dtype=np.float32)

# Graceful Exit Handler
def signal_handler(signum, frame):
    exit(-1)


signal.signal(signal.SIGINT, signal_handler)

with sd.OutputStream(blocksize=dsp_out, callback=process,
                     samplerate=afs, channels=1):
    while True:
        [address, message] = socket.recv_multipart()
        que.put_nowait(message)
