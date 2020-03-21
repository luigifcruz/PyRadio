#!/usr/bin/env python3

import queue
import numpy as np
import signal
import sounddevice as sd
import zmq

# Demodulator Settings
freq = 96.9e6
afs = int(32e3)

# Queue and Shared Memory Allocation
que = queue.Queue()

#  Create Server
print("Creating ZeroMQ Server...")
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")

address = int(freq).to_bytes(4, byteorder='little')
socket.setsockopt(zmq.SUBSCRIBE, address)

# Demodulation Function
def process(outdata, f, t, s):
    outdata[:, 0] = np.frombuffer(que.get(), dtype=np.float32)


# Graceful Exit Handler
def signal_handler(signum, frame):
    exit(-1)


signal.signal(signal.SIGINT, signal_handler)

with sd.OutputStream(blocksize=afs, callback=process,
                     samplerate=afs, channels=1):
    while True:
        [address, message] = socket.recv_multipart()
        que.put_nowait(message)
