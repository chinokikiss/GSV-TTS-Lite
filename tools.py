import queue
import time
import numpy as np
import threading

class AudioStreamer:
    def __init__(self):
        self.q = queue.Queue()
        self.buffer = np.zeros((0, 1), dtype='float32')

    def put(self, data):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.q.put(data)

    def callback(self, outdata, frames, time, status):
        while len(self.buffer) < frames:
            try:
                self.buffer = np.concatenate((self.buffer, self.q.get_nowait()))
            except queue.Empty:
                break
        n = min(len(self.buffer), frames)
        outdata[:n] = self.buffer[:n]
        outdata[n:] = 0
        self.buffer = self.buffer[n:]

class SubtitlesQueue:
    def __init__(self):
        self.q = queue.Queue()
        self.t = None
    
    def process(self):
        first = True
        last_i = 0

        while True:
            subtitles, text = self.q.get()
            
            if subtitles is None:
                break
            
            if first:
                t = time.time()
                first = False

            for subtitle in subtitles:
                if subtitle["end_s"]:
                    if subtitle["orig_idx_end"] > last_i:
                        print(text[last_i:subtitle["orig_idx_end"]], end="", flush=True)
                        last_i = subtitle["orig_idx_end"]
                    while time.time() - t < subtitle["end_s"]:
                        time.sleep(0.01)

        self.t = None
    
    def add(self, subtitles, text):
        self.q.put((subtitles, text))
        if self.t is None:
            self.t = threading.Thread(target=self.process, daemon=True)
            self.t.start()