import os
import json
import queue
import numpy as np
import soundfile as sf
import threading


class AudioQueue:
    def __init__(self):
        self.q = queue.Queue()
        self.buffer = np.zeros((0, 1), dtype='float32')
        self.playback_finished = threading.Event()

    def put(self, data):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.q.put(data)
        self.playback_finished.clear()

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

        if n < frames and self.q.empty():
            self.playback_finished.set()
    
    def clear(self):
        """
        Clears all audio data currently in the playback queue.
        """
        with self.q.mutex:
            self.q.queue.clear()
        self.buffer = np.zeros((0, 1), dtype='float32')
        self.playback_finished.set()
    
    def wait(self):
        """
        Waits until all audio currently in the queue has finished playing.
        """
        self.playback_finished.wait()


class AudioClip:
    def __init__(self, audio_queue, audio_data, samplerate, audio_len_s, subtitles, orig_text):
        self.audio_queue: AudioQueue = audio_queue
        self.audio_data = audio_data
        self.samplerate = samplerate
        self.audio_len_s = audio_len_s
        self.subtitles = subtitles
        self.orig_text = orig_text
    
    def play(self):
        self.audio_queue.put(self.audio_data)
    
    def save(self, save_path: str, is_save_subtitles: bool = False):
        sf.write(save_path, self.audio_data, self.samplerate)

        if is_save_subtitles:
            subtitles_path, _ = os.path.splitext(save_path)
            subtitles_path = subtitles_path + ".json"
            with open(subtitles_path, 'w', encoding='utf-8') as f:
                json.dump({"orig_text":self.orig_text, "subtitles":self.subtitles}, f, indent=4, ensure_ascii=False)