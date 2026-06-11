import av
import json
import asyncio
import logging
import argparse
import fractions
import numpy as np
from gsv_tts import TTS
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription

pcs = set()

class AudioTrack(MediaStreamTrack):
    # WebRTC 协商的默认音频参数通常是 48kHz, 双声道, 16位
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.out_sample_rate = 48000
        self.out_samples = 960
        self.pts = 0
        self.queue = asyncio.Queue()
        self.buffer = np.array([], dtype=np.int16)
        self.resampler = None

    async def put_audio(self, clip):
        if len(clip.audio_data) == 0:
            return
        
        audio_data = clip.audio_data.flatten()

        audio_int16 = np.clip(audio_data * 32767.0, -32768, 32767).astype(np.int16)

        if len(audio_int16) % 2 != 0:
            audio_int16 = np.append(audio_int16, np.int16(0))

        in_rate = clip.samplerate
        if self.resampler is None:
            self.resampler = av.AudioResampler(
                format="s16",
                layout="mono",
                rate=self.out_sample_rate
            )

        in_frame = av.AudioFrame.from_ndarray(audio_int16.reshape(1, -1), format='s16', layout='mono')
        in_frame.sample_rate = in_rate
        in_frame.time_base = fractions.Fraction(1, in_rate)

        resampled_frames = self.resampler.resample(in_frame)
        
        for f in resampled_frames:
            samples = f.to_ndarray().flatten()
            self.buffer = np.concatenate((self.buffer, samples))

        while len(self.buffer) >= self.out_samples:
            chunk = self.buffer[:self.out_samples]
            self.buffer = self.buffer[self.out_samples:]

            out_frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format='s16', layout='mono')
            out_frame.sample_rate = self.out_sample_rate
            out_frame.time_base = fractions.Fraction(1, self.out_sample_rate)
            
            await self.queue.put(out_frame)

    async def recv(self):
        frame = await self.queue.get()
        
        frame.pts = self.pts
        self.pts += frame.samples
        
        return frame


async def handle_offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    audio_track = AudioTrack()
    pc.addTrack(audio_track)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logging.info(f"ConnectionState: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)

    @pc.on("datachannel")
    def on_datachannel(channel):
        logging.info(f"Channel: {channel.label}")

        @channel.on("message")
        def on_message(message):
            data = json.loads(message)

            async def generate():
                try:
                    async for clip in tts.infer_stream_async(**data):
                        await audio_track.put_audio(clip)
                    channel.send(json.dumps({"status": "ok"}))
                except Exception as e:
                    logging.error(e)
                    channel.send(json.dumps({"status": "error", "message": str(e)}))

            asyncio.ensure_future(generate())

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


def parse_args():
    parser = argparse.ArgumentParser(description="GSV TTS RealTime")
    
    parser.add_argument(
        "--models_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
    )
    parser.add_argument(
        "--use_bert",
        action="store_true",
    )
    parser.add_argument(
        "--no_auto_bert",
        action="store_false",
        dest="auto_bert",
        default=True,
    )
    parser.add_argument(
        "--use_jieba_fast",
        action="store_true",
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    tts = TTS(
        gpt_cache=[(1, 512), (1, 768), (1, 1024)],
        models_dir=args.models_dir,
        device=args.device,
        dtype=args.dtype,
        use_flash_attn=args.use_flash_attn,
        use_bert=args.use_bert,
        auto_bert=args.auto_bert,
        use_jieba_fast=args.use_jieba_fast,
    )

    app = web.Application()
    app.router.add_post("/offer", handle_offer)
    app.on_shutdown.append(on_shutdown)
    
    web.run_app(app, host="0.0.0.0", port=8080)