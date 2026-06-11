import json
import asyncio
import time
import sys
import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription

# ================= 配置区域 =================
SERVER_URL = "http://127.0.0.1:8080/offer"
SPK_AUDIO = r"C:\baidunetdiskdownload\洛琪希GSV模型\参考音频实例\害羞\えっと、ルーデオスさん、その、ありがとうございました.wav"
PROMPT_AUDIO = r"C:\baidunetdiskdownload\洛琪希GSV模型\参考音频实例\害羞\えっと、ルーデオスさん、その、ありがとうございました.wav"
PROMPT_TEXT = "えっと、ルーデオスさん、その、ありがとうございました。"
GPT_MODEL = None
SOVITS_MODEL = r"C:\baidunetdiskdownload\洛琪希GSV模型\洛琪希.pth"
# ============================================

# 尝试初始化音频播放器
audio_stream = None
try:
    import pyaudio
    p = pyaudio.PyAudio()
    # WebRTC 协商的默认音频参数通常是 48kHz, 双声道, 16位
    audio_stream = p.open(format=pyaudio.paInt16, channels=2, rate=48000, output=True)
except ImportError:
    print("未安装 pyaudio，将只测算延迟，不会播放声音。")

async def receive_audio(track, state):
    """处理接收到的远端音频帧"""
    try:
        while True:
            frame = await track.recv()
            if state["wait_first"]:
                delay = (time.time() - state["t_sent"]) * 1000
                print(f"\n首包音频延迟 (TTFA): {delay:.2f} ms")
                state["wait_first"] = False
            
            if audio_stream:
                pcm_data = frame.to_ndarray().tobytes()
                await asyncio.to_thread(audio_stream.write, pcm_data)
    except Exception:
        pass

async def run_client():
    state = {"t_sent": 0, "wait_first": False, "connected": False}
    pc = RTCPeerConnection()
    pc.addTransceiver("audio", direction="recvonly")
    channel = pc.createDataChannel("chat")

    @pc.on("track")
    def on_track(track):
        asyncio.ensure_future(receive_audio(track, state))

    @pc.on("connectionstatechange")
    async def on_state_change():
        if pc.connectionState == "connected":
            state["connected"] = True
            print("WebRTC 链路已连接成功！")

    @channel.on("message")
    def on_message(message):
        data = json.loads(message)
        if data.get("status") == "error":
            print(f"\n服务端报错: {data.get('message')}")

    # 创建 Offer 并交换信令
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(SERVER_URL, json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}) as response:
                ans_data = await response.json()
    except Exception as e:
        print(f"无法连接到服务端: {e}")
        await pc.close()
        return

    await pc.setRemoteDescription(RTCSessionDescription(sdp=ans_data["sdp"], type=ans_data["type"]))

    # 等待 P2P 连接建立
    while not state["connected"]:
        await asyncio.sleep(0.1)

    print("\n交互已就绪。输入文本后按回车发送，输入 'quit' 退出。")
    while True:
        text = (await asyncio.to_thread(input, "\n>> ")).strip()
        if not text:
            continue
        if text.lower() in ["quit", "exit"]:
            break

        state["t_sent"] = time.time()
        state["wait_first"] = True

        # 发送请求
        channel.send(json.dumps({
            "spk_audio_path": SPK_AUDIO,
            "prompt_audio_path": PROMPT_AUDIO,
            "prompt_audio_text": PROMPT_TEXT,
            "text": text,
            "gpt_model": GPT_MODEL,
            "sovits_model": SOVITS_MODEL,
        }))

    # 释放资源
    if audio_stream:
        audio_stream.stop_stream()
        audio_stream.close()
        p.terminate()
    await pc.close()

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        print("\n程序已手动终止。")