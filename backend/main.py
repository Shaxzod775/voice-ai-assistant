import os
import io
import json
import asyncio
import base64
import tempfile
import wave
from typing import Optional
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from faster_whisper import WhisperModel
import httpx

# Global models container (using dict to avoid lifespan global variable issues)
models = {
    "whisper": None,
    "silero": None,
    "silero_sample_rate": 48000
}

# AWS Bedrock config
AWS_BEARER_TOKEN = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")
BEDROCK_ENDPOINT = "https://bedrock-runtime.us-east-1.amazonaws.com"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")

    # Load Whisper
    print("Loading Faster-Whisper model...")
    models["whisper"] = WhisperModel("large-v3", device="cuda", compute_type="float16")
    print("Whisper loaded!")

    # Load Silero TTS
    print("Loading Silero TTS model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    silero, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language='ru',
        speaker='v4_ru'
    )
    # Note: silero.to() modifies in-place and returns None, so don't reassign
    silero.to(device)
    models["silero"] = silero
    models["silero_sample_rate"] = 48000
    print(f"Silero TTS loaded on {device}!")

    yield

    # Cleanup
    models["whisper"] = None
    models["silero"] = None
    torch.cuda.empty_cache()

app = FastAPI(title="Voice AI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Serve the frontend"""
    return FileResponse("/root/voice-ai-app/frontend/index.html")

@app.get("/health")
async def health():
    return {"status": "ok", "cuda": torch.cuda.is_available()}

@app.get("/debug")
async def debug():
    return {
        "whisper_loaded": models["whisper"] is not None,
        "silero_loaded": models["silero"] is not None,
        "silero_type": str(type(models["silero"])),
        "sample_rate": models["silero_sample_rate"]
    }

@app.post("/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    """Convert speech to text using Whisper"""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        segments, info = models["whisper"].transcribe(tmp_path, language="ru")
        text = " ".join([seg.text for seg in segments])
        os.unlink(tmp_path)

        return {"text": text.strip(), "language": info.language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def call_claude(text: str, conversation_history: list = None) -> str:
    """Call Claude Haiku 3.5 via AWS Bedrock using Converse API"""
    if conversation_history is None:
        conversation_history = []

    formatted_history = []
    for msg in conversation_history:
        formatted_history.append({
            "role": msg["role"],
            "content": [{"text": msg["content"]}]
        })

    formatted_history.append({
        "role": "user",
        "content": [{"text": text}]
    })

    payload = {
        "messages": formatted_history,
        "system": [{"text": "You are a helpful AI assistant. Respond concisely in Russian. Keep responses short and conversational, under 100 words."}],
        "inferenceConfig": {
            "maxTokens": 512,
            "temperature": 0.7
        }
    }

    headers = {
        "Authorization": f"Bearer {AWS_BEARER_TOKEN}",
        "Content-Type": "application/json",
    }

    model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0"

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{BEDROCK_ENDPOINT}/model/{model_id}/converse",
            json=payload,
            headers=headers
        )

        if response.status_code != 200:
            print(f"Bedrock API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)

        result = response.json()
        return result["output"]["message"]["content"][0]["text"]

def generate_tts(text: str) -> bytes:
    """Generate speech using Silero TTS with kseniya voice"""
    silero_model = models["silero"]
    sample_rate = 48000  # Original quality

    # Generate audio with Silero
    audio = silero_model.apply_tts(
        text=text,
        speaker='baya',
        sample_rate=sample_rate,
        put_accent=True,
        put_yo=True
    )

    # Convert to numpy
    if isinstance(audio, torch.Tensor):
        audio_np = audio.cpu().numpy()
    else:
        audio_np = np.array(audio)

    # Normalize and convert to int16
    audio_np = audio_np / np.abs(audio_np).max() * 0.9
    audio_np = (audio_np * 32767).astype(np.int16)

    # Create WAV bytes
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_np.tobytes())

    buffer.seek(0)
    return buffer.getvalue()

@app.post("/chat")
async def chat(request: dict):
    """Chat with Claude"""
    text = request.get("text", "")
    history = request.get("history", [])

    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    response = await call_claude(text, history)
    return {"response": response}

@app.post("/tts")
async def text_to_speech(request: dict):
    """Convert text to speech using Silero TTS"""
    text = request.get("text", "")

    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        audio_bytes = generate_tts(text)
        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav")
    except Exception as e:
        print(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """WebSocket for real-time voice conversation"""
    await websocket.accept()

    conversation_history = []

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "audio":
                audio_bytes = base64.b64decode(data["audio"])

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                # STT
                await websocket.send_json({"type": "status", "message": "Transcribing..."})
                segments, info = models["whisper"].transcribe(tmp_path, language="ru")
                user_text = " ".join([seg.text for seg in segments]).strip()
                os.unlink(tmp_path)

                await websocket.send_json({"type": "transcript", "text": user_text})

                if not user_text:
                    continue

                # LLM
                await websocket.send_json({"type": "status", "message": "Thinking..."})
                ai_response = await call_claude(user_text, conversation_history)

                conversation_history.append({"role": "user", "content": user_text})
                conversation_history.append({"role": "assistant", "content": ai_response})

                if len(conversation_history) > 20:
                    conversation_history = conversation_history[-20:]

                await websocket.send_json({"type": "response", "text": ai_response})

                # TTS with Silero
                await websocket.send_json({"type": "status", "message": "Generating speech..."})

                try:
                    audio_bytes = generate_tts(ai_response)
                    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

                    await websocket.send_json({
                        "type": "audio",
                        "audio": audio_base64,
                        "sample_rate": 48000
                    })
                except Exception as e:
                    print(f"TTS error: {e}")
                    await websocket.send_json({"type": "error", "message": f"TTS error: {str(e)}"})

            elif data.get("type") == "clear":
                conversation_history = []
                await websocket.send_json({"type": "status", "message": "History cleared"})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
