import os
import io
import json
import asyncio
import base64
import tempfile
from typing import Optional
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from chatterbox.tts import ChatterboxTTS
import httpx

# Global models
whisper_model = None
tts_model = None

# AWS Bedrock config
AWS_BEARER_TOKEN = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")
BEDROCK_ENDPOINT = "https://bedrock-runtime.us-east-1.amazonaws.com"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model, tts_model
    print("Loading models...")

    # Load Whisper
    print("Loading Faster-Whisper model...")
    whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    print("Whisper loaded!")

    # Load Chatterbox TTS
    print("Loading Chatterbox TTS model...")
    tts_model = ChatterboxTTS.from_pretrained(device="cuda")
    print("Chatterbox loaded!")

    yield

    # Cleanup
    del whisper_model, tts_model
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

@app.post("/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    """Convert speech to text using Whisper"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Transcribe
        segments, info = whisper_model.transcribe(tmp_path, language="ru")
        text = " ".join([seg.text for seg in segments])

        # Cleanup
        os.unlink(tmp_path)

        return {"text": text.strip(), "language": info.language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def call_claude(text: str, conversation_history: list = None) -> str:
    """Call Claude Haiku 4.5 via AWS Bedrock using Converse API"""
    if conversation_history is None:
        conversation_history = []

    # Format messages for Bedrock Converse API
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

    # Model ID for Claude Haiku 4.5
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
        # Converse API response format
        return result["output"]["message"]["content"][0]["text"]

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
    """Convert text to speech using Chatterbox"""
    text = request.get("text", "")

    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        # Generate audio
        wav = tts_model.generate(text)

        # Convert to bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav, tts_model.sr, format="wav")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """WebSocket for real-time voice conversation"""
    await websocket.accept()

    conversation_history = []

    try:
        while True:
            # Receive audio data (base64 encoded)
            data = await websocket.receive_json()

            if data.get("type") == "audio":
                # Decode audio
                audio_bytes = base64.b64decode(data["audio"])

                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                # STT
                await websocket.send_json({"type": "status", "message": "Transcribing..."})
                segments, info = whisper_model.transcribe(tmp_path, language="ru")
                user_text = " ".join([seg.text for seg in segments]).strip()
                os.unlink(tmp_path)

                await websocket.send_json({"type": "transcript", "text": user_text})

                if not user_text:
                    continue

                # LLM
                await websocket.send_json({"type": "status", "message": "Thinking..."})
                ai_response = await call_claude(user_text, conversation_history)

                # Update history
                conversation_history.append({"role": "user", "content": user_text})
                conversation_history.append({"role": "assistant", "content": ai_response})

                # Keep history manageable
                if len(conversation_history) > 20:
                    conversation_history = conversation_history[-20:]

                await websocket.send_json({"type": "response", "text": ai_response})

                # TTS
                await websocket.send_json({"type": "status", "message": "Generating speech..."})
                wav = tts_model.generate(ai_response)

                # Convert to base64
                buffer = io.BytesIO()
                torchaudio.save(buffer, wav, tts_model.sr, format="wav")
                audio_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                await websocket.send_json({
                    "type": "audio",
                    "audio": audio_base64,
                    "sample_rate": tts_model.sr
                })

            elif data.get("type") == "clear":
                conversation_history = []
                await websocket.send_json({"type": "status", "message": "History cleared"})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
