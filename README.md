# Voice AI Assistant

Real-time voice conversation with AI using Whisper (STT), Claude Haiku 4.5 (LLM), and Chatterbox (TTS).

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 GPU Server (Vast.ai)                │
│  ┌─────────────────┐    ┌─────────────────────┐    │
│  │  Faster-Whisper │    │   Chatterbox TTS    │    │
│  │   (STT - GPU)   │    │   (TTS - GPU)       │    │
│  └────────┬────────┘    └──────────┬──────────┘    │
│           │                        │               │
│           └──────────┬─────────────┘               │
│                      │                              │
│              ┌───────▼───────┐                      │
│              │    FastAPI    │                      │
│              │   WebSocket   │                      │
│              └───────┬───────┘                      │
└──────────────────────┼──────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  AWS Bedrock   │
              │  Claude Haiku  │
              │      4.5       │
              └────────────────┘
```

## Features

- 🎤 **Speech-to-Text**: Faster-Whisper (large-v3) on GPU
- 🧠 **LLM**: Claude Haiku 4.5 via AWS Bedrock
- 🔊 **Text-to-Speech**: Chatterbox TTS with streaming
- 🌐 **Real-time**: WebSocket communication
- 🇷🇺 **Russian language** support

## Requirements

### GPU Server
- **GPU**: RTX 3090/4090 or better (24GB VRAM recommended)
- **RAM**: 32GB+
- **Storage**: 50GB+
- **CUDA**: 12.0+

### Services
- AWS Bedrock API Key (ABSK token)
- Vast.ai account (or similar GPU cloud)

## Quick Start

### 1. Rent GPU on Vast.ai

1. Go to [vast.ai](https://vast.ai)
2. Select **PyTorch** template
3. Choose GPU with **24GB VRAM** (RTX 3090/4090)
4. Ensure **32GB+ RAM**

### 2. Setup Server

SSH into your server:
```bash
ssh -p <PORT> root@<IP> -L 8080:localhost:8080
```

Clone and setup:
```bash
git clone https://github.com/<YOUR_USERNAME>/voice-ai-assistant.git
cd voice-ai-assistant

# Install dependencies
pip install faster-whisper fastapi uvicorn python-multipart aiofiles websockets httpx
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install Chatterbox
git clone https://github.com/resemble-ai/chatterbox.git /tmp/chatterbox
cd /tmp/chatterbox
sed -i 's/numpy>=1.24.0,<1.26.0/numpy>=1.24.0/' pyproject.toml
pip install -e . --no-build-isolation
cd -
```

### 3. Configure Environment

```bash
export AWS_BEARER_TOKEN_BEDROCK="your_ABSK_token_here"
```

### 4. Run Server

```bash
cd backend
python main.py
```

Server will be available at `http://localhost:8080`

### 5. Access Frontend

Open in browser: `http://localhost:8080`

## API Endpoints

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend HTML |
| `/health` | GET | Health check |
| `/stt` | POST | Speech-to-text (file upload) |
| `/chat` | POST | Chat with Claude |
| `/tts` | POST | Text-to-speech |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `/ws/voice` | Real-time voice conversation |

#### WebSocket Messages

**Client → Server:**
```json
{"type": "audio", "audio": "<base64_wav>"}
{"type": "clear"}
```

**Server → Client:**
```json
{"type": "status", "message": "Processing..."}
{"type": "transcript", "text": "User said..."}
{"type": "response", "text": "AI response..."}
{"type": "audio", "audio": "<base64_wav>", "sample_rate": 24000}
{"type": "error", "message": "Error details"}
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AWS_BEARER_TOKEN_BEDROCK` | AWS Bedrock API key (ABSK format) | Yes |

### Model IDs

- **Whisper**: `large-v3` (can change to `medium` for faster inference)
- **Claude**: `us.anthropic.claude-3-5-haiku-20241022-v1:0`
- **TTS**: Chatterbox default

## Vast.ai Port Mapping

Vast.ai maps internal ports to external ports:

| Internal Port | Usage |
|---------------|-------|
| 8080 | API Server |
| 22 | SSH |

Check your external ports:
```bash
env | grep VAST_TCP_PORT
```

## Troubleshooting

### Port already in use
```bash
fuser -k 8080/tcp
```

### CUDA out of memory
- Use `medium` Whisper model instead of `large-v3`
- Reduce batch size
- Check GPU memory: `nvidia-smi`

### WebSocket connection failed
- Ensure port 8080 is accessible
- Check firewall settings
- Use SSH tunnel for local access

## Project Structure

```
voice-ai-assistant/
├── README.md
├── backend/
│   └── main.py          # FastAPI server
├── frontend/
│   └── index.html       # React frontend
└── start.sh             # Startup script
```

## License

MIT

## Credits

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox)
- [Claude by Anthropic](https://www.anthropic.com)
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
