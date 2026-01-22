#!/bin/bash

# Start Fish Speech API server in background
echo "Starting Fish Speech API server on port 8001..."
cd /root/fish-speech
python3 tools/api_server.py \
    --listen 127.0.0.1:8001 \
    --llama-checkpoint-path checkpoints/fish-speech-1.5 \
    --decoder-checkpoint-path checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth \
    --decoder-config-name firefly_gan_vq \
    --device cuda \
    --half \
    > /root/fish_speech.log 2>&1 &

FISH_PID=$!
echo "Fish Speech API started with PID $FISH_PID"

# Wait for Fish Speech to load models
echo "Waiting for Fish Speech to initialize (30 seconds)..."
sleep 30

# Check if Fish Speech is running
if ! kill -0 $FISH_PID 2>/dev/null; then
    echo "Fish Speech failed to start! Check /root/fish_speech.log"
    cat /root/fish_speech.log
    exit 1
fi

# Start main backend
echo "Starting main Voice AI backend on port 8888..."
cd /root/voice-ai-app/backend
python3 main.py > /root/backend.log 2>&1 &

BACKEND_PID=$!
echo "Backend started with PID $BACKEND_PID"

echo ""
echo "Both servers are starting..."
echo "Fish Speech API: http://127.0.0.1:8001"
echo "Voice AI Backend: http://0.0.0.0:8888"
echo ""
echo "Logs:"
echo "  Fish Speech: /root/fish_speech.log"
echo "  Backend: /root/backend.log"
echo ""
echo "To check status: curl http://localhost:8888/health"
echo "To stop: kill $FISH_PID $BACKEND_PID"

# Keep script running and show logs
tail -f /root/fish_speech.log /root/backend.log
