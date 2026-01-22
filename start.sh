#!/bin/bash

# Voice AI Assistant Startup Script
# Run this on your Vast.ai GPU server

# Set your AWS Bedrock API token
export AWS_BEARER_TOKEN_BEDROCK="YOUR_ABSK_TOKEN_HERE"

# Navigate to backend directory
cd /root/voice-ai-app/backend

# Start the server
python main.py
