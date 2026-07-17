#!/bin/bash
# Serving LLM for stance detection.

#activate the venv
source ~/.virtualenv/ai-lab/bin/activate


nohup python3 bsky_api.py > ~/projects/information-diffusion/logs/bsky_data.log &
# Record the PID to a file
echo $! > bsky_data.pid

# Optionally, you can print a message confirming the PID has been saved
echo "LLM server started with PID $(cat bsky_data.pid)"