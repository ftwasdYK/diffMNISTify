#!/bin/bash

# Check if argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <absolute_path_to_directory>"
    exit 1
fi

DIR_PATH="$1"

docker build -t fastapi-app .
docker run -d -p 8000:8000 -v "$DIR_PATH":/app/checkpoints fastapi-app