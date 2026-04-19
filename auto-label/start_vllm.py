#!/bin/bash
# vLLM startup script
# Usage: ./start_vllm.sh [native|docker]

set -e

MODE="${1:-native}"
MODEL="${VLLM_MODEL:-huihui-ai/Huihui-Qwen3.5-2B-abliterated}"
PORT="${VLLM_PORT:-8000}"
HOST="${VLLM_HOST:-127.0.0.1}"
GPU_MEM_OVAILABLE="${GPU_MEMORY_OVAILABLE:-0.9}"

echo "=== vLLM Startup ==="
echo "Mode: $MODE"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Host: $HOST"

# =============================================================================
# Option 1: Native vLLM (direct installation)
# =============================================================================
start_native() {
    echo "[Native] Starting vLLM..."

    # Create vLLM virtual environment
    if [ ! -d "venv" ]; then
        python -m venv venv
    fi

    source venv/bin/activate

    pip install --upgrade pip
    pip install vllm>=0.6.3

    # Calculate tensor parallel size (1 for single GPU)
    TP_SIZE=1

    echo "[Native] Launching vLLM on $HOST:$PORT"
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --trust_remote_code \
        --gpu-memory-utilization "$GPU_MEM_OVAILABLE"
}

# =============================================================================
# Option 2: Docker vLLM (recommended for production)
# =============================================================================
start_docker() {
    echo "[Docker] Starting vLLM..."

    # Pull image
    docker pull vllmai/vllm-openai:latest

    # Run container
    docker run --runtime nvidia \
        --gpus '"device=0"' \
        -e NVIDIA_VISIBLE_DEVICES=0 \
        -e MODEL="$MODEL" \
        -p "$PORT":8000 \
        --name vllm \
        -d vllmai/vllm-openai:latest \
        --model "$MODEL" \
        --host "$HOST" \
        --port 8000 \
        --tensor-parallel-size 1 \
        --trust_remote_code

    echo "[Docker] vLLM running at http://$HOST:$PORT"

    # Wait for server ready
    echo "Waiting for server..."
    for i in {1..30}; do
        if curl -s "http://$HOST:$PORT/v1/models" > /dev/null 2>&1; then
            echo "Server ready!"
            exit 0
        fi
        sleep 2
    done
    echo "Server may not be ready yet. Check with: curl http://$HOST:$PORT/v1/models"
}

# =============================================================================
# Option 3: Docker Compose (for easy management)
# =============================================================================
start_compose() {
    echo "[Docker Compose] Starting vLLM..."

    docker compose up -d

    echo "Server starting... Check status with: docker compose ps"
}

# =============================================================================
# Main
# =============================================================================
case "$MODE" in
    native)
        start_native
        ;;
    docker)
        # start_docker  # Uncomment to enable
        echo "[Docker] Commented out by default. Uncomment in script to enable."
        ;;
    compose)
        # start_compose  # Uncomment to enable
        echo "[Docker Compose] Commented out by default. Uncomment in script to enable."
        ;;
    *)
        echo "Usage: $0 [native|docker|compose]"
        echo "  native   - Run vLLM directly (requires GPU with 96GB+ VRAM)"
        echo "  docker  - Run via Docker (commented out by default)"
        echo "  compose - Run via Docker Compose (commented out by default)"
        exit 1
        ;;
esac