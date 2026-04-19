# Auto-Label

AI labeling module for automated image quality scoring and tagging.

## Two Environments

### 1. Jupyter Notebook (AMD OnClick AI)

Use this to determine prompts for AI-generated fields:

```bash
cd auto-label
jupyter lab notebook.ipynb
```

### 2. Production (NVIDIA GH200)

```bash
# Install dependencies
cd auto-label
pip install -r requirements.txt

# Start vLLM (choose one)
./start_vllm.sh native    # Direct installation
# ./start_vllm.sh docker   # Docker (commented by default)

# Run labeling
python label.py --album 45545
python label.py --all
python label.py --unlabeled
```

## vLLM Setup

### Option 1: Native (Direct Installation)

```bash
./start_vllm.sh native
```

Requirements:
- NVIDIA GPU with 96GB+ VRAM (GH200)
- Python 3.10+

### Option 2: Docker (Recommended)

```bash
# Uncomment in start_vllm.sh to enable
./start_vllm.sh docker
```

Or use Docker Compose:

```bash
docker compose up -d
docker compose logs -f
docker compose down
```

## Environment Variables

Create `.env` file:

```bash
# vLLM
VLLM_URL=http://127.0.0.1:8000
VLLM_MODEL=huihui-ai/Huihui-Qwen3.5-2B-abliterated
VLLM_PORT=8000
VLLM_HOST=127.0.0.1

# Paths
DOWNLOADS_DIR=../downloads
LABELS_DIR=../labels
```

## AI-Generated Fields

Based on the dataset schema:

| Field | Description | Note |
|-------|-------------|------|
| ai_score | Quality score (1-9) | Based on score prompts |
| ai_reason | Reasoning for score | Generated from vLLM |
| ai_tags | Generated tags | Comprehensive tag cloud |
| has_face | Face detected | Vision detection |
| has_fullbody | Full body detected | Pose detection |

## Files

| File | Description |
|------|-------------|
| `notebook.ipynb` | Prompt engineering notebook |
| `label.py` | Production labeling script |
| `start_vllm.sh` | vLLM startup script |
| `docker-compose.yml` | Docker Compose config |
| `requirements.txt` | Python dependencies |

## Scoring Definitions

| Score | Label | Description |
|-------|-------|------------|
| 1 | 极差 | Terrible - major issues |
| 2 | 差 | Poor - significant problems |
| 3 | 较差 | Below average |
| 4 | 一般- | Slightly below average |
| 5 | 一般 | Average |
| 6 | 一般+ | Slightly above average |
| 7 | 较好 | Good |
| 8 | 好 | Very good |
| 9 | 极好 | Excellent |

## Quick Start

```bash
# 1. Start vLLM
./start_vllm.sh native

# 2. Test connection
curl http://127.0.0.1:8000/v1/models

# 3. Run labeling
python label.py --unlabeled
```