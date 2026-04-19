# xsnv-dataset

A dataset pipeline for downloading, labeling, and uploading Chinese beauty image albums.

## Workflow

```
┌─────────┐    ┌─────────┐    ┌────────────┐    ┌──────────┐
│  scrap  │───▶│  pack   │───▶│  download  │───▶│ auto/view│
│  xsnv   │    │   HF    │    │    HF      │    │  label   │
└─────────┘    └─────────┘    └────────────┘    └──────────┘
   (1)          (2)             (3)               (4)
```

1. **scrap/** - Download albums from xsnvshen.com
2. **pack/** - Pack & upload to HuggingFace
3. **download/** - Download from HuggingFace
4. **auto-label/** - Auto label with AI
4. **view/** - Manual labeling UI

## Folder Structure

```
xsnv-dataset/
├── README.md
├── pyproject.toml
├── package.json
├── .env.example
│
├── scrap/          # Download from xsnvshen.com
│   ├── download.py
│   └── requirements.txt
│
├── pack/           # Pack & upload to HuggingFace
│   ├── pack.py
│   └── requirements.txt
│
├── download/       # Download from HuggingFace
│   ├── huggingface.py
│   └── requirements.txt
│
├── auto-label/     # AI labeling (placeholder)
│   └── README.md
│
└── view/          # Manual labeling UI
    ├── server.js
    ├── package.json
    ├── core/
    └── public/
```

## Dataset Schema

HuggingFace: [DownFlow/meizi](https://huggingface.co/datasets/DownFlow/meizi)

| Column | Type | Description |
|--------|------|-------------|
| image | image | Image file |
| file_name | string | Filename |
| title | string | Album title |
| model_name | string | Model name |
| tags | list | Original tags |
| album_id | int64 | Album ID |
| text_en | string | English caption |
| text_cn | string | Chinese caption |
| album_score | int8 | Human score (1-9) |
| album_reason | string | Human reasoning |
| ai_score | int8 | AI score (1-9) |
| ai_reason | string | AI reasoning |
| has_face | bool | Face detected |
| has_fullbody | bool | Full body detected |

## Quick Start

### 1. Install Dependencies

```bash
# Python
cd scrap && pip install -r requirements.txt
cd pack && pip install -r requirements.txt
cd download && pip install -r requirements.txt

# Node.js
cd view && npm install
```

### 2. Download from xsnvshen.com

```bash
cd scrap

# Single album
python download.py --album 45545

# All albums from a model
python download.py --model 28036

# All albums from a tag
python download.py --tag 116        # 浴室

# Options
-w 4    # concurrent workers (default: 4)
-d 1.0  # delay between albums (default: 0.5)
-p 5    # limit pages
-o ../downloads  # output directory
```

### 3. Pack & Upload to HuggingFace

```bash
cd pack
python pack.py -d ../downloads -l ../labels -o ./dataset

# Upload
HF_TOKEN=xxx hf upload DownFlow/meizi ./dataset --repo-type dataset
```

### 4. Download from HuggingFace

```bash
cd download
HF_TOKEN=xxx python huggingface.py DownFlow/meizi ../downloads
```

### 5. Manual Labeling

```bash
cd view
npm run dev
# Open http://localhost:3000
```

## Environment Variables

Create `.env` file in root:

```bash
# Auth token for non-localhost access (view server)
AUTH_TOKEN=your-secret-token

# HuggingFace token
HF_TOKEN=hf_xxxx

# vLLM endpoint (auto-label)
VLLM_URL=http://127.0.0.1:8000
VLLM_MODEL=huihui-ai/Huihui-Qwen3.5-2B-abliterated

# Server port (view)
PORT=3000
```

## Scoring System

| Score | Label |
|-------|-------|
| 1 | 极差 |
| 2 | 差 |
| 3 | 较差 |
| 4 | 一般- |
| 5 | 一般 |
| 6 | 一般+ |
| 7 | 较好 |
| 8 | 好 |
| 9 | 极好 |

## Common Tag IDs (xsnvshen.com)

| Tag | ID |
|-----|-----|
| 性感 | 2 |
| 清纯 | 107 |
| 浴室 | 116 |
| 秀人网 | 147 |
| 美腿 | 166 |
| 黑丝 | 183 |

## License

MIT
