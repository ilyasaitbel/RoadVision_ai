<div align="center">

# 🚗 RoadVision AI

**Professional vehicle detection — on your machine, or on your server.**

[![Streamlit App](https://img.shields.io/badge/streamlit-online-FF4B4B?style=for-the-badge)](#-deployment)
[![GitHub](https://img.shields.io/badge/repo-GitHub-185FA5?style=for-the-badge&logo=github)](https://github.com/ilyasaitbel/RoadVision_ai)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/model-YOLOv8-0F6E56)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)

*RoadVision AI (formerly VeyloraX) detects cars, buses, trucks, motorcycles, bicycles and trains with a real YOLOv8 model — a full-featured Streamlit experience for image, video and webcam detection.*

</div>

---

## ✨ Highlights

| | Feature | |
|---|---|---|
| 🖼️ | **Image detection** — side-by-side view, per-class stats, confidence table, PNG export | |
| 🎬 | **Video detection** — frame-by-frame processing, progress bar, live preview, class-breakdown chart, MP4 export | |
| 📸 | **Webcam snapshots** — instant detection from your camera | |
| 🎛️ | **Tunable pipeline** — confidence threshold, NMS IoU, per-class filters | |
| 📊 | **Instant analytics** — vehicle counts, average confidence, inference time, class distribution chart | |
| 🔄 | **Adaptive model loading** — swap in your fine-tuned 21-class model and the app unlocks it automatically | |

## 🚀 Quick start

### Streamlit app online (no install)

Open the deployed app on **Streamlit Community Cloud** (see [Deployment](#-deployment)) — image, video and webcam detection, nothing to install.

### Streamlit app (local)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
streamlit run app.py            # → http://localhost:8501
```

The app loads the weights from `src/models/`.

## 🧠 How it works

```
            ┌──────────────────────────────────────────────────┐
            │  Kaggle road-vehicle dataset (21 classes)        │
            └───────────────┬──────────────────────────────────┘
                            │ preprocessing · 80/10/10 split
            ┌───────────────┴───────────────┐
            ▼                               ▼
 ┌─────────────────────┐          ┌─────────────────────┐
 │ SimpleYOLO          │          │ YOLOv9c fine-tuning │
 │ (from scratch,      │          │ (transfer learning, │
 │  custom PyTorch)    │          │  src/espace)        │
 └─────────┬───────────┘          └──────────┬──────────┘
           └────────── vehicle_detection.pt ─┘
                           │
                           ▼
              ┌────────────────────────────┐
              │ app.py (Streamlit)         │
              │  · image · video · webcam  │
              │  · stats · filters · export│
              └────────────┬───────────────┘
                           ▼
              ┌────────────────────────────┐
              │ annotated PNG / MP4        │
              │ + live analytics           │
              └────────────────────────────┘
```

### The model

`src/models/vehicle_detection.pt` ships with **COCO-pretrained YOLOv8s** weights, filtered to six street-vehicle classes:

`bicycle · car · motorcycle · bus · train · truck`

### Train your own 21-class model

Target the full road-vehicle dataset (ambulance, taxi, rickshaw, van…) on a CUDA GPU:

```bash
python src/espace/fine_tuning/app.py     # downloads data, fine-tunes YOLOv9c, evaluates mAP
cp veylorax.pt src/models/vehicle_detection.pt
```

The app **auto-detects** fine-tuned checkpoints (by inspecting class names) and unlocks all 21 classes — no code changes needed.

## 📁 Project structure

```
RoadVision_ai/
├── app.py                          # Streamlit app (image · video · webcam)
├── requirements.txt
├── src/
│   ├── models/                     # .pt + .onnx weights
│   ├── espace/
│   │   ├── detection_from_scratch/ # SimpleYOLO network (custom PyTorch)
│   │   └── fine_tuning/app.py      # transfer-learning training script
│   └── pretraitement/              # dataset preprocessing
└── notebook/eda.ipynb              # exploratory data analysis
```

## 🚀 Deployment

### Streamlit Community Cloud (recommended — free)

The app is live on Streamlit Community Cloud, deployed straight from this repo:

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Repository: `ilyasaitbel/RoadVision_ai` · Branch: `main` · **Main file: `app.py`**
5. Click **Deploy** — first build installs `requirements.txt` and loads the YOLO weights (~5 min), then your public URL is ready (`https://<app-name>.streamlit.app`).

> Notes: the 22 MB `.pt` weights load from the repo at runtime; the app uses the free tier (1 GB RAM) which is enough for YOLOv8s inference. Each git push to `main` auto-redeploys the app.

> ⚠️ The **Streamlit** app needs a long-running Python process — Streamlit Community Cloud, Render, Railway or Hugging Face Spaces work; static hosts do not.

## 🔬 Performance

| Runtime | Hardware | Image inference |
|---|---|---|
| Ultralytics PyTorch | CPU | ~0.4 s |
| Ultralytics PyTorch | CUDA GPU | ~20 ms |

## 🤝 Contributing

1. Fork the repository
2. Create your branch (`git checkout -b feature/amazing-idea`)
3. Commit your changes (`git commit -m "feat: add amazing idea"`)
4. Push and open a Pull Request


---

<div align="center">
Built with YOLOv8 · Streamlit<br/>
Made by <a href="https://github.com/ilyasaitbel">Ilyas Ait Benlhoussain</a>
</div>
