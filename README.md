---
title: Plant Doctor
emoji: 🌿
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: cc-by-sa-3.0
short_description: AI-powered plant health screening with Grad-CAM explanations
---

# Plant Doctor

**Understand your plants. Protect your harvest.**

Plant Doctor is an AI-powered plant health screening web application. Upload a photograph of a leaf, optionally crop to the suspicious area, and a fine-tuned EfficientNet-B0 identifies the most likely of 38 healthy/diseased conditions across 14 crops, shows its confidence and alternatives, explains the decision with a live Grad-CAM heatmap, and presents sourced, non-prescriptive guidance about the condition. It is an **educational screening aid, not a diagnostic device**.

> Model metrics, dataset licence notes and known limitations are in **[artifacts/evaluation_report.md](artifacts/evaluation_report.md)** and in the app's *About & model* tab. The short user manual is **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)**.

## Features

* **Real trained model** – EfficientNet-B0 (ImageNet weights) fully fine-tuned on PlantVillage; the checkpoint is loaded strictly and the app refuses to run analyses if it is missing (no silent fallback).
* **Cropping that matters** – click two corners (or use sliders); the crop is literally the tensor the network sees. Full-image vs. region comparison explains disagreements.
* **Honest confidence** – softmax percentages, top-3 alternatives, and a configurable abstention stage (blur, feature-space novelty, confidence threshold) that answers *"We couldn't confidently identify a plant condition from this image…"* instead of guessing.
* **Grad-CAM** computed from the real gradients of the selected class on every analysis; original / heatmap / overlay views and a comparison slider.
* **Disease information database** for all 38 classes (symptoms, favourable conditions, prevention, next steps, references to extension services); no treatment prescriptions.
* **Session history**, **PDF report download**, **educational "How did the AI decide?"** section, **About & model** page with metrics and limitations.
* Premium, responsive interface (desktop, tablet, phone).

## Results (shipped checkpoint)

Trained on a 4-core CPU in 126 minutes (4 epochs, best epoch 4). Validation (leaf-grouped, 4,389 images): accuracy 99.75%, macro-F1 99.54%.

| Held-out test metric | Official PlantVillage test split (10709 images) | Strict subset without leaf-ID overlap (10466 images) |
|---|---|---|
| Accuracy | 99.51% | 99.53% |
| Top-3 accuracy | 99.99% | 99.99% |
| Macro precision / recall | 99.45% / 99.34% | 99.45% / 99.32% |
| Macro F1 | 99.39% | 99.38% |

Weakest classes are the two corn leaf blights (gray leaf spot F1 0.933, northern leaf blight F1 0.964), which are confused with each other. Mean top-1 softmax on test images is 90.05% (label smoothing keeps it below 100 %); 0.67% of test images fall under the 60 % abstention threshold, and accuracy among confident predictions is 99.75%.

**These are laboratory-photo benchmark numbers.** They say nothing about photographs taken in the field; see the limitations section and `artifacts/evaluation_report.md`.

## Project layout

```
plant-doctor/
├── app.py                     Gradio application (entry point for local use and Hugging Face Spaces)
├── config.py                  Paths, preprocessing constants, safety thresholds
├── requirements.txt
├── run_plant_doctor.bat       Windows one-click setup + launch
├── run_plant_doctor.sh        macOS / Linux launch
├── models/
│   ├── model.py               EfficientNet-B0 definition, strict checkpoint load/save
│   ├── inference.py           Cached predictor (softmax, features, quality checks, Grad-CAM)
│   ├── gradcam.py             Grad-CAM implementation and heatmap rendering
│   └── quality.py             Blur / novelty / confidence abstention logic
├── core/
│   ├── image_processing.py    Upload validation, decoding, cropping, sharpness, vegetation heuristics
│   ├── cropper.py             Selection preview drawing and box maths
│   ├── disease_info.py        Access to the disease database
│   ├── history.py             Session history
│   └── report.py              PDF report
├── ui/                        CSS design system and HTML renderers
├── training/
│   ├── preprocessing.py       Leaf-grouped manifests, transforms
│   ├── train.py               Reproducible fine-tuning pipeline
│   └── evaluate.py            Test metrics, confusion matrix, novelty-detector fitting, report
├── data/disease_information.json
├── notebooks/training_colab.ipynb
├── tests/                     pytest suite
├── docs/USER_GUIDE.md
└── artifacts/                 Trained checkpoint, class mapping, configs, metadata, evaluation outputs
```

## Running locally

### Windows (recommended for non-programmers)
1. Install Python 3.11 or newer from <https://www.python.org/downloads/windows/> and tick **Add python.exe to PATH**.
2. Make sure `artifacts/plant_doctor_efficientnet_b0.pt` is present (it ships with the project; otherwise see *Training*).
3. Double-click **`run_plant_doctor.bat`**. The first run creates a private environment and installs dependencies (5–10 minutes). The browser opens at <http://127.0.0.1:7860>.

### macOS / Linux
```bash
./run_plant_doctor.sh
```

### Manual
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Environment variables: `PLANT_DOCTOR_MODEL` (checkpoint path), `PLANT_DOCTOR_MIN_CONFIDENCE` (default 0.60), `PLANT_DOCTOR_BLUR_THRESHOLD` (25), `PLANT_DOCTOR_OOD_LIMIT` (1.0), `GRADIO_SERVER_NAME`, `PORT`.

Hardware: CPU is enough (≈0.2–1 s per analysis on a laptop, ≈1 GB RAM). A CUDA GPU is used automatically if PyTorch detects one (install the matching torch wheel from pytorch.org instead of the CPU wheel in `requirements.txt`).

## Deploying to Hugging Face Spaces

This folder is already a valid Gradio Space (the YAML header at the top of this file is the Space configuration).

**Costs and plan requirements (checked November 2026 on huggingface.co/docs/hub/spaces-overview and huggingface.co/pricing — please re-check, they change):**
* The *CPU Basic* hardware (2 vCPU, 16 GB RAM) has **no hourly charge**, but Hugging Face currently states that *creating* a Gradio or Docker Space **requires a paid plan: PRO (about US$9/month) for personal accounts**, or Team/Enterprise for organisations.
* **Free personal accounts in good standing may still host up to 2 Gradio Spaces on "ZeroGPU" hardware.** Plant Doctor runs fine on ZeroGPU (it simply uses the CPU; no GPU code is required).
* Free/CPU Basic Spaces go to sleep after inactivity (currently 48 h) and wake automatically when visited (first load then takes ~1 min).
* Paid upgrades (CPU Upgrade ≈ US$0.03/h, T4 GPU ≈ US$0.40/h) are not needed for this app.

Nothing in this project pays for anything or stores credentials. Decide which option you want before creating the Space.

**Steps**
1. Create a free account at <https://huggingface.co/join> (and, if you choose the PRO route, subscribe at <https://huggingface.co/pricing>).
2. Go to <https://huggingface.co/new-space>: name it `plant-doctor`, SDK **Gradio**, hardware **ZeroGPU** (free route) or **CPU basic** (PRO route), visibility public or private, then *Create Space*.
3. Upload the project files. Easiest without Git: on the Space page choose *Files → Add file → Upload files* and drag in everything **except** `.venv/`, `artifacts/last_checkpoint.pt`, `artifacts/splits/` and `tests/`. The checkpoint (`artifacts/plant_doctor_efficientnet_b0.pt`, ~16 MB) and `artifacts/ood_statistics.npz` must be included; files over 10 MB are stored with Git LFS automatically when uploaded through the website (`.gitattributes` is provided for command-line pushes).
4. The Space builds automatically (3–5 minutes) and shows the app at `https://huggingface.co/spaces/<your-name>/plant-doctor`.

Command-line alternative (requires a *write* token from <https://huggingface.co/settings/tokens>; never put the token in any file):
```bash
pip install huggingface_hub
hf auth login
hf upload <your-name>/plant-doctor . . --repo-type space --exclude ".venv/*" "artifacts/last_checkpoint.pt" "artifacts/splits/*" "tests/*"
```

## Training (reproducible pipeline)

```bash
# dataset: https://huggingface.co/datasets/mohanty/PlantVillage  (data.zip + splits/ + leaf_grouping/)
python -m training.train --data-root /path/to/plantvillage --epochs 4      # writes artifacts/
python -m training.evaluate                                                # test metrics, OOD stats, report
```
* Official PlantVillage train/test lists are used; the validation set holds out 10 % of the training *leaves* of every class using the dataset's leaf-identity map (no photographs of the same leaf on both sides). The test split is used once, at the end.
* AdamW, label-smoothed cross-entropy, warm-up + cosine schedule, augmentation on training data only, early stopping on validation macro-F1, fixed seeds, resumable checkpoints (`--resume`).
* No GPU? Training the shipped model took ~2.3 hours on a 4-core CPU. Or open **`notebooks/training_colab.ipynb`** in Google Colab, select a GPU runtime and *Run all*.

## Tests

```bash
python -m pytest tests -q
```
The suite covers valid healthy/diseased/cropped/blurry/unrelated/multi-leaf/very large images, unsupported and corrupted files, repeated analyses, checkpoint loading, Grad-CAM gradient behaviour, the abstention rules, the PDF report, history and the application handlers. Tests that need the trained checkpoint are skipped automatically when it is absent.

## Dataset attribution and licence

PlantVillage images: Hughes, D. P. & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060; Mohanty, S. P., Hughes, D. P. & Salathé, M. (2016). *Using deep learning for image-based plant disease detection.* Frontiers in Plant Science 7:1419. Data obtained from <https://huggingface.co/datasets/mohanty/PlantVillage>, whose dataset card states the licence **CC BY-SA 3.0**. The original GitHub repository (<https://github.com/spMohanty/PlantVillage-Dataset>) contains no licence file, and other mirrors list different terms (a Zenodo mirror: CC BY-SA 3.0 US; Meta-Album lists the Mendeley release as CC0). Treat the data and the trained weights as **CC BY-SA 3.0 with attribution required**, and verify the terms yourself before any commercial use.

Disease information was compiled from publicly available plant-pathology extension resources (Cornell, University of Minnesota, UC IPM, Penn State, Ohio State, Michigan State, University of Florida IFAS, USDA APHIS, Purdue, APS, RHS, WSU, Crop Protection Network, Iowa State, University of Kentucky); see `data/disease_information.json` for per-class references.

## Limitations (summary)
Benchmark accuracy on PlantVillage does **not** mean field accuracy; only 14 crops / 38 conditions are known; softmax confidence is not calibrated; Grad-CAM shows where the model looked, not where the disease is; the novelty detector only catches images far from the training data; no treatment advice is given. Full discussion: `artifacts/evaluation_report.md`, section 7.
