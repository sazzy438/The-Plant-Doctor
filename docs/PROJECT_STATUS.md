# Plant Doctor — Project status and delivery notes

Generated 2026-11-16 at the end of the build session.

## 1. What was delivered

| Deliverable | Status |
|---|---|
| Complete application source code (Gradio 6, modular: `models/`, `core/`, `ui/`, `training/`) | **Implemented and tested** |
| Genuine trained model (EfficientNet-B0, fully fine-tuned on PlantVillage, 38 classes) | **Implemented, trained in this session (126 min on a 4-core CPU), tested** |
| Reproducible training pipeline (`training/train.py`, leaf-grouped validation, early stopping, resumable) | **Implemented and used to produce the shipped checkpoint** |
| Google Colab training notebook (`notebooks/training_colab.ipynb`, Run-all) | **Prepared** (notebook structure written and JSON-validated; not executed in this session because no Colab access is available here) |
| Model evaluation report (`artifacts/evaluation_report.md`, metrics JSON, confusion matrix, per-class F1, curves) | **Implemented, run on the official test split** |
| Grad-CAM implementation (real gradients, last conv block, three views + slider) | **Implemented and tested** (gradient test: zeroing a class's weights empties its map) |
| Abstention stage (blur, feature-space novelty detector, confidence threshold, vegetation floor) | **Implemented, calibrated on validation data, tested on synthetic non-leaf images** |
| Disease information database (38 classes, references, no treatment prescriptions) | **Implemented** (compiled from extension knowledge; URLs point to organisation portals, see `data/disease_information.json` `_meta`) |
| Premium responsive interface (Home, Analyse, History, About & model, educational section) | **Implemented, screenshot-verified on desktop (1366 px) and phone (390 px) widths** |
| Cropping that changes the model input (two-click + sliders, full vs selection comparison) | **Implemented and tested** (unit + browser) |
| Session history with reopen | **Implemented and tested** |
| Downloadable PDF report | **Implemented and tested** |
| Automated tests (`tests/`, 38 tests) | **Implemented; all pass against the trained checkpoint** |
| Windows one-click launcher (`run_plant_doctor.bat`), macOS/Linux launcher | **Prepared** (the .bat was written for Windows and could not be executed in this Linux session; the .sh logic was exercised manually) |
| Hugging Face Spaces files (README front matter, requirements, .gitattributes, app entry point) | **Prepared** (no Space was created because that needs your account; costs are explained in README) |
| README and user guide | **Delivered** |

## 2. Measured results (not estimates)

* Validation (4,389 leaf-grouped images): accuracy 99.75%, macro-F1 99.54%.
* Official PlantVillage test split (10709 images): accuracy 99.51%, macro-F1 99.39%, top-3 99.99%.
* Strict test subset without leaf-ID overlap (10466 images): accuracy 99.53%, macro-F1 99.38%.
* Weakest class: corn gray leaf spot (F1 0.933), confused with northern leaf blight.
* Abstention on synthetic non-leaf images (noise, flat grey, sky gradient, text document, face-like drawing, green disc): all six rejected after the vegetation floor was added.

**These are laboratory-photo benchmark numbers. Field performance was not measured and is expected to be substantially lower.**

## 3. What was actually tested

Automated (pytest, all passing with the shipped checkpoint):
1. valid healthy leaf → recognised as Tomato / healthy; 2. valid diseased leaf → Tomato / early blight; 3. cropped leaf → crop pixels equal the model input, prediction changes; 4. blurry photo → abstains ("too blurry"); 5. unrelated photo → abstains; 6. unsupported format (GIF, BMP renamed to .jpg) → friendly error; 7. corrupted files (truncated JPEG, garbage bytes) → friendly error; 8. very large image (4500×3000, 2048×2048 Grad-CAM) → handled, downscaled; 9. multi-leaf collage → runs, Grad-CAM aligned; 10. repeated analysis → identical probabilities. Also: predictions change when network weights are perturbed (no lookup/metadata path), checkpoint round-trip and strict load errors, class mapping ⇄ checkpoint consistency, leaf-grouped manifests have no train/val leakage, disease info exists for every class and matches the predicted class, PDF report builds, history limit/reopen, cropper maths, novelty detector fit/score/save/load, abstention rules, application handlers end to end.

Browser (headless Chromium via Playwright, trained model): home → workspace → upload → full analysis → two-click crop → selection analysis → compare → reset → history → about, on desktop and mobile widths; no page errors; no horizontal overflow at 390 px after moving the mobile CSS into the page head (Gradio scopes `css=` rules under its container). Screenshots: `docs/screenshots/`.

Not tested here: the Windows .bat on a real Windows machine; the Colab notebook end to end; a live Hugging Face Space; real field photographs.

## 4. Known limitations and open items
* Gradio 6.32's built-in image-editor crop does not export the cropped pixels (verified in the browser), so a custom cropper is used instead; it is deliberately simple (two clicks or sliders), not a drag-handle editor.
* Disease-information references link to the organisations' resource portals rather than individual pages; a specialist should review the texts before any public launch.
* The novelty detector is calibrated on PlantVillage-style photographs: field photos of supported crops may be flagged as unusual.
* Softmax confidence is capped around 90 % by label smoothing; the 60 % threshold was chosen so that only 0.67 % of in-distribution test images abstain.
* Dataset licence: treat as CC BY-SA 3.0 with attribution; the original GitHub repository has no licence file.

## 5. Costs
* Local use: free (CPU laptop; ~1 GB RAM, ~1 GB disk including dependencies).
* Hugging Face Spaces (checked 2026-11-16): CPU Basic hardware is free of hourly charges, but Hugging Face states Gradio Spaces require a PRO plan (~US$9/month) for personal accounts; free personal accounts may host up to two Gradio Spaces on ZeroGPU hardware, which this app can use. No GPU is needed. Nothing has been purchased.
* Re-training: free on Google Colab's GPU tier (~30–45 min on a T4) or ~2 h on a 4-core CPU.

## 6. Files
- `.gitattributes`
- `.gitignore`
- `README.md`
- `app.py`
- `artifacts/class_mapping.json`
- `artifacts/confusion_matrix.png`
- `artifacts/evaluation_metrics.json`
- `artifacts/evaluation_report.md`
- `artifacts/model_config.json`
- `artifacts/ood_statistics.npz`
- `artifacts/per_class_f1.png`
- `artifacts/plant_doctor_efficientnet_b0.pt`
- `artifacts/preprocessing_config.json`
- `artifacts/training_curves.png`
- `artifacts/training_log.json`
- `artifacts/training_metadata.json`
- `assets/logo.svg`
- `assets/samples/ATTRIBUTION.md`
- `assets/samples/apple_cedar_rust.jpg`
- `assets/samples/corn_common_rust.jpg`
- `assets/samples/grape_black_rot.jpg`
- `assets/samples/potato_late_blight.jpg`
- `assets/samples/tomato_early_blight.jpg`
- `assets/samples/tomato_healthy.jpg`
- `config.py`
- `core/__init__.py`
- `core/cropper.py`
- `core/disease_info.py`
- `core/history.py`
- `core/image_processing.py`
- `core/report.py`
- `data/disease_information.json`
- `docs/USER_GUIDE.md`
- `models/__init__.py`
- `models/gradcam.py`
- `models/inference.py`
- `models/model.py`
- `models/quality.py`
- `notebooks/training_colab.ipynb`
- `requirements.txt`
- `run_plant_doctor.bat`
- `run_plant_doctor.sh`
- `tests/__init__.py`
- `tests/conftest.py`
- `tests/test_app.py`
- `tests/test_image_processing.py`
- `tests/test_inference.py`
- `tests/test_model.py`
- `tests/test_support_modules.py`
- `tests/test_training_pipeline.py`
- `training/__init__.py`
- `training/evaluate.py`
- `training/preprocessing.py`
- `training/train.py`
- `ui/__init__.py`
- `ui/render.py`
- `ui/styles.py`
