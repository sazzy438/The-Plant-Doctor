# Plant Doctor — User Guide

*Understand your plants. Protect your harvest.*

Plant Doctor screens photographs of plant leaves with a neural network and shows you the most likely condition, how sure the model is, which parts of the image influenced the result, and general information about the condition. It is an **educational screening aid**, not a diagnosis.

## 1. Starting the application

**Windows:** double-click `run_plant_doctor.bat`. The first run installs everything (5–10 minutes); later runs start in seconds. Your browser opens at `http://127.0.0.1:7860`. Keep the black window open while you use the app; close it to stop.

**macOS / Linux:** run `./run_plant_doctor.sh` in a terminal.

**Online (Hugging Face Space):** open the Space link in any browser; nothing to install.

## 2. Analysing a leaf

1. Open the **Analyse** tab (or press *Upload Leaf Image* on the Home tab).
2. Drag a photo onto the upload box or click it to choose a file. JPG, PNG and WEBP up to 25 MB are accepted.
3. *(Optional)* Select a region: click two opposite corners on the preview. The darkened area is ignored; only the bright rectangle is analysed. Open **Fine-tune the selection** to adjust the edges with sliders, or press **Reset crop**.
4. Press **Analyse full image** or **Analyse selection**. **Compare full vs selection** runs both and shows them side by side.
5. Read the results (see below). Press **Download PDF report** to save a report, or **Analyse another image** to start over.

### Tips for good photographs
* One leaf, filling most of the frame, on a plain background.
* Even daylight, no harsh shadows, camera held steady and focused.
* Photograph the surface that shows the symptoms (often the upper side; for mildews and moulds also the underside).

## 3. Reading the results

| Section | Meaning |
|---|---|
| **Most likely condition** | The crop and condition the model prefers, with a *Healthy* or *Possible disease* badge. |
| **Confidence ring** | The softmax probability of that condition (0–100 %). It shows the model's preference among its 38 known categories, **not** the chance that the result is correct for your plant. |
| **Model predictions** | The top condition and the three runners-up. Close values mean the model is unsure. |
| **Visual explanation** | Drag the slider to compare the photo with the Grad-CAM overlay. Warm colours show where the model looked. If the warm area is on the background or the leaf edge, be sceptical of the result. |
| **Image checks** | Sharpness, plant-coloured area, novelty (how unusual the image is compared with training leaves) and top confidence. |
| **About this condition** | Symptoms, favourable conditions, general prevention and next steps, with the references used. |

### "No confident result"
Plant Doctor refuses to name a condition when the photo is too blurry, looks unlike the leaf photographs it was trained on, or the model's confidence is below 60 %. Retake the photo following the tips above. The model's best (unreliable) guess is shown for transparency only.

## 4. History
The **History** tab lists the analyses from your current browser session with thumbnails, prediction, confidence and time. Choose one and press **Reopen** to see it again. Nothing is saved on the server; closing the tab clears the history.

## 5. What Plant Doctor can and cannot do
* It knows **14 crops and 38 conditions** (listed on the Home tab) and nothing else. Other crops, nutrient problems, herbicide injury and most pests cannot be recognised.
* It was trained on laboratory-style photos. Accuracy on field photos is lower and has not been measured.
* It never recommends pesticides. Confirm any suspected disease with a plant diagnostic laboratory or your agricultural extension service before treating.

## 6. Troubleshooting
| Problem | What to do |
|---|---|
| "The trained model could not be loaded" | Make sure `artifacts/plant_doctor_efficientnet_b0.pt` exists. See README for training or downloading it. |
| Browser does not open | Open `http://127.0.0.1:7860` manually. |
| "Python was not found" (Windows) | Install Python 3.11+ from python.org and tick *Add python.exe to PATH*, then run the .bat again. |
| Analysis is slow | On a CPU an analysis takes 0.2–2 s. Very large photos are downscaled automatically. |
| A file is rejected | Only JPG/PNG/WEBP are accepted; the image must be at least 64 px per side and at most 8000 px per side and 25 MB. |
