# Industrial Defect Detection & Quality Inspection (Computer Vision Project)

**Pipeline:** Camera/Image → YOLO model → Detect defect → Classify defect → Severity → Verdict → Database → Dashboard

This guide takes you from zero to a working project. Follow the steps in order.

---

## Part 0: The basics you need to understand first

**Computer Vision (CV):** teaching computers to understand images.

**Three levels of CV tasks (know the difference for your viva):**
| Task | Question it answers | Output |
|---|---|---|
| Classification | "What is in this image?" | One label |
| **Object detection** (our project) | "What is where?" | Boxes + label + confidence |
| Segmentation | "Which exact pixels?" | Pixel mask |

**Our task = object detection.** The model looks at a steel surface image and draws a box around each defect (scratch, crack-like crazing, etc.) with a class name and confidence score.

**YOLO (You Only Look Once):** a fast, popular detection model. It looks at the whole image once and predicts all boxes together. We use **YOLOv8** via the `ultralytics` library because it is beginner-friendly and used in industry.

**Key terms:**
- **Bounding box:** rectangle around a defect (x1, y1, x2, y2).
- **Confidence:** how sure the model is (0 to 1).
- **IoU (Intersection over Union):** how much a predicted box overlaps the true box.
- **mAP@0.5:** the standard score for detectors (higher is better; 0.7+ is good for this dataset).
- **Precision:** of the defects it flagged, how many were real. **Recall:** of the real defects, how many it found. In factories, missing a defect (low recall) is usually worse than a false alarm.
- **Epoch:** one full pass through the training data.
- **Overfitting:** model memorizes training images but fails on new ones. We check this with a separate validation set.
- **Transfer learning:** we start from `yolov8n.pt` (already trained on everyday objects) and fine-tune it on defects. This is why we need only ~1,800 images.

---

## Part 1: Setup

1. Install **Python 3.10 or 3.11** from python.org.
2. Create a folder `defect-inspection` and put all project files in it.
3. Create a virtual environment and install packages:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

4. **GPU:** Training on CPU is very slow. Use **Google Colab (free GPU)** or **Kaggle Notebooks (free GPU)** for the training step only. Everything else runs fine on a laptop. See Part 4.

---

## Part 2: Get the images (dataset)

Use the **NEU Surface Defect Database (NEU-DET)**: 1,800 grayscale images of hot-rolled steel strips, 200×200 px, 6 defect types, 300 images each, with bounding-box annotations (Pascal VOC XML).

| Class | What it looks like |
|---|---|
| crazing | network of fine cracks |
| inclusion | embedded foreign particles/spots |
| patches | irregular patchy regions |
| pitted_surface | small pits/holes |
| rolled-in_scale | scale pressed into the surface |
| scratches | linear scratch marks |

**How to download:** search **"NEU surface defect database"** or **"NEU-DET"** on Kaggle and download it (free Kaggle account needed). Unzip it somewhere, for example `NEU-DET/`. It should contain images and `.xml` annotation files (usually in `train/` and `validation/` subfolders, our script handles any layout).

**Other datasets you can mention / use as extensions:**
- **MVTec AD**: 15 product categories (bottle, cable, screw, etc.) for *anomaly detection* (needs only good images for training). Great for an advanced extension.
- **Severstal Steel Defect (Kaggle)**: larger, segmentation-based.
- **Your own images:** photograph objects with defects and label them with **Roboflow** or **LabelImg** (export in YOLO format). This makes your project look original.

---

## Part 3: Prepare the data (convert to YOLO format)

YOLO needs labels as text files: `class cx cy width height` (all normalized 0 to 1). NEU-DET gives XML, so we convert and split into train (80%) and validation (20%):

```bash
python prepare_dataset.py --src path/to/NEU-DET --dst dataset
```

This creates:
```
dataset/
  images/train  images/val
  labels/train  labels/val
  data.yaml
```

**Check it worked:** open any image in `dataset/images/train` and its `.txt` in `dataset/labels/train`. You should see lines like `5 0.512 0.430 0.210 0.980`.

---

## Part 4: Train the model

### Option A: Local (if you have an NVIDIA GPU)
```bash
python train.py --epochs 50
```

### Option B: Google Colab (recommended, free GPU)
1. Go to colab.research.google.com → New notebook → Runtime → Change runtime type → **T4 GPU**.
2. Zip your `dataset/` folder, upload it to Colab (left sidebar → Files), then run:

```python
!pip install ultralytics
!unzip -q dataset.zip -d /content

# Fix the path inside data.yaml for Colab
import re, pathlib
p = pathlib.Path("/content/dataset/data.yaml")
txt = p.read_text()
txt = re.sub(r"^path:.*$", "path: /content/dataset", txt, flags=re.M)
p.write_text(txt)

from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="/content/dataset/data.yaml", epochs=50, imgsz=640, batch=16,
            project="runs", name="neu_det", patience=15, flipud=0.5, fliplr=0.5, degrees=10)
```
3. When done, download `runs/neu_det/weights/best.pt` (right-click → Download) and put it in your project folder at `runs/neu_det/weights/best.pt`.

**Training time:** about 10 to 20 minutes on a free Colab GPU.

### What gets produced in `runs/neu_det/`
- `weights/best.pt`: the model you will use
- `results.png`: loss and mAP curves over epochs (put this in your report)
- `confusion_matrix.png`: which defect gets confused with which (great for analysis)
- `val_batch0_pred.jpg`: sample predictions

---

## Part 5: Evaluate honestly

Open `results.png` and check:
- Loss should go **down** and mAP should go **up**.
- If training loss keeps falling but validation mAP falls, you are overfitting. Use fewer epochs or more augmentation.

Report these numbers: **mAP@0.5, precision, recall, per-class AP**. Expect roughly 0.70 to 0.80 mAP@0.5 on NEU-DET with YOLOv8n/s. Do not claim more than you measured.

Look at the confusion matrix and write 2 to 3 sentences about which classes are hard (typically *crazing* and *rolled-in_scale*, because they have faint, diffuse textures).

---

## Part 6: Severity, verdict, and database (the "industrial" part)

This is what makes it more than a plain detector.

**`severity.py`** assigns each defect Low/Medium/High:
- Compute the defect's **area as a % of the image**.
- ≤ 2% → Low, ≤ 8% → Medium, > 8% → High.
- Critical defect types (crazing, inclusion, pitted_surface) get bumped up one level.

**Verdict per product:**
- No defects → **PASS**
- Any High defect, or 3+ defects → **REJECT**
- Otherwise → **REVIEW** (send to human)

**`db.py`** stores every inspection in SQLite (`inspections.db`, created automatically, no setup needed):

| Table `inspections` | Table `defects` |
|---|---|
| id, image_name, created_at, verdict, num_defects | id, inspection_id, class_name, confidence, x1,y1,x2,y2, area_pct, severity |

**`inspector.py`** ties the pipeline together. Test it on a folder of images (use some from `dataset/images/val`):

```bash
python inspector.py --weights runs/neu_det/weights/best.pt --folder dataset/images/val
```
Annotated images go to `outputs/`, and every result is saved in the database.

> The thresholds in `severity.py` are reasonable starting rules, not industry standards. In your report, say they are configurable and would be set by a quality engineer in a real factory.

---

## Part 7: The dashboard

```bash
streamlit run app.py
```
A browser tab opens with:
- **Inspect tab:** upload images, see original vs detected, defect table, verdict.
- **Dashboard tab:** counts of passed/review/rejected, defects by type and severity, full history.

If the model path in the sidebar is wrong, the app tells you to train first.

---

## Part 7b: Hybrid detection (works on ANY steel photo, even before training)

**Problem we found:** a YOLO model trained only on NEU-DET (200x200 grayscale flat steel close-ups) can miss defects in an ordinary colour photo of a steel part. This is called a *domain gap*.

**Fix:** the app now runs two detectors together:

| Detector | Colour of box | Needs training? | Finds |
|---|---|---|---|
| YOLO (`best.pt`) | green | yes | the 6 NEU-DET defect types |
| Classical (`classic_detector.py`) | red | **no** | thin scratch / crack-like lines on any steel photo |

How the classical detector works (explain this in your viva):
1. Convert to grayscale, remove lighting gradients.
2. Multi-scale Hessian ridge filter finds thin line-like structures.
3. Long outlines of the part itself (object edges) are removed, because they are not defects.
4. Hysteresis thresholding (strong seed + weaker connected pixels) joins a broken scratch into one object.
5. Components must be long, thin, and have similar brightness on both sides (a scratch is a ridge, an object edge is a step).

**Strict mode** (`STRICT_MODE` in `severity.py`, or the sidebar checkbox): any detected defect gives REJECT, which suits zero-defect parts. Turn it off for the graded PASS / REVIEW / REJECT logic.

**Run without a trained model:**
```bash
python inspector.py --folder my_photos --no-yolo
streamlit run app.py        # leave the weights path as is; the app falls back to classical-only
```

**Honest limits (put these in your report):**
- The classical detector only finds thin, linear defects (scratches, cracks). It does not find dents, stains or missing parts.
- I tuned and tested it on one scratched-bar photo plus synthetic images, not a large dataset. Other photos may give false alarms (grooves, text, weld lines) or misses. Use the "Classical sensitivity" slider: lower finds fainter scratches, higher gives fewer false alarms.
- For reliable results on your own products, still collect and label 80-150 photos and fine-tune YOLO (see Part 9).

---

## Part 8: Project timeline (suggested, ~3 weeks)

| Days | Task |
|---|---|
| 1-2 | Learn basics (Part 0), install, run YOLOv8 on any sample image |
| 3-4 | Download dataset, convert, inspect labels visually |
| 5-7 | Train, evaluate, write down metrics |
| 8-10 | Severity + database + `inspector.py` |
| 11-13 | Dashboard, polish, screenshots |
| 14-15 | Improvements (Part 9), report, slides |

---

## Part 9: Ideas to improve (pick 1 or 2 to stand out)

1. **Compare models:** train `yolov8n`, `yolov8s`, `yolov8m` and compare mAP vs speed (FPS) in a table.
2. **Add preprocessing:** CLAHE contrast enhancement with OpenCV before detection, then compare mAP.
3. **Real-time camera:** use `cv2.VideoCapture(0)` and call `inspect_image` on each frame.
4. **Anomaly detection:** train on only "good" images using MVTec AD with PatchCore/Anomalib.
5. **Own data:** shoot and label 100+ real images (coins, PCBs, fabric, bottle caps) with Roboflow.
6. **Export:** `model.export(format="onnx")` for faster deployment.
7. **Alerts:** auto-flag REJECT results with a red banner or email.

---

## Part 10: Common problems

| Problem | Fix |
|---|---|
| `No image/XML pairs found` | Check `--src` path; XML and image file names must match |
| CUDA out of memory | Lower `--batch` to 8 or 4 |
| Very low mAP (< 0.3) | Check labels are correct; train longer; verify `data.yaml` path |
| Streamlit can't find model | Train first, or set the correct path in the sidebar |
| Training is extremely slow | You are on CPU; use Colab |

---

## Part 11: Viva questions you should be ready for

1. Why object detection and not classification? *(Need location and multiple defects per image.)*
2. Why YOLO? *(Single-pass, fast enough for production lines.)*
3. What is transfer learning and why use it? *(Small dataset; pretrained features.)*
4. What does mAP mean? *(Average precision across classes at an IoU threshold.)*
5. Precision vs recall in a factory? *(Missed defects are costly, so favor recall.)*
6. How do you decide severity? *(Area plus defect criticality; configurable rules.)*
7. How would you deploy this on a real line? *(Fixed camera and lighting, ONNX/TensorRT, edge device, PLC signal for reject.)*
8. Limitations? *(Dataset is only steel, 200×200 grayscale; lighting changes; rule-based severity.)*

---

## Files in this project

| File | Purpose |
|---|---|
| `requirements.txt` | Python packages |
| `prepare_dataset.py` | VOC XML → YOLO conversion + train/val split |
| `train.py` | Train and validate YOLOv8 |
| `severity.py` | Severity and verdict rules |
| `db.py` | SQLite database layer |
| `classic_detector.py` | Training-free scratch detector (OpenCV) |
| `inspector.py` | Full hybrid pipeline + folder CLI |
| `app.py` | Streamlit dashboard |
