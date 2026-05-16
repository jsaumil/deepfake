# DeepFake Detection — Fusion Model (Swin Transformer + GAN)

## Architecture (matches the flow diagram)

```
Dataset Input (DFDC / FaceForensics++)
        ↓
Image Enhancement  (noise reduction, sharpness, contrast)
        ↓
Pre-processing  (resize 224×224, train/test split, augmentation)
        ↓
   ┌────────────────────────┐
   │     Model Training     │
   │  ┌──────────────────┐  │
   │  │ Swin Transformer │  │  ← Shifted Window Attention (768-dim features)
   │  └────────┬─────────┘  │
   │           │             │
   │  ┌────────┴──────────┐ │
   │  │  GAN Discriminator│ │  ← Synthesis artefact features (256-dim)
   │  └────────┬──────────┘ │
   └───────────┼────────────┘
               │  Concatenate → 1024-dim
               ↓
         Fusion MLP Head
               ↓
    Output: REAL / FAKE + confidence score
```

---

## Project Structure

| File | Purpose |
|------|---------|
| `swin_transformer.py` | Swin-T backbone (global shifted-window attention) |
| `gan.py` | Generator + Discriminator (GAN pre-training) |
| `fusion_model.py` | Concatenates Swin + GAN features → binary classifier |
| `dataset.py` | Image/Video dataset + enhancement + augmentation |
| `train.py` | Two-phase training pipeline |
| `evaluate.py` | All metrics: AUC-ROC, F1, Accuracy, Precision, Loss, Confusion Matrix |
| `inference.py` | Run on a single image, video, or folder |
| `requirements.txt` | Python dependencies |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Data Folder Structure

```
data/
  train/
    real/   *.jpg / *.png
    fake/   *.jpg / *.png
  val/
    real/
    fake/
  test/
    real/
    fake/
```

---

## Training

```bash
# Full two-phase training (GAN pre-train → fusion fine-tune)
python train.py \
  --data_root ./data \
  --gan_epochs 10 \
  --epochs 30 \
  --batch_size 16 \
  --device cuda

# Skip GAN pre-training (if checkpoint already exists)
python train.py \
  --data_root ./data \
  --gan_epochs 0 \
  --epochs 30 \
  --batch_size 16
```

---

## Evaluation

```bash
python evaluate.py \
  --checkpoint checkpoints/fusion_best.pt \
  --data_root ./data \
  --split test \
  --plot_roc \
  --save_results results.json
```

**Output metrics:**
- Loss (Binary Cross-Entropy)
- AUC-ROC
- F1-score (harmonic mean P & R)
- Accuracy (correct predictions / total)
- Precision (TP / predicted positives)
- Confusion Matrix (TP · TN · FP · FN)

---

## Inference

```bash
# Single image
python inference.py --checkpoint checkpoints/fusion_best.pt --input face.jpg

# Video (aggregates frame predictions)
python inference.py --checkpoint checkpoints/fusion_best.pt --input video.mp4 \
  --strategy mean --max_frames 30

# Folder of images/videos
python inference.py --checkpoint checkpoints/fusion_best.pt --input ./test_faces/
```

---

## Key Design Decisions

- **Swin Transformer** (96-dim base, 768-dim output) handles global context via shifted window attention.
- **GAN Discriminator** (64-dim base, 256-dim output) is pre-trained adversarially to detect synthesis artefacts, then its backbone is frozen/fine-tuned in the fusion model.
- **Concatenation** of both 768 + 256 = 1024-dim feature vectors feeds a 4-layer MLP fusion head.
- **Image normalization**: inputs to the model should be in `[0, 1]`; the fusion model handles the `[-1, 1]` conversion needed by the GAN backbone internally.
