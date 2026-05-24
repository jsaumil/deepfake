# Autonomous AI Researcher - DeepFake Video Detector

## Goal
Maximize the **val_auc** metric (**Area Under ROC Curve**) in `results.json`.

Higher value = better performance.

---

## Rules

- You may **ONLY edit `train.py`**
- **Do NOT edit** `prepare.py` or `program.md`
- Training runs for **exactly 10 minutes (wall-clock time)** and cannot be changed
- The model receives video sequences with shape:

```python
(B, 8, 3, 224, 224)
```

- Baseline model:
  - Fusion of **Swin Transformer**
  - **GAN Discriminator**
  - **BiLSTM**

- After every experiment:
  1. Read `results.json`
  2. Check whether `val_auc` improved
  3. Record findings in `experiment_log.md`

---

## Suggested Experiment Ideas

### Backbone Changes
- Change `embed_dim` of Swin Transformer
- Change `embed_dim` of GAN backbone
- Adjust feature dimensions

### Regularization
- Try different dropout values:
  - `0.1`
  - `0.2`
  - `0.3`
  - `0.4`
  - `0.5`

### Temporal Aggregation
Compare different approaches:

- BiLSTM
- Average Pooling
- Max Pooling
- Attention-based pooling

### Optimization Experiments

Try different optimizers:

- AdamW
- SGD
- RMSProp

Learning rate ideas:

```python
1e-3
5e-4
1e-4
5e-5
```

### Fusion Improvements

Possible ideas:

- Add residual connections in Fusion MLP
- Add normalization layers
- Modify hidden layer sizes
- Add attention mechanisms

---

## Experiment Workflow

### Step 1: Modify model

Edit:

```bash
train.py
```

Implement one hypothesis/change.

---

### Step 2: Run training

```bash
python train.py
```

Wait for completion (~10 minutes).

---

### Step 3: Evaluate performance

Read:

```bash
results.json
```

Check:

```json
{
    "val_auc": 0.95
}
```

Compare with previous experiment.

---

### Step 4: Update experiment log

Record:

- Experiment number
- Changes made
- Hypothesis
- Validation AUC
- Observations
- Next plan

Example:

```md
## Experiment 1

Change:
- Increased Swin embed_dim from 96 → 128

Hypothesis:
- Larger feature representation improves detection quality.

Result:
- val_auc = 0.921

Observation:
- Slight improvement.

Next:
- Test dropout = 0.3
```

---

## Objective

Iteratively improve the model and maximize:

**val_auc**