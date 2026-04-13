# Sentiment-aware Image Captioning with Visually Consistent Sentiment Calibration (VCSC)

This repository provides a **manuscript code** of the paper:

**Sentiment-aware image captioning with visually consistent sentiment calibration**

It is organized to mirror the paper's **unified 5-class pipeline**:
- weak sentiment supervision with **TextBlob**
- **five-class** sentiment labeling and emoji mapping
- **classical baselines** using TF-IDF:
  - SVM
  - Random Forest
  - Decision Tree
  - Naive Bayes
- **deep sentiment baselines**:
  - LSTM
  - GRU
  - CNN
  - Transformer
- **BERT** five-class sentiment classification
- **Vision--GPT** caption generation using a ViT encoder and GPT-2 decoder
- **VCSC** as a **post-hoc calibration** stage fitted on validation outputs and applied to test predictions

---

## Implementation note

As with most deep-learning pipelines, **exact numerical reproduction may vary** due to:
- software versions
- hardware differences
- random seeds
- dataset export format
- preprocessing and training configuration

The repository is therefore best understood as a **manuscript code** of the proposed method.

---

## Main files

- `manuscript_full_pipeline_refined.py` — unified end-to-end implementation
- `requirements.txt` — Python dependencies
- `REPRODUCIBILITY_NOTE.md` — short reproducibility statement for manuscript/repo use
- `REPO_STRUCTURE.txt` — expected project layout

---


---

## Installation

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) NLTK resources

The script downloads required NLTK resources automatically if they are missing. If needed, you can also run:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

---

## Dataset input expectations

The implementation is written for **Flickr30k** and supports the common public export layouts, including Kaggle-style packages.

It can work in either of the following ways:

### Option A — explicit paths
Provide both:
- `--captions-file`
- `--images-dir`

### Option B — dataset root auto-discovery
Provide:
- `--data-root`

The script will try to discover:
- caption file: `results.csv`, `captions.csv`, or `captions.txt`
- image folder: `flickr30k_images/` or `images/`

It also handles common alternate column names such as:
- `image_name`, `filename`, `file_name` → `image`
- `comment`, `sentence`, `caption_text` → `caption`

---

## Expected folder structure

```text
project_root/
├── manuscript_full_pipeline_refined.py
├── requirements.txt
├── README.md
├── REPRODUCIBILITY_NOTE.md
├── REPO_STRUCTURE.txt
├── data/
│   └── flickr30k/
│       ├── results.csv
│       ├── captions.txt
│       └── flickr30k_images/
│           ├── 1000092795.jpg
│           ├── 10002456.jpg
│           └── ...
└── outputs/
    ├── seed_42/
    ├── seed_123/
    └── ...
```

---

## Pipeline overview

The implementation follows the manuscript structure:

1. **Data loading and preprocessing**
   - load Flickr30k captions and image paths
   - lowercase, normalize, and optionally remove stopwords

2. **Weak sentiment supervision**
   - compute TextBlob polarity
   - map polarity to 5 classes using manuscript thresholds:
     - very negative
     - negative
     - neutral
     - positive
     - very positive

3. **Split creation**
   - train / validation / test
   - stratified by the weak sentiment labels

4. **Sentiment baselines**
   - TF-IDF + SVM / Random Forest / Decision Tree / Naive Bayes
   - LSTM / GRU / CNN / Transformer
   - BERT classifier

5. **Caption generation**
   - Vision Transformer encoder
   - GPT-2 decoder
   - Vision--GPT training/evaluation flow

6. **Post-hoc VCSC calibration**
   - fit coarse visual affect on validation split
   - use BERT confidence and visual affect to recalibrate test predictions

7. **Outputs for review and inspection**
   - metrics
   - confusion matrices
   - classification reports
   - per-model predictions
   - per-seed summaries

---

## Example CLI commands

### Full pipeline with explicit paths

```bash
python manuscript_full_pipeline_refined.py \
  --captions-file data/flickr30k/results.csv \
  --images-dir data/flickr30k/flickr30k_images \
  --output-dir outputs \
  --run-all
```

### Full pipeline with dataset-root auto-discovery

```bash
python manuscript_full_pipeline_refined.py \
  --data-root data/flickr30k \
  --output-dir outputs \
  --run-all
```

### Run only sentiment baselines + BERT + VCSC

```bash
python manuscript_full_pipeline_refined.py \
  --data-root data/flickr30k \
  --output-dir outputs \
  --run-classical \
  --run-torch-baselines \
  --run-bert \
  --run-vcsc
```

### Run captioning only

```bash
python manuscript_full_pipeline_refined.py \
  --data-root data/flickr30k \
  --output-dir outputs \
  --run-captioning
```

### Faster smoke test

```bash
python manuscript_full_pipeline_refined.py \
  --data-root data/flickr30k \
  --output-dir outputs_debug \
  --run-all \
  --max-samples 1000 \
  --sentiment-epochs 1 \
  --caption-epochs 1
```

### Multi-seed evaluation

```bash
python manuscript_full_pipeline_refined.py \
  --data-root data/flickr30k \
  --output-dir outputs \
  --run-classical \
  --run-torch-baselines \
  --run-bert \
  --run-vcsc \
  --multi-seed 42 123 456
```

---

## Key CLI arguments

- `--captions-file` : explicit caption file path
- `--images-dir` : explicit image directory path
- `--data-root` : Flickr30k root for auto-discovery
- `--output-dir` : where outputs are written
- `--seed` : single random seed
- `--multi-seed` : repeated runs with several seeds
- `--max-samples` : optional subset size for debugging
- `--train-ratio`, `--val-ratio`, `--test-ratio` : split ratios
- `--sentiment-epochs` : epochs for sentiment models
- `--caption-epochs` : epochs for captioning model
- `--vcsc-delta-c` : confidence threshold for VCSC activation
- `--run-all` : run all components
- `--run-classical` : run TF-IDF baselines only
- `--run-torch-baselines` : run LSTM/GRU/CNN/Transformer baselines
- `--run-bert` : run BERT classifier
- `--run-vcsc` : run VCSC calibration on top of BERT outputs
- `--run-captioning` : run Vision--GPT captioning
- `--export-splits` : save train/val/test CSV files

---

## Output artifacts

Typical files written under `outputs/seed_<N>/` include:
- `results_summary.json`
- confusion matrices
- classification reports
- per-model predictions
- VCSC outputs
- captioning metrics

If `--multi-seed` is used, the repository also writes:
- `multi_seed_summary.json`

---

## reproducibility guidance

> The repository provides a manuscript code of the proposed 5-class sentiment-aware image captioning pipeline, including Vision--GPT caption generation, multiple sentiment baselines, and post-hoc VCSC calibration. Exact numerical reproduction may vary with software versions, hardware, random seeds, and dataset preparation.

---


