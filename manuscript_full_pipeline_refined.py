#!/usr/bin/env python3
"""
Manuscript-faithful reference implementation for:
"Sentiment-aware image captioning with visually consistent sentiment calibration"

This file is organized to reflect the paper's unified 5-class pipeline:
1. Data loading and caption preprocessing.
2. Weak sentiment supervision with TextBlob and five-class mapping.
3. Train/validation/test splitting.
4. Multiple sentiment baselines:
   - TF-IDF + SVM / Random Forest / Decision Tree / Naive Bayes
   - LSTM / GRU / CNN / Transformer
   - BERT
5. Vision--GPT caption generation.
6. Post-hoc VCSC fitted on validation data and applied to test predictions.
7. Export of metrics, predictions, confusion matrices, and reports.

Honest implementation note
--------------------------
This repository is intended as a manuscript-aligned reference implementation.
It aims to faithfully represent the method described in the paper, while making
reasonable engineering choices where the manuscript does not specify every low-
level detail. Exact numerical reproduction may vary with package versions,
hardware, random seeds, and dataset preparation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import nltk
from textblob import TextBlob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2TokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    VisionEncoderDecoderModel,
    default_data_collator,
    set_seed,
)

try:
    import evaluate
except Exception:
    evaluate = None


SENTIMENT_LABELS = [
    "VERY_POSITIVE",
    "POSITIVE",
    "NEUTRAL",
    "NEGATIVE",
    "VERY_NEGATIVE",
]
LABEL_TO_ID = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
SENTIMENT_TO_EMOJI = {
    "VERY_POSITIVE": "😍",
    "POSITIVE": "🙂",
    "NEUTRAL": "😐",
    "NEGATIVE": "🙁",
    "VERY_NEGATIVE": "😡",
}
THRESHOLDS = (-0.60, -0.10, 0.10, 0.60)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)



def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def maybe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def ensure_nltk() -> None:
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt", "punkt"),
    ]
    for resource_path, name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(name, quiet=True)



def clean_caption(text: str, remove_stopwords: bool = True) -> str:
    from nltk.corpus import stopwords

    text = str(text).lower().strip()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if remove_stopwords:
        sw = set(stopwords.words("english"))
        text = " ".join(tok for tok in text.split() if tok not in sw)
    return text



def textblob_polarity(text: str) -> float:
    return float(TextBlob(str(text)).sentiment.polarity)



def polarity_to_5class(p: float, thresholds: Tuple[float, float, float, float] = THRESHOLDS) -> str:
    t_neg2, t_neg1, t_pos1, t_pos2 = thresholds
    if p < t_neg2:
        return "VERY_NEGATIVE"
    if p < t_neg1:
        return "NEGATIVE"
    if p <= t_pos1:
        return "NEUTRAL"
    if p <= t_pos2:
        return "POSITIVE"
    return "VERY_POSITIVE"



def class_to_signed_polarity(label_id: int) -> float:
    return {
        LABEL_TO_ID["VERY_NEGATIVE"]: -1.0,
        LABEL_TO_ID["NEGATIVE"]: -0.5,
        LABEL_TO_ID["NEUTRAL"]: 0.0,
        LABEL_TO_ID["POSITIVE"]: 0.5,
        LABEL_TO_ID["VERY_POSITIVE"]: 1.0,
    }[int(label_id)]



def signed_polarity_to_class_id(value: float, thresholds: Tuple[float, float, float, float] = THRESHOLDS) -> int:
    return LABEL_TO_ID[polarity_to_5class(float(value), thresholds)]



def compute_sentiment_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }





def discover_dataset_paths(data_root: str | Path) -> Tuple[Path, Path]:
    """Discover a caption file and image directory under a Flickr30k root.

    Supported caption files: results.csv, captions.csv, captions.txt
    Supported image directories: flickr30k_images, images
    """
    root = Path(data_root)
    caption_candidates = [root / 'results.csv', root / 'captions.csv', root / 'captions.txt']
    image_candidates = [root / 'flickr30k_images', root / 'images']

    captions_file = next((p for p in caption_candidates if p.exists()), None)
    images_dir = next((p for p in image_candidates if p.exists()), None)

    if captions_file is None or images_dir is None:
        raise FileNotFoundError(
            f'Could not auto-discover Flickr30k inputs under {root}. '
            'Expected one of {results.csv, captions.csv, captions.txt} and '
            'one of {flickr30k_images/, images/}.'
        )
    return captions_file, images_dir


def resolve_input_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    """Resolve dataset input paths from either explicit files or a dataset root."""
    captions_file = getattr(args, 'captions_file', None)
    images_dir = getattr(args, 'images_dir', None)
    data_root = getattr(args, 'data_root', None)

    if captions_file and images_dir:
        return Path(captions_file), Path(images_dir)
    if data_root:
        return discover_dataset_paths(data_root)
    raise ValueError('Provide either --captions-file and --images-dir, or --data-root.')

def load_flickr30k(captions_file: str | Path, images_dir: str | Path) -> pd.DataFrame:
    captions_file = Path(captions_file)
    images_dir = Path(images_dir)

    try:
        df = pd.read_csv(captions_file)
    except Exception:
        df = pd.read_csv(captions_file, delimiter=",")

    rename_map = {}
    if "image_name" in df.columns:
        rename_map["image_name"] = "image"
    if "comment" in df.columns:
        rename_map["comment"] = "caption"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"image", "caption"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    df = df[["image", "caption"]].copy()
    df["image"] = df["image"].map(lambda x: str(images_dir / str(x).strip()))
    df["caption"] = df["caption"].astype(str).str.strip().str.lower()
    df = df[df["image"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    return df



def add_weak_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cleaned_caption"] = df["caption"].apply(clean_caption)
    df["polarity"] = df["cleaned_caption"].apply(textblob_polarity)
    df["sentiment"] = df["polarity"].apply(polarity_to_5class)
    df["label"] = df["sentiment"].map(LABEL_TO_ID)
    df["emoji"] = df["sentiment"].map(SENTIMENT_TO_EMOJI)
    return df



def stratified_splits(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train/val/test ratios must sum to 1.0")

    train_df, tmp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=df["label"],
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        tmp_df,
        test_size=relative_test,
        random_state=seed,
        stratify=tmp_df["label"],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class CaptionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_processor, tokenizer, max_length: int = 64) -> None:
        self.df = df.reset_index(drop=True)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image = Image.open(row["image"]).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values[0]
        enc = self.tokenizer(
            row["caption"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = enc.input_ids[0]
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "decoder_attention_mask": enc.attention_mask[0],
        }



def build_vision_gpt2_model() -> Tuple[VisionEncoderDecoderModel, Any, GPT2TokenizerFast]:
    encoder_name = "google/vit-base-patch16-224-in21k"
    decoder_name = "gpt2"
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_name, decoder_name)
    image_processor = AutoImageProcessor.from_pretrained(encoder_name)
    tokenizer = GPT2TokenizerFast.from_pretrained(decoder_name)
    tokenizer.pad_token = tokenizer.eos_token

    model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = 64
    model.config.num_beams = 4
    model.config.no_repeat_ngram_size = 2
    model.config.length_penalty = 1.0
    return model, image_processor, tokenizer



def compute_caption_metrics(eval_preds, tokenizer: GPT2TokenizerFast) -> Dict[str, float]:
    predictions, labels = eval_preds
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    references = [[s.strip()] for s in label_str]
    predictions_clean = [s.strip() for s in pred_str]
    metrics: Dict[str, float] = {}

    if evaluate is not None:
        try:
            bleu = evaluate.load("bleu")
            rouge = evaluate.load("rouge")
            meteor = evaluate.load("meteor")
            metrics["bleu"] = float(bleu.compute(predictions=predictions_clean, references=references)["bleu"])
            rouge_out = rouge.compute(predictions=predictions_clean, references=[r[0] for r in references])
            metrics["rougeL"] = float(rouge_out.get("rougeL", 0.0))
            metrics["meteor"] = float(meteor.compute(predictions=predictions_clean, references=[r[0] for r in references])["meteor"])
        except Exception:
            pass
        try:
            cider = evaluate.load("cider")
            metrics["cider"] = float(cider.compute(predictions=predictions_clean, references=references)["cider"])
        except Exception:
            metrics["cider"] = float("nan")
    return metrics


class TextClassificationDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer, max_length: int = 128):
        self.enc = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class Vocab:
    def __init__(self, texts: Sequence[str], min_freq: int = 1):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        counts: Dict[str, int] = {}
        for text in texts:
            for tok in str(text).split():
                counts[tok] = counts.get(tok, 0) + 1
        self.itos = [self.pad_token, self.unk_token] + [t for t, c in counts.items() if c >= min_freq]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, text: str, max_length: int) -> List[int]:
        ids = [self.stoi.get(tok, 1) for tok in str(text).split()][:max_length]
        ids += [0] * max(0, max_length - len(ids))
        return ids

    @property
    def size(self) -> int:
        return len(self.itos)


class TokenDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], vocab: Vocab, max_length: int = 64):
        self.ids = [vocab.encode(t, max_length) for t in texts]
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.ids[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        _, (h, _) = self.rnn(x)
        return self.fc(self.dropout(h[-1]))


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        _, h = self.rnn(x)
        return self.fc(self.dropout(h[-1]))


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, num_classes: int, filters: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, filters, k) for k in [3, 4, 5]])
        self.fc = nn.Linear(filters * 3, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids).transpose(1, 2)
        feats = [torch.max(torch.relu(conv(x)), dim=2).values for conv in self.convs]
        return self.fc(self.dropout(torch.cat(feats, dim=1)))


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, num_classes: int, max_len: int = 64, nhead: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, emb_dim)
        layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.fc = nn.Linear(emb_dim, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pos_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.emb(input_ids) + self.pos(pos_ids)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.fc(self.dropout(pooled))


@dataclass
class TorchTrainConfig:
    epochs: int = 3
    lr: float = 1e-3
    batch_size: int = 32
    max_length: int = 64
    emb_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.2



def save_sentiment_outputs(
    outdir: Path,
    model_name: str,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    confidences: Optional[Sequence[float]] = None,
) -> None:
    maybe_mkdir(outdir)
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    if confidences is not None:
        df["confidence"] = confidences
    df.to_csv(outdir / f"{model_name}_predictions.csv", index=False)
    np.savetxt(outdir / f"{model_name}_confusion_matrix.csv", confusion_matrix(y_true, y_pred), delimiter=",", fmt="%d")
    with open(outdir / f"{model_name}_classification_report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=SENTIMENT_LABELS, zero_division=0))



def train_torch_text_model(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    config: TorchTrainConfig,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[nn.Module, Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    best_f1 = -1.0
    best_state = None

    for _ in range(config.epochs):
        model.train()
        for batch in train_dl:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

        model.eval()
        val_true: List[int] = []
        val_pred: List[int] = []
        with torch.no_grad():
            for batch in val_dl:
                x = batch["input_ids"].to(device)
                y = batch["labels"].to(device)
                probs = torch.softmax(model(x), dim=1)
                preds = torch.argmax(probs, dim=1)
                val_true.extend(y.cpu().tolist())
                val_pred.extend(preds.cpu().tolist())
        val_f1 = f1_score(val_true, val_pred, average="macro")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    confs: List[float] = []
    with torch.no_grad():
        for batch in test_dl:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            probs = torch.softmax(model(x), dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            confs.extend(torch.max(probs, dim=1).values.cpu().tolist())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    conf_arr = np.array(confs)
    metrics = compute_sentiment_metrics(y_true_arr, y_pred_arr)
    return model, metrics, y_true_arr, y_pred_arr, conf_arr


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
        )
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss



def compute_hf_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return compute_sentiment_metrics(labels, preds)


class VCSC:
    """Confidence-sensitive visual sentiment calibration."""

    def __init__(self, vit_name: str = "google/vit-base-patch16-224-in21k") -> None:
        self.device = get_device()
        self.processor = AutoImageProcessor.from_pretrained(vit_name)
        self.vit = AutoModel.from_pretrained(vit_name).to(self.device)
        self.regressor = Ridge(alpha=1.0)
        self.is_fit = False

    @torch.no_grad()
    def encode_images(self, image_paths: Sequence[str], batch_size: int = 16) -> np.ndarray:
        feats: List[np.ndarray] = []
        self.vit.eval()
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.vit(**inputs)
            pooled = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            feats.append(pooled)
        return np.concatenate(feats, axis=0)

    def fit(self, image_paths: Sequence[str], target_polarity: Sequence[float]) -> None:
        X = self.encode_images(image_paths)
        y = np.asarray(target_polarity, dtype=np.float32)
        self.regressor.fit(X, y)
        self.is_fit = True

    def predict_visual_affect(self, image_paths: Sequence[str]) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError("VCSC must be fit before calling predict_visual_affect().")
        X = self.encode_images(image_paths)
        return np.tanh(self.regressor.predict(X))

    def calibrate(
        self,
        original_polarity: Sequence[float],
        confidences: Sequence[float],
        image_paths: Sequence[str],
        delta_c: float = 0.60,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = self.predict_visual_affect(image_paths)
        p = np.asarray(original_polarity, dtype=np.float32)
        c = np.asarray(confidences, dtype=np.float32)
        gamma = (c < float(delta_c)).astype(np.float32) * np.abs(a)
        adjusted = (1.0 - gamma) * p + gamma * a
        return adjusted, a, gamma



def run_classical_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    outdir: Path,
) -> Dict[str, Dict[str, float]]:
    vect = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train = vect.fit_transform(train_df["cleaned_caption"])
    X_test = vect.transform(test_df["cleaned_caption"])
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    models = {
        "SVM": LinearSVC(C=1.0, tol=1e-3),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=20, criterion="gini", random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=20, criterion="entropy", min_samples_split=2, random_state=42),
        "NaiveBayes": MultinomialNB(alpha=1.0),
    }

    results: Dict[str, Dict[str, float]] = {}
    for name, clf in models.items():
        start = time.time()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        metrics = compute_sentiment_metrics(y_test, preds)
        metrics["train_eval_time_sec"] = float(time.time() - start)
        results[name] = metrics
        save_sentiment_outputs(outdir, name, y_test, preds)
    return results



def run_torch_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
    outdir: Path,
    epochs: int,
) -> Dict[str, Dict[str, float]]:
    config = TorchTrainConfig(epochs=epochs)
    vocab = Vocab(train_df["cleaned_caption"].tolist(), min_freq=1)
    train_ds = TokenDataset(train_df["cleaned_caption"], train_df["label"], vocab, config.max_length)
    val_ds = TokenDataset(val_df["cleaned_caption"], val_df["label"], vocab, config.max_length)
    test_ds = TokenDataset(test_df["cleaned_caption"], test_df["label"], vocab, config.max_length)
    num_classes = len(SENTIMENT_LABELS)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=train_df["label"].to_numpy(),
    )
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

    models = {
        "LSTM": LSTMClassifier(vocab.size, config.emb_dim, config.hidden_dim, num_classes, config.dropout),
        "GRU": GRUClassifier(vocab.size, config.emb_dim, config.hidden_dim, num_classes, config.dropout),
        "CNN": CNNClassifier(vocab.size, config.emb_dim, num_classes),
        "Transformer": TransformerClassifier(vocab.size, config.emb_dim, num_classes, max_len=config.max_length),
    }

    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        start = time.time()
        _, metrics, y_true, y_pred, conf = train_torch_text_model(
            model,
            train_ds,
            val_ds,
            test_ds,
            config,
            device,
            class_weights=class_weights_t,
        )
        metrics["train_eval_time_sec"] = float(time.time() - start)
        results[name] = metrics
        save_sentiment_outputs(outdir, name, y_true, y_pred, conf)
    return results



def run_bert_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    epochs: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(SENTIMENT_LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    train_ds = TextClassificationDataset(train_df["cleaned_caption"], train_df["label"], tokenizer)
    val_ds = TextClassificationDataset(val_df["cleaned_caption"], val_df["label"], tokenizer)
    test_ds = TextClassificationDataset(test_df["cleaned_caption"], test_df["label"], tokenizer)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(SENTIMENT_LABELS)),
        y=train_df["label"].to_numpy(),
    )
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=[],
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_hf_metrics,
        class_weights=class_weights_t,
    )
    start = time.time()
    trainer.train()
    preds = trainer.predict(test_ds)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)
    probs = torch.softmax(torch.tensor(preds.predictions), dim=1).numpy()
    conf = probs.max(axis=1)
    metrics = compute_sentiment_metrics(y_true, y_pred)
    metrics["train_eval_time_sec"] = float(time.time() - start)
    save_sentiment_outputs(output_dir, "BERT", y_true, y_pred, conf)
    return metrics, y_true, y_pred, conf



def run_vcsc(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_pred_ids: np.ndarray,
    test_confidences: np.ndarray,
    delta_c: float,
    outdir: Path,
) -> Dict[str, float]:
    vcsc = VCSC()
    val_targets = [class_to_signed_polarity(x) for x in val_df["label"].tolist()]
    vcsc.fit(val_df["image"].tolist(), val_targets)

    original_p = np.array([class_to_signed_polarity(x) for x in test_pred_ids], dtype=np.float32)
    adjusted, visual_affect, gamma = vcsc.calibrate(
        original_p,
        test_confidences,
        test_df["image"].tolist(),
        delta_c=delta_c,
    )
    calibrated_ids = np.array([signed_polarity_to_class_id(v) for v in adjusted])
    metrics = compute_sentiment_metrics(test_df["label"].to_numpy(), calibrated_ids)

    pd.DataFrame({
        "image": test_df["image"],
        "caption": test_df["caption"],
        "y_true": test_df["label"],
        "y_pred_before": test_pred_ids,
        "confidence": test_confidences,
        "original_polarity": original_p,
        "visual_affect": visual_affect,
        "gamma": gamma,
        "adjusted_polarity": adjusted,
        "y_pred_after": calibrated_ids,
    }).to_csv(outdir / "vcsc_predictions.csv", index=False)
    np.savetxt(outdir / "vcsc_confusion_matrix.csv", confusion_matrix(test_df["label"], calibrated_ids), delimiter=",", fmt="%d")
    with open(outdir / "vcsc_classification_report.txt", "w") as f:
        f.write(classification_report(test_df["label"], calibrated_ids, target_names=SENTIMENT_LABELS, zero_division=0))
    return metrics



def run_captioning(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    epochs: int,
) -> Dict[str, float]:
    model, image_processor, tokenizer = build_vision_gpt2_model()
    train_ds = CaptionDataset(train_df[["image", "caption"]], image_processor, tokenizer)
    val_ds = CaptionDataset(val_df[["image", "caption"]], image_processor, tokenizer)
    test_ds = CaptionDataset(test_df[["image", "caption"]], image_processor, tokenizer)

    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        logging_steps=50,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        processing_class=tokenizer,
        compute_metrics=lambda eval_preds: compute_caption_metrics(eval_preds, tokenizer),
    )
    start = time.time()
    trainer.train()
    metrics = trainer.predict(test_ds, max_length=64).metrics
    out = {k.replace("test_", ""): float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    out["train_eval_time_sec"] = float(time.time() - start)
    return out



def select_best_model(results: Dict[str, Dict[str, float]], key: str = "f1_macro") -> Optional[str]:
    if not results:
        return None
    return max(results.items(), key=lambda kv: kv[1].get(key, float("-inf")))[0]



def run_single_seed(args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    seed_everything(seed)
    ensure_nltk()
    device = get_device()

    base_outdir = Path(args.output_dir) / f"seed_{seed}"
    maybe_mkdir(base_outdir)

    df = load_flickr30k(args.captions_file, args.images_dir)
    if args.max_samples and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=seed).reset_index(drop=True)
    df = add_weak_labels(df)

    train_df, val_df, test_df = stratified_splits(
        df,
        seed=seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    if args.export_splits:
        train_df.to_csv(base_outdir / "train_split.csv", index=False)
        val_df.to_csv(base_outdir / "val_split.csv", index=False)
        test_df.to_csv(base_outdir / "test_split.csv", index=False)

    summary: Dict[str, Any] = {
        "seed": seed,
        "dataset": {
            "n_total": int(len(df)),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
        },
        "label_distribution": {
            split: {k: int(v) for k, v in frame["sentiment"].value_counts().sort_index().to_dict().items()}
            for split, frame in [("train", train_df), ("val", val_df), ("test", test_df)]
        },
    }

    classical_results: Dict[str, Dict[str, float]] = {}
    torch_results: Dict[str, Dict[str, float]] = {}
    bert_metrics = None
    bert_y_pred = None
    bert_conf = None

    if args.run_classical or args.run_all:
        classical_results = run_classical_baselines(train_df, test_df, base_outdir)
        summary["classical"] = classical_results

    if args.run_torch_baselines or args.run_all:
        torch_results = run_torch_baselines(train_df, val_df, test_df, device, base_outdir, epochs=args.sentiment_epochs)
        summary["torch_baselines"] = torch_results

    if args.run_bert or args.run_all:
        bert_metrics, y_true, bert_y_pred, bert_conf = run_bert_classifier(
            train_df,
            val_df,
            test_df,
            base_outdir / "bert",
            epochs=args.sentiment_epochs,
        )
        summary["bert"] = bert_metrics

    if args.run_vcsc or args.run_all:
        if bert_y_pred is None or bert_conf is None:
            raise ValueError("VCSC requires BERT predictions. Enable --run-bert or --run-all.")
        vcsc_metrics = run_vcsc(
            val_df,
            test_df,
            bert_y_pred,
            bert_conf,
            delta_c=args.vcsc_delta_c,
            outdir=base_outdir,
        )
        summary["vcsc"] = vcsc_metrics

    if args.run_captioning or args.run_all:
        caption_metrics = run_captioning(train_df, val_df, test_df, base_outdir / "vision_gpt", epochs=args.caption_epochs)
        summary["captioning"] = caption_metrics

    candidate_results = {}
    candidate_results.update(classical_results)
    candidate_results.update(torch_results)
    if bert_metrics is not None:
        candidate_results["BERT"] = bert_metrics
    best_model = select_best_model(candidate_results, key="f1_macro")
    summary["best_sentiment_model_by_macro_f1"] = best_model

    with open(base_outdir / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary



def aggregate_multi_seed(seed_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    aggregated: Dict[str, Any] = {"num_seeds": len(seed_summaries)}
    model_metrics: Dict[str, Dict[str, List[float]]] = {}

    for summary in seed_summaries:
        for section in ["classical", "torch_baselines"]:
            for model_name, metrics in summary.get(section, {}).items():
                model_metrics.setdefault(model_name, {})
                for metric_name, value in metrics.items():
                    model_metrics[model_name].setdefault(metric_name, []).append(float(value))
        for model_name in ["bert", "vcsc", "captioning"]:
            if model_name in summary:
                key_name = model_name.upper() if model_name == "bert" else model_name
                model_metrics.setdefault(key_name, {})
                for metric_name, value in summary[model_name].items():
                    model_metrics[key_name].setdefault(metric_name, []).append(float(value))

    aggregated["models"] = {}
    for model_name, metrics in model_metrics.items():
        aggregated["models"][model_name] = {}
        for metric_name, values in metrics.items():
            arr = np.asarray(values, dtype=np.float64)
            aggregated["models"][model_name][metric_name] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
    return aggregated



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refined unified manuscript implementation")
    parser.add_argument("--captions-file", type=str, default=None, help="Path to Flickr30k captions file")
    parser.add_argument("--images-dir", type=str, default=None, help="Path to Flickr30k image directory")
    parser.add_argument("--data-root", type=str, default=None, help="Optional Flickr30k dataset root for auto-discovery")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-seed", type=int, nargs="*", default=None, help="Optional list of seeds for repeated runs")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional subsample size for faster experiments")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--sentiment-epochs", type=int, default=3)
    parser.add_argument("--caption-epochs", type=int, default=3)
    parser.add_argument("--vcsc-delta-c", type=float, default=0.60)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--run-captioning", action="store_true")
    parser.add_argument("--run-classical", action="store_true")
    parser.add_argument("--run-torch-baselines", action="store_true")
    parser.add_argument("--run-bert", action="store_true")
    parser.add_argument("--run-vcsc", action="store_true")
    parser.add_argument("--export-splits", action="store_true")
    return parser



def main(args: argparse.Namespace) -> None:
    seeds = args.multi_seed if args.multi_seed else [args.seed]
    all_summaries: List[Dict[str, Any]] = []

    for seed in seeds:
        print(f"Running seed {seed}...")
        summary = run_single_seed(args, seed)
        all_summaries.append(summary)

    if len(all_summaries) > 1:
        aggregated = aggregate_multi_seed(all_summaries)
        with open(Path(args.output_dir) / "multi_seed_summary.json", "w") as f:
            json.dump(aggregated, f, indent=2)
        print(json.dumps(aggregated, indent=2))
    else:
        print(json.dumps(all_summaries[0], indent=2))


if __name__ == "__main__":
    main(build_parser().parse_args())
