# Reproducibility Note

This repository provides a **manuscript code** of the proposed method described in:

**Sentiment-aware image captioning with visually consistent sentiment calibration**

The implementation includes:
- Vision--GPT caption generation
- five-class sentiment labeling with TextBlob weak supervision
- classical TF-IDF baselines
- LSTM, GRU, CNN, and Transformer baselines
- BERT five-class sentiment classification
- VCSC as a **post-hoc calibration** mechanism

The code implements the method and exports inspectable outputs such as predictions, metrics, confusion matrices, and reports.

As with most machine-learning pipelines, exact reported values may vary due to:
- package versions
- hardware differences
- random seed choice
- dataset export format
- training configuration

Accordingly, this repository should be interpreted as a **manuscript code**.
