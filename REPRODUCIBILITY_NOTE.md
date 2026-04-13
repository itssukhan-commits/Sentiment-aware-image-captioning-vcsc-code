# Reproducibility Note

This repository should be interpreted as **manuscript code**, intended to:

- facilitate verification of the proposed method  
- support reproducible research practices  
- enable further development and benchmarking in multimodal sentiment-aware image captioning 

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

## Citation

If this code is used, please cite:

```bibtex
@software{Khan2026VCSC,
  author = {Khan, Sajid Ullah},
  title = {Manuscript Code: Sentiment-aware Image Captioning with Visually Consistent Sentiment Calibration (VCSC)},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.19558277}
}

