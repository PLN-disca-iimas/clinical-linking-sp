# Clinical Entity Linking — Spanish NER → SNOMED CT

> A production-grade NLP pipeline that links Spanish clinical named entities to standardized **SNOMED CT** concepts using neural machine translation, multilingual embeddings, and approximate nearest-neighbour search.

---

## Table of Contents

- [Overview](#overview)
- [Presentation](#presentation)
- [Pipeline Architecture](#pipeline-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Models](#models)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Clinical free-text in Spanish often contains medical entities (diagnoses, symptoms, procedures) that need to be normalized to a controlled vocabulary for interoperability, research, and billing. This notebook implements an end-to-end **entity linking** pipeline evaluated on the [DisTEMIST](https://temu.bsc.es/distemist/) shared task dataset.

The pipeline resolves each Spanish mention to its most likely **SNOMED CT concept ID** without any task-specific fine-tuning, relying entirely on:

1. A curated medical translation dictionary.
2. Neural machine translation as a fallback.
3. Semantic similarity via multilingual sentence embeddings.
4. Efficient vector search (FAISS).

---

## Presentation

A slide deck summarising the motivation, methodology, and results of this project is available directly in this repository:

**[Vinculacion de Entidades Medicas (PDF)](Presentation%20-%20Vinculacion%20de%20Entidades%20Medicas.pdf)**

It covers the problem statement, the pipeline design decisions, and a discussion of the evaluation results — recommended as a starting point before diving into the notebook.

---

## Pipeline Architecture

```
Spanish NER spans
        │
        ▼
┌───────────────────────┐
│  Medical Dictionary   │  ← Rule-based, high-precision translations
│  (MEDICAL_DICT)       │
└──────────┬────────────┘
           │ unmapped spans
           ▼
┌───────────────────────┐
│  Neural MT            │  ← Helsinki-NLP/opus-mt-es-en (MarianMT)
│  (ES → EN)            │    + post-processing corrections
└──────────┬────────────┘
           │ English mentions
           ▼
┌───────────────────────┐
│  Multilingual         │  ← intfloat/multilingual-e5-large
│  Embeddings           │    query prefix: "query:"
└──────────┬────────────┘
           │ query vectors
           ▼
┌───────────────────────┐
│  FAISS Index          │  ← IndexFlatIP (Inner Product / cosine)
│  (SNOMED CT KB)       │    passage prefix: "passage:"
└──────────┬────────────┘
           │ top-K candidates
           ▼
     SNOMED CT Code
```

---

## Requirements

| Dependency | Version |
|---|---|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.0 |
| Transformers (HuggingFace) | ≥ 4.38 |
| FAISS | `faiss-cpu` or `faiss-gpu` |
| pandas | ≥ 2.0 |
| numpy | ≥ 1.24 |
| tqdm | ≥ 4.0 |

> **GPU note:** The notebook auto-detects CUDA. GPU is strongly recommended for the embedding and translation steps. CPU execution is supported but significantly slower.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Install dependencies
pip install transformers accelerate torch faiss-cpu numpy pandas tqdm
```

For GPU-accelerated FAISS:

```bash
pip install faiss-gpu
```

---

## Data

> **The SNOMED CT snapshot is NOT included in this repository** — it is a licensed file that cannot be redistributed. The DisTEMIST `.tsv` file is included under `data/`. See the instructions below for obtaining the SNOMED snapshot.

### Input — NER entities (included ✅)

```
data/distemist_subtrack2_training1_linking.tsv
```

A TSV file with at minimum the following columns:

| Column | Description |
|---|---|
| `span` | Raw Spanish clinical mention (e.g., `"fractura conminuta"`) |
| `code` | Gold-standard SNOMED CT concept ID |

This file is part of the **DisTEMIST** corpus. You can also download it from the [official shared task page](https://temu.bsc.es/distemist/).

### Input — SNOMED CT snapshot (excluded  — licence required)

```
sct2_Description_Snapshot-en_INT_20260101.txt
```

The official SNOMED CT International Edition description snapshot (tab-separated, ~226 MB). This file is **excluded from the repository** for two reasons: it exceeds GitHub's 100 MB file limit, and it is proprietary — redistribution is not permitted under the SNOMED CT licence.

**How to obtain it:**
1. Register for a free licence at [SNOMED International](https://www.snomed.org/get-snomed).
2. Download the International Edition release package.
3. Locate `sct2_Description_Snapshot-en_INT_<release_date>.txt` inside the archive.
4. Place it in `/content/` (Colab) or update the `path_snomed` variable in the notebook.


---

## Usage

### Google Colab (recommended)

1. Open `Clinical_coding.ipynb` in Google Colab.
2. Upload `distemist_subtrack2_training1_linking.tsv` and `sct2_Description_Snapshot-en_INT_20260101.txt` to `/content/`.
3. Run all cells in order.

### Local execution

Update the path variables at the top of the data-loading cells:

```python
path_train  = "/path/to/distemist_subtrack2_training1_linking.tsv"
path_snomed = "/path/to/sct2_Description_Snapshot-en_INT_20260101.txt"
```

Then run the notebook cell by cell or via:

```bash
jupyter nbconvert --to notebook --execute Clinical_coding.ipynb
```

---

## Models

| Model | Role | Source |
|---|---|---|
| `Helsinki-NLP/opus-mt-es-en` | Spanish → English translation | [HuggingFace Hub](https://huggingface.co/Helsinki-NLP/opus-mt-es-en) |
| `intfloat/multilingual-e5-large` | Sentence embeddings (query & passage) | [HuggingFace Hub](https://huggingface.co/intfloat/multilingual-e5-large) |

Both models are downloaded automatically on first run.

---

## Evaluation

The pipeline is evaluated using standard **recall at K** metrics on the DisTEMIST linking subset:

| Metric | Description |
|---|---|
| **Accuracy / Recall@1** | True SNOMED code is the top-ranked prediction |
| **Recall@5** | True code appears within the top 5 candidates |
| **Recall@10** | True code appears within the top 10 candidates |

Results are printed after the FAISS search step. A full prediction DataFrame (`df_preds`) is also produced with per-entity breakdown including `match`, `in_top5`, and `in_top10` flags for downstream error analysis.

---

## Project Structure

```
.
├── Clinical_coding.ipynb                              # Main pipeline notebook
├── Presentation - Vinculacion de Entidades Medicas.pdf  # Project slide deck
├── README.md                                          # This file
├── .gitignore                                         # Excludes the SNOMED snapshot
└── data/
    └── distemist_subtrack2_training1_linking.tsv      # DisTEMIST NER entities (included)
    # sct2_Description_Snapshot-en_INT_*.txt           # ← NOT included (licence + size)
```

---

## Known Limitations

- **Dictionary coverage:** `MEDICAL_DICT` covers a curated set of common terms. Rare or domain-specific expressions fall back to neural MT, which may introduce translation noise.
- **SNOMED subsetting:** To keep the FAISS index tractable, non-target concepts are stratified-sampled at 3,000 per semantic tag. This improves speed but may reduce recall for concepts in underrepresented tags.
- **No fine-tuning:** The embedding model is used zero-shot. Task-specific fine-tuning on DisTEMIST data is expected to improve Recall@1 substantially.
- **Colab paths:** Cell 16–17 contain Colab-specific code (`drive.mount`, `git clone`). These cells should be skipped or adapted for local execution.

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. When contributing, please:

- Follow [PEP 8](https://pep8.org/) style guidelines.
- Add or update docstrings for any new functions.
- Ensure the notebook runs end-to-end before submitting.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

> **Third-party licences:** SNOMED CT is licenced separately by [SNOMED International](https://www.snomed.org/get-snomed). Ensure compliance with your local affiliate's terms before using the snapshot file.
