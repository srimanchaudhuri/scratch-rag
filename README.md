# scratch-rag

**A practical, well-evaluated retrieval-augmented generation system.** The retrieval stack (tokenizer, chunker, TF-IDF/BM25, word2vec embeddings, ANN index, hybrid retriever) is built with standard libraries, focused on correct system design and rigorous evaluation. The generator — a tiny GPT — is still hand-built from first principles with raw PyTorch tensors and autograd, since that's the part where understanding the internals is the actual point.

The goal isn't answer quality for its own sake. It's a correctly engineered, measurable RAG pipeline, with the one component (the generator) built and understood from the ground up rather than imported.

[![Python](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Build system](https://img.shields.io/badge/build-uv-de5fe9)](https://docs.astral.sh/uv/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#license)
[![Status](https://img.shields.io/badge/status-in%20progress-yellow)](#progress)

---

## Library policy

Most of the retrieval stack leans on standard tools — the skill being proven there is system design and evaluation rigor, not reimplementing well-trodden algorithms. The generator is the exception: it stays hand-built, since understanding transformer internals is the actual point of that stage.

| Stage | Approach |
|---|---|
| Tokenizer | Library-based (NLTK) |
| Chunking | Practical engineering |
| Sparse retrieval (TF-IDF / BM25) | scikit-learn |
| Dense embeddings (word2vec) | gensim |
| ANN / hybrid index | Undecided — see note below |
| Generator (tiny GPT) | From scratch, PyTorch tensors + autograd only |

| Allowed | Forbidden |
|---|---|
| Python standard library | LangChain, LlamaIndex (orchestration frameworks) |
| NumPy, pandas | Hugging Face (transformers/tokenizers/pretrained models) |
| scikit-learn, gensim, NLTK | Pretrained weights or external LLM APIs |
| PyTorch tensors + autograd, `nn.Parameter`, `nn.Linear`, `nn.Embedding`, `torch.optim` (generator only) | `nn.Transformer*`, `nn.MultiheadAttention`, `F.scaled_dot_product_attention` |
| `mps` device | |
| matplotlib (for plots) | |

> **Note:** FAISS is still forbidden pending a decision on Checkpoint 5 (ANN index) — not yet discussed.

The generator is still where the math gets derived and coded by hand. Everything upstream of it is about assembling the pieces correctly and proving they work with real evaluation numbers (Recall@k, MRR) rather than just "it ran."

## Architecture

```mermaid
flowchart LR
    A[Corpus<br/>crawler output] --> B[Clean + split<br/>sentences]
    B --> C[BPE tokenizer]
    C --> D[Chunker]
    D --> E[Sparse index<br/>TF-IDF / BM25]
    D --> F[Dense index<br/>word2vec + SIF]
    Q[Question] --> G[Hybrid retrieve<br/>RRF + ANN]
    E --> G
    F --> G
    G --> H[Generator<br/>extractive / tiny GPT]
    H --> I[Answer + cited chunks]
```

Everything left of the indexes runs once, offline. The query path runs from question to answer through hybrid retrieval and generation.

## Progress

Roughly 15–17 weeks at ~10 hours/week. The first demo-ready system lands at the end of Checkpoint 6.

| # | Checkpoint | Est. weeks | Deliverable | Status |
|---|---|---|---|---|
| 0 | Setup + corpus | 1 | Repo skeleton, 2–10 MB clean domain corpus | 🔄 In progress |
| 1 | Text processing + tokenization | 1.5 | Corpus tokenized with NLTK's word tokenizer, vocab + encode/decode round-trip | ⬜ Not started |
| 2 | Chunking | 1 | Fixed + sentence-aware chunkers, chunk stats | ⬜ Not started |
| 3 | Sparse retrieval + eval | 2 | TF-IDF, BM25, labeled question set, Recall@k/MRR | ⬜ Not started |
| 4 | Dense embeddings | 2 | word2vec (SGNS) + SIF sentence vectors | ⬜ Not started |
| 5 | Hybrid + ANN index | 1.5 | RRF fusion, LSH or IVF index, speed vs recall plot | ⬜ Not started |
| 6 | Extractive answering | 1 | **End-to-end RAG v1** with CLI demo | ⬜ Not started |
| 7 | Tiny transformer | 3 | GPT-style model pretrained on the corpus | ⬜ Not started |
| 8 | RAG fine-tuning | 2 | Context-conditioned generator + evaluation | ⬜ Not started |
| 9 | Polish + showcase | 1 | README, ablation tables, demo video/UI | ⬜ Not started |

## Repo layout

```
rag/
├── text/        # cleaning, sentence splitting
├── tokenizer/   # BPE
├── chunking/    # fixed + sentence-aware chunkers
├── sparse/      # TF-IDF, BM25
├── dense/       # word2vec (SGNS), SIF
├── index/       # LSH / IVF, RRF fusion
├── generate/    # extractive answering, tiny GPT
└── eval/        # Recall@k, MRR, ablations
data/            # corpus.jsonl and derived artifacts
notes/           # cpN.md — design-choice writeups per checkpoint
tests/
```

## Getting started

```bash
uv sync
uv run scratch-rag
```

## Notes

Each checkpoint has a `notes/cpN.md` explaining the design choices behind it — why a given algorithm, what tradeoffs it makes, and what the "done when" test looked like. These double as the interview-answer version of this README.

## License

MIT
