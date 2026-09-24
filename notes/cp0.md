# Checkpoint 0 — Setup & Corpus

## Domain: ML concepts

Chose a narrow set of machine learning concepts (supervised/unsupervised/reinforcement learning, neural network fundamentals, classic algorithms, evaluation metrics, etc.) as the corpus domain. Priority was a small, consistent vocabulary — a narrow, related topic cluster gives more repeated word co-occurrence per document than a broad or mixed-domain corpus would, which matters for anything downstream that relies on word statistics (tokenizer vocab, word2vec).

## Sourcing: hand-curated, no scraping

Originally the roadmap called for reusing an async web crawler to pull a real docs site. That was deliberately dropped for this checkpoint — building/running a crawler is its own scope of engineering work, separate from the corpus and pipeline design this checkpoint is actually about. Instead, `data/corpus.jsonl` was hand-written: 30 short documents, one per ML concept, in the `{id, title, url, text}` schema. URLs use a `scratch-rag.demo/...` namespace since this is synthetic content, not real scraped pages.

## Scale tradeoff: 30 docs instead of 2–10MB

Final corpus: **30 documents, 2,373 words, ~18.3KB** — far below the roadmap's original 2–10MB target. This was a deliberate, informed tradeoff, not an oversight: hand-curating text doesn't scale to megabytes, and the alternative (scraping or bulk-downloading a real dump) was explicitly set aside for this checkpoint.

Consequence understood and accepted: later checkpoints that depend on volume — word2vec (Checkpoint 4) needs a lot of repeated co-occurrence to produce meaningful vectors, and the tiny GPT (Checkpoints 7–8) needs enough tokens to learn anything beyond memorization — will likely show visibly weaker results than they would on a full-size corpus. That's fine here: the goal for those checkpoints is still proving the mechanics are implemented correctly, not producing high-quality embeddings or fluent generation. Output quality was never the point of this project; understanding each layer is.

## Library policy pivot

Partway through this checkpoint, the project's from-scratch constraints were revised: `scikit-learn`, `gensim`, `NLTK`, and `pandas` are now allowed for the retrieval stack (tokenizer, chunking, sparse/dense retrieval, indexing), since the skill being proven there is correct system design and evaluation rigor rather than reimplementing well-known algorithms. The generator (tiny GPT, Checkpoints 7–8) stays fully hand-built with raw PyTorch tensors and autograd — that remains the part of the project where understanding the internals is the actual goal. See the README's "Library policy" section for the full per-stage breakdown.

## Train / val / test split

Split at the document level using `sklearn.train_test_split`, applied twice: first peeling off a combined val+test portion from the full corpus, then splitting that portion in half into val and test. Result: **20 train / 5 val / 5 test** (66.7% / 16.7% / 16.7%). Splitting at the document level (not on chunks or sentences) avoids near-duplicate content leaking across splits once chunking exists in Checkpoint 2. `random_state=0` is currently a fixed literal directly on the split call, hardcoded rather than sourced from a project-wide config/seed — `config.yml` and `seed_everything()` are intentionally still unimplemented and will be wired in properly in a follow-up pass, at which point this reproducibility story becomes explicit and configurable instead of an implicit fixed value.

## Split output format: three separate JSONL files

Chose to write `data/train.jsonl`, `data/val.jsonl`, and `data/test.jsonl` as fully self-contained files, rather than keeping a single `corpus.jsonl` plus a `splits.json` id→label mapping. Reasoning: simplicity — each downstream stage can load exactly the split it needs directly, without an extra join/lookup step against the full corpus. Trade-off accepted: the corpus text is now duplicated across `corpus.jsonl` and the three split files, so if `corpus.jsonl` is ever hand-edited later, the split files would silently drift out of sync with it. Given the corpus is static and small at this stage, that risk was judged acceptable.

## Done-when check

- `data/corpus.jsonl` exists: 30 docs / 2,373 words / ~18.3KB, no HTML debris (hand-written, none to strip), no duplicate documents (distinct ML concept per entry).
- `data/{train,val,test}.jsonl` exist, valid line-delimited JSON, sized 20/5/5 documents respectively, reproducible via a fixed random seed.
