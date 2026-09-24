from scratch_rag.data import Data
from pathlib import Path

dataWriter = Data(Path("/Users/sriman/Documents/scratch-rag/data/corpus.jsonl"), 0.32, 0.5)

if __name__ == "__main__":
    dataWriter.write_splits(Path("/Users/sriman/Documents/scratch-rag/data"))