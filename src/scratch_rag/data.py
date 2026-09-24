from sklearn.model_selection import train_test_split
from pathlib import Path
import json

class Data:
    def __init__(self, filepath: Path, train_temp_split: float, test_val_split: float):
        self.filepath = filepath
        self.train_temp_split = train_temp_split
        self.test_val_split = test_val_split

    def write_splits(self, writepath: Path):
        with open(self.filepath, "r") as file:
            content = file.read()

        splits = content.split('\n')

        json_splits = []
        for split in splits:
            if len(split) > 0:
                json_splits.append(json.loads(split))

        X = json_splits

        X_train, X_temp = train_test_split(X, test_size=self.train_temp_split, random_state=0)

        X_val, X_test = train_test_split(X_temp, test_size=self.test_val_split, random_state=0)

        for dataset, split in [(X_train, 'train.jsonl'), (X_test, 'test.jsonl'), (X_val, 'val.jsonl')]:
            with open(writepath.joinpath(split), "w") as file:
                out = self._write(dataset)
                file.write(out)

    def _write(self, json_list: list) -> str:
        output = ""
        for json_ob in json_list:
            if json_ob is None:
                continue
            output += json.dumps(json_ob)+"\n"

        return output