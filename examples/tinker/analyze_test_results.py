import os

import pandas as pd

for file in os.listdir("test-results"):
    if file.endswith(".jsonl") and file.startswith("twenty_questions_"):
        print(file)
        df = pd.read_json(os.path.join("test-results", file), lines=True)
        # df = df[df["category"].isin(["animal", "character", "instrument", "nature", "object", "person", "place"])]
        print(df["correct"].mean())
        print(df.groupby("category")["correct"].mean())
        print(df.groupby("category")["correct"].count())
