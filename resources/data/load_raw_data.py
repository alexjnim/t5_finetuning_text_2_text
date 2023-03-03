import os

import pandas as pd
from sklearn.model_selection import train_test_split


def process_data() -> None:
    if os.path.isfile("./data/splits/train.csv"):
        return
    else:
        data = pd.read_csv("./data/original/news_summary.csv", encoding="ISO-8859-1")

        text_data = data[["text", "ctext"]]
        text_data.rename(columns={"text": "summary", "ctext": "text"}, inplace=True)
        text_data["text"] = "Summarise: " + text_data["text"]

        train, test = train_test_split(
            text_data, shuffle=True, random_state=42, test_size=0.2
        )
        train, val = train_test_split(
            train, shuffle=True, random_state=42, test_size=0.1
        )

        train.to_csv("./data/splits/train.csv", index=False)
        val.to_csv("./data/splits/val.csv", index=False)
        test.to_csv("./data/splits/test.csv", index=False)
        return


def load_training_data() -> pd.DataFrame:
    data = pd.read_csv("./data/splits/train.csv", encoding="ISO-8859-1")
    return data


def load_validation_data() -> pd.DataFrame:
    data = pd.read_csv("./data/splits/val.csv", encoding="ISO-8859-1")
    return data


def load_test_data() -> pd.DataFrame:
    data = pd.read_csv("./data/splits/test.csv", encoding="ISO-8859-1")
    return data
