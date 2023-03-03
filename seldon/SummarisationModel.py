import torch

from resources.inference.model import T5SUM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class T5SUM:
    def __init__(self):
        pass

    def load(self):
        self.t5_model = T5SUM(
            save_model_path="./model/30_epochs_03_03_2023_t5-small_model",
            max_token_len=50,
        )
        self.g2p_model.to(device)
        pass

    def predict_raw(self, entry: dict[str, str]) -> dict[str, str]:
        text = entry["text"]
        summary = self.t5_model.generate_summary(text)

        entry["summary"] = summary
        return entry

    def init_metadata(self):
        meta = {
            "inputs": [{"messagetype": "jsonData", "schema": {"text": "example"}}],
            "outputs": [
                {
                    "messagetype": "jsonData",
                    "schema": {"text": "example", "summary": "example"},
                }
            ],
        }
        return meta
