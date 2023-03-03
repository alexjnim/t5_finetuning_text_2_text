import torch

from resources.inference.model import T5G2P

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class T5SummarisationService:
    def __init__(self):
        pass

    def load(self):
        self.g2p_model = T5G2P(
            save_model_path="./model/30_epochs_13_02_2023_t5-small_model",
            max_token_len=50,
        )
        self.g2p_model.to(device)
        pass

    def predict_raw(self, entry: dict[str, str]) -> dict[str, str]:
        grapheme = entry["grapheme"]
        phoneme = self.g2p_model.generate_phonemes(grapheme)

        entry["phoneme"] = phoneme
        return entry

    def init_metadata(self):
        meta = {
            "inputs": [{"messagetype": "jsonData", "schema": {"grapheme": "example"}}],
            "outputs": [
                {
                    "messagetype": "jsonData",
                    "schema": {"grapheme": "example", "phoneme": "example"},
                }
            ],
        }
        return meta
