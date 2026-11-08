import torch.nn as nn
from transformers import AutoTokenizer, BertModel


class HFEncoder(nn.Module):
    """
    Temporary class to wrap a HuggingFace encoder model. 
    This is a simple wrapper that takes care of tokenization and pooling.
    """
    def __init__(self, name, **_):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained(name)

    def forward(self, texts):
        b = self.tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        b = {k: v.to(next(self.parameters()).device) for k, v in b.items()}
        h = self.model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1)
        return (h * m).sum(1) / m.sum(1)                   # (B, 384) tensor
