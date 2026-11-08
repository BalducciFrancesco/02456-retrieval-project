import torch
import torch.nn as nn
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

BATCH_SIZE = 32
LR = 1e-5

class Trainer:
    def __init__(self, split: Dataset):
        # split the positives into multiple rows: (q, [d1, d2]) -> [(q, d1, id), (q, d2, id)]
        # query_id is "q12" -> 12, so it can be encoded as a tensor
        pairs = [(q, d, int(qid[1:]))
                 for q, docs, qid in zip(split["question"], split["text"], split["query_id"])
                 for d in docs]
        self.loader = DataLoader(pairs, batch_size=BATCH_SIZE, shuffle=True)

    def train(self, query_encoder: nn.Module, document_encoder: nn.Module):
        model = nn.ModuleList([query_encoder, document_encoder])
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

        # Forward pass
        for queries, documents, ids in tqdm(self.loader, desc="Training batch"):
            queries_embeddings = query_encoder(queries)        # (B, embedding_dim)
            documents_embeddings = document_encoder(documents) # (B, embedding_dim)
            ids = ids.to(queries_embeddings.device)

            # Compute loss and backpropagate
            optimizer.zero_grad()
            loss = self.compute_loss(queries_embeddings, documents_embeddings, ids)
            loss.backward()  
            optimizer.step()

    def compute_loss(self, queries_embeddings: torch.Tensor, documents_embeddings: torch.Tensor, ids: torch.Tensor):
        B = len(ids)
        labels = torch.arange(B, device=ids.device)  # Correct document for each query is at the same index
        scores = queries_embeddings @ documents_embeddings.T    # (B, B) where the diagonal contains the correct pairs

        # Mask out off-diagonal correct documents (when more than one positive exists)
        mask = ids[:, None] == ids[None, :]     # True if positive
        mask.fill_diagonal_(False)              # False if on the diagonal (true positive)
        scores = scores.masked_fill(mask, float("-inf"))    # mask all True's

        return F.cross_entropy(scores, labels)