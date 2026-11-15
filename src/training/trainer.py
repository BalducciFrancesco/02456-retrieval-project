import random
import torch
import torch.nn as nn
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

class Trainer:
    def __init__(self, split: Dataset, cfg, batch_size: int):
        self.epochs = cfg.epochs
        self.lr = float(cfg.lr)  # YAML reads 1e-5 (no dot) as a string
        rows = list(zip(split["question"], split["text"]))
        self.loader = DataLoader(rows, batch_size=batch_size, shuffle=True, collate_fn=self.collate)

    @staticmethod
    def collate(batch): # applied to every batch
        """
        Turn a batch of (question, [doc1, doc2, ...]) positives into (question, doc),
        pairing every question with one randomly chosen positive.
        """
        queries = [q for q, _ in batch]
        documents = [random.choice(docs) for _, docs in batch]
        return queries, documents

    def train(self, query_encoder: nn.Module, document_encoder: nn.Module):
        model = nn.ModuleList([query_encoder, document_encoder])
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            total_loss = 0.0
            # Forward pass
            for queries, documents in tqdm(self.loader, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                queries_embeddings = query_encoder(queries)        # (B, embedding_dim)
                documents_embeddings = document_encoder(documents) # (B, embedding_dim)

                # Compute loss and backpropagate
                optimizer.zero_grad()

                # the positive doc is on the diagonal, everything else is a negative
                scores = queries_embeddings @ documents_embeddings.T    # (B, B)
                labels = torch.arange(len(scores), device=scores.device)
                loss = F.cross_entropy(scores, labels)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}: average loss {total_loss / len(self.loader):.4f}")

        model.eval()  # turn dropout off again before the encoders are used for indexing/retrieval