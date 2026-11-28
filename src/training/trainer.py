import random
import pandas as pd
from itertools import product
import torch
import torch.nn as nn
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from src.retrievers.dpr import DPR
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate


class Trainer:
    def __init__(self, cfg, corpus, val_queries):
        self.cfg = cfg
        self.corpus = corpus
        self.val_queries = val_queries

    def train(self):
        """Grid search on train (scored on val), then train the final DPR with the best params on train + val."""
        best = self.grid_search()
        print(f"\n{'=' * 50}\nTraining final DPR with {best}")
        return self.build_dpr(best, "final", concatenate_datasets([self.corpus.train, self.corpus.val]))

    def grid_search(self) -> dict:
        """Train a DPR per combination in cfg.retriever.grid_search; return the params with the best validation MRR."""
        gs = self.cfg.retriever.grid_search
        best_mrr, best, results = -1, None, []
        for epochs, lr, batch_size, pooling in product(gs.epochs, gs.lr, gs.batch_size, gs.pooling_type):
            params = {"epochs": epochs, "lr": lr, "batch_size": batch_size, "pooling_type": pooling}
            print(f"\n{'=' * 50}\nGrid search trial: {params}")
            model = self.build_dpr(params, "trial", self.corpus.train)
            relevancies = retrieveQueryAndGetRelevancies(model, self.val_queries, k=self.cfg.top_k, batch_size=self.cfg.batch_size)
            val_scores = Evaluate(relevancies, self.val_queries, k=self.cfg.top_k, batch_size=self.cfg.batch_size)
            print(f"Validation results: {val_scores}")
            results.append({**params, **val_scores})
            if val_scores['MRR'] > best_mrr:
                best_mrr, best = val_scores['MRR'], params

        print(pd.DataFrame(results))
        print(f"\n{'=' * 50}\nBest DPR params: {best} (val MRR {best_mrr:.4f})")
        return best

    def build_dpr(self, params: dict, kind: str, split: Dataset):
        """Build a DPR with these params and train/index it on split (reusing saved files from an earlier run)."""
        # named by its params, so a rerun reuses saved weights/index
        name = f"{self.cfg.retriever.grid_search.name}_{kind}_e{params['epochs']}_lr{params['lr']}_bs{params['batch_size']}_{params['pooling_type']}"
        dataset = self.cfg.dataset.name
        model = DPR(
            documents=self.corpus.all_documents,
            cfg=self.cfg,
            pooling_type=params["pooling_type"],
            index_path=f"index/{name}_{dataset}.pkl",
            weights_path=f"weights/{name}_{{encoder}}_{dataset}.pt",
        )
    
        if not model.weights_loaded:
            print("DPR training now...")
            self.fit(model.query_encoder, model.document_encoder, model.Embed, split, params["epochs"], params["lr"], params["batch_size"])
            model.SaveEncoders()

        model.BuildIndex()  # skips if already loaded from a saved index
        return model

    @staticmethod
    def collate(batch): # applied to every batch
        """
        Turn a batch of (question, [doc1, doc2, ...]) positives into (question, doc),
        pairing every question with one randomly chosen positive.
        """
        queries = [q for q, _ in batch]
        documents = [random.choice(docs) for _, docs in batch]
        return queries, documents

    @staticmethod
    def fit(query_encoder: nn.Module, document_encoder: nn.Module, embed, split: Dataset, epochs: int, lr: float, batch_size: int):
        """Contrastive training of both encoders. embed(encoder, texts) -> (B, emb_dim) tokenizes and encodes texts."""
        lr = float(lr)  
        rows = list(zip(split["question"], split["text"]))
        loader = DataLoader(rows, batch_size=batch_size, shuffle=True, collate_fn=Trainer.collate)

        model = nn.ModuleList([query_encoder, document_encoder])
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0.0
            # Forward pass
            for queries, documents in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                queries_embeddings = embed(query_encoder, queries)        # (B, embedding_dim)
                documents_embeddings = embed(document_encoder, documents) # (B, embedding_dim)

                # Compute loss and backpropagate
                optimizer.zero_grad()

                # the positive doc is on the diagonal, everything else is a negative
                scores = queries_embeddings @ documents_embeddings.T    # (B, B)
                labels = torch.arange(len(scores), device=scores.device)
                loss = F.cross_entropy(scores, labels)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}: average loss {total_loss / len(loader):.4f}")

        model.eval()  # turn dropout off again before the encoders are used for indexing/retrieval
