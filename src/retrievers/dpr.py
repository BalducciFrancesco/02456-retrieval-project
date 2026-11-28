import numpy as np
import torch
from numpy.typing import NDArray
from src.retrievers.base_retriever import Retriever
from src.retrievers.models.encoder import TransformerEncoder
from transformers import AutoTokenizer
import os
import pickle
from src.data_utils.dataset import DocumentDataset, batched

class DPR(Retriever):
    def __init__(self, documents: list[dict], index_path: str, cfg: dict, pooling_type: str, weights_path: str) -> None:
        """weights_path: where each encoder's weights are saved, with an {encoder} placeholder (query / document)."""
        self.cfg = cfg
        self.pooling_type = pooling_type
        self.weights_path = weights_path
        self.index_path = index_path
        super().__init__(documents, self.index_path)
        self.__load_models()

    def __load_models(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.retriever.tokenizer)
        self.weights_loaded = True  # stays True only if every encoder found its saved weights
        for encoder_name in ('query_encoder', 'document_encoder'):
            c = getattr(self.cfg.retriever, encoder_name)
            encoder = TransformerEncoder(
                n_layers=c.n_layers,
                n_heads=c.n_heads,
                emb_dim=c.emb_dim,
                feed_forward_dim=c.feed_forward_dim,
                vocab_size=len(self.tokenizer),
                pool_type=self.pooling_type,
            ).to(self.cfg.device)
            
            weights_path = self.__weights_path(encoder_name)
            if os.path.exists(weights_path):
                encoder.load_state_dict(torch.load(weights_path, map_location=self.cfg.device))
            else:
                self.weights_loaded = False
            setattr(self, encoder_name, encoder)

    def Embed(self, encoder, texts: list[str]) -> torch.Tensor:
        """Tokenize texts and run them through an encoder: (len(texts), emb_dim)."""
        b = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.cfg.device)
        return encoder(b["input_ids"], b["attention_mask"])

    def __weights_path(self, encoder_name: str) -> str:
        return self.weights_path.format(encoder=encoder_name.removesuffix("_encoder"))

    def SaveEncoders(self):
        """Save both encoders' weights, so a later run can reuse them instead of retraining."""
        for encoder_name in ('query_encoder', 'document_encoder'):
            weights_path = self.__weights_path(encoder_name)
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            torch.save(getattr(self, encoder_name).state_dict(), weights_path)

    def BuildIndex(self) -> DocumentDataset:
        """Embed every document and save the index, unless a saved one was already loaded."""
        if self.index.GetEmbeddings() is not None:
            print("DPR loaded saved index, skipping index build...")
            return self.index
        print("DPR building index now...")
        # embed the index's own documents, so embeddings and documents share one order by construction
        self.index.embeddings = self.__EmbedDocuments([d.GetText() for d in self.index.GetDocuments()])
        # save the index to file
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, 'wb') as file:
            pickle.dump(self.index, file)
        return self.index

    @torch.no_grad()
    def __EmbedDocuments(self, document_texts: list[str]) -> NDArray[np.float32]:
        document_embeddings = [self.Embed(self.document_encoder, batch).cpu() for batch in batched(document_texts, self.cfg.batch_size)]
        return torch.cat(document_embeddings).numpy()

    @torch.no_grad()
    def CalculateScores(self, queries: list[str]) -> np.ndarray: 
        """
        Embeds and computes the similarity scores between the queries and the document embeddings (requires index).
        @param queries: The input text(s) to which relevant passages should be found.
        @return: A numpy array of shape (n_queries, n_documents) with the scores for each query-document pair.
        """
        # get indexed document embeddings
        document_embeddings = self.index.GetEmbeddings()
        if document_embeddings is None:
            raise ValueError("Document embeddings are not available. Build the index first.")
        # create query embeddings
        query_embeddings = [self.Embed(self.query_encoder, batch).cpu() for batch in batched(queries, self.cfg.batch_size)]
        query_embeddings = torch.cat(query_embeddings).numpy()  # numpy, to match the stored document embeddings
        # compute similarity (dot product)
        scores = query_embeddings @ document_embeddings.T
        return scores
