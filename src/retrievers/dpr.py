import numpy as np
import torch
from numpy.typing import NDArray
from src.retrievers.base_retriever import Retriever
from src.retrievers.models.hf_encoder import HFEncoder
import os
import pickle
from src.data_utils.dataset import DocumentDataset, batched

class DPR(Retriever):
    def __init__(self, documents: list[dict], index_path: str, cfg: dict) -> None:
        self.cfg = cfg
        self.index_path = index_path
        super().__init__(documents, self.index_path)
        self.__load_models()

    def __load_models(self):
        for encoder_name in ('query_encoder', 'document_encoder'):
            encoder_cfg = getattr(self.cfg.retriever, encoder_name)
            encoder = HFEncoder(**encoder_cfg).to(self.cfg.device)
            weights_path = self.__weights_path(encoder_name)
            if weights_path and os.path.exists(weights_path):
                encoder.load_state_dict(torch.load(weights_path, map_location=self.cfg.device))
            setattr(self, encoder_name, encoder)

    def __weights_path(self, encoder_name: str) -> str | None:
        weights_path = getattr(self.cfg.retriever, encoder_name).get('weights_path', None)
        return weights_path.format(dataset=self.cfg.dataset.name) if weights_path else None

    def SaveEncoders(self):
        """Save both encoders' weights, so a later run can reuse them instead of retraining."""
        for encoder_name in ('query_encoder', 'document_encoder'):
            weights_path = self.__weights_path(encoder_name)
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            torch.save(getattr(self, encoder_name).state_dict(), weights_path)

    def BuildIndex(self) -> DocumentDataset:
        """
        @param dataset: The dataset for which an index containing embeddings should be built and saved.
        """
        # embed the index's own documents, so embeddings and documents share one order by construction
        self.index.embeddings = self.__EmbedDocuments([d.GetText() for d in self.index.GetDocuments()])
        # save the index to file
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, 'wb') as file:
            pickle.dump(self.index, file)
        return self.index

    @torch.no_grad()
    def __EmbedDocuments(self, document_texts: list[str]) -> NDArray[np.float32]:
        document_embeddings = [self.document_encoder(batch).cpu() for batch in batched(document_texts, self.cfg.batch_size)]
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
        query_embeddings = [self.query_encoder(batch).cpu() for batch in batched(queries, self.cfg.batch_size)]
        query_embeddings = torch.cat(query_embeddings).numpy()  # numpy, to match the stored document embeddings
        # compute similarity (dot product)
        scores = query_embeddings @ document_embeddings.T
        return scores
