import numpy as np
import torch
from numpy.typing import NDArray
from src.retrievers.base_retriever import Retriever
from src.retrievers.models.encoder import TransformerEncoder

class DPR(Retriever):
    def __init__(self, cfg, documents: list[dict]) -> None:
        self.cfg = cfg
        self.__load_models()
        super().__init__(documents, cfg.retriever.index_path.format(dataset=cfg.dataset.name))

    def __load_models(self):
        # n_heads, n_layers, emb_dim, hidden_dim, feed_forward_dim
        self.query_encoder = TransformerEncoder(**self.cfg.model.query_encoder)
        self.document_encoder = TransformerEncoder(**self.cfg.model.document_encoder)

        query_weights_path = self.cfg.model.get('query_encoder_weights_path', None)
        if self.cfg.model.get('query_encoder_weights_path', None):
            self.query_encoder.load_state_dict(torch.load(query_weights_path))

        document_weights_path = self.cfg.model.get('document_encoder_weights_path', None)
        if self.cfg.model.get('document_encoder_weights_path', None):
            self.document_encoder.load_state_dict(torch.load(document_weights_path))

    def EmbedDocuments(self, documents: list[dict]) -> NDArray[np.float32]:
        document_texts = [doc['text'] for doc in documents]
        document_embeddings = self.document_encoder(document_texts)
        return document_embeddings

    def CalculateScores(self, queries: list[str]) -> np.ndarray: 
        """
        Embeds and computes the similarity scores between the queries and the document embeddings (requires index).
        @param queries: The input text(s) to which relevant passages should be found.
        @return: A numpy array of shape (n_queries, n_documents) with the scores for each query-document pair.
        """
        # get query and document embeddings
        query_embeddings = self.query_encoder(queries)
        document_embeddings = self.index.GetEmbeddings()
        # compute similarity (dot product)
        scores = np.dot(query_embeddings, document_embeddings.T)
        return scores