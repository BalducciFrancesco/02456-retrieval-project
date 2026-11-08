import numpy as np
import torch
from numpy.typing import NDArray
from src.retrievers.base_retriever import Retriever
from src.retrievers.models.hf_encoder import HFEncoder

class DPR(Retriever):
    def __init__(self, cfg, documents: list[dict]) -> None:
        super().__init__(documents, cfg.retriever.index_path.format(dataset=cfg.dataset.name))
        self.cfg = cfg
        self.__load_models()

    def __load_models(self):
        for encoder_name in ('query_encoder', 'document_encoder'):
            encoder_cfg = getattr(self.cfg.retriever, encoder_name)
            encoder = HFEncoder(**encoder_cfg).to(self.cfg.device)
            weights_path = encoder_cfg.get('weights_path', None)
            if weights_path:
                encoder.load_state_dict(torch.load(weights_path))
            setattr(self, encoder_name, encoder)

    def EmbedDocuments(self, documents: list[dict]) -> NDArray[np.float32]:
        document_texts = [t for entry in documents for t in entry['text']]  # same flattening/order as DocumentDataset
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
        scores = query_embeddings @ document_embeddings.T
        return scores