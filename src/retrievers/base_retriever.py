import os
import pickle
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from src.data_utils.dataset import DocumentDataset, batched

class Retriever(ABC):
    def __init__(self, documents: list[dict], index_path: str) -> None:
        if os.path.exists(index_path):
            self.index = self.__LoadIndex(index_path)
        else:
            self.index = self.__InitIndex(documents)

    def __InitIndex(self, documents: list[dict]):
        return DocumentDataset(documents)

    def __LoadIndex(self, index_path: str) -> DocumentDataset:
        """
        @param index_path: The path to the pre-computed index.
        """
        with open(index_path, 'rb') as file:
            index = pickle.load(file)
        return index

    def Lookup(self, queries: list[str], k: int, batch_size: int) -> list:
        """
        Retrieves the top-k relevant documents for each input query.

        @param queries: The input text(s) to which relevant passages should be found.
        @param k: The number of relevant passages to retrieve for each input query.
        @param batch_size: Queries scored at once, so the score matrix stays (batch_size, n_documents).
        @return: A list of lists of the top-k relevant documents for each input query.
        """
        documents = self.index.GetDocuments()
        results = []
        for batch in batched(queries, batch_size):
            scores = self.CalculateScores(batch)               # (batch_size, n_documents)
            top_k = np.argsort(-scores, axis=1)[:, :k]         # document indices, highest score first
            results += [[documents[i] for i in row] for row in top_k]
        return results

    @abstractmethod
    def CalculateScores(self, query: list[str]) -> NDArray[np.float32]:
        """
        Embeds and computes the similarity scores between the queries and the document embeddings (requires index).

        @param query: The input text(s) to which relevant passages should be found.
        @return: A numpy array of shape (n_queries, n_documents) with the scores for each query-document pair.
        """
        #raise NotImplementedError("Must overwrite")
        ...

