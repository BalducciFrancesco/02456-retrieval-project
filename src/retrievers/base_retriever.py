import os
import pickle
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from src.data_utils.dataset import DocumentDataset

class Retriever(ABC):
    def __init__(self, documents: list[dict], index_path: str) -> None:
        if os.path.exists(index_path):
            self.index = self.__LoadIndex(index_path)
        else:
            self.index = self.__BuildIndex(index_path, documents)

    def __BuildIndex(self, index_path: str, documents: list[dict]) -> DocumentDataset:
        """
        @param dataset: The dataset for which an index containing embeddings should be built and saved.
        """
        embeddings = self.EmbedDocuments(documents)
        index = DocumentDataset(documents, embeddings)
        # save the index
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, 'wb') as file:
            pickle.dump(index, file)
        return index

    def __LoadIndex(self, index_path: str) -> DocumentDataset:
        """
        @param index_path: The path to the pre-computed index.
        """
        with open(index_path, 'rb') as file:
            index = pickle.load(file)
        return index

    def Lookup(self, queries: list[str], k: int) -> list:
        """
        @param queries: The input text(s) to which relevant passages should be found.
        @param k: The number of relevant passages to retrieve for each input query.
        """
        scores = self.CalculateScores(queries)
        ranked_documents = [[d for _, d in sorted(zip(query_scores, self.index.GetDocuments()), key=lambda pair: pair[0], reverse=True)] for query_scores in scores]
        return [ranked_document[:min(k, len(ranked_documents[0]))] for ranked_document in ranked_documents]

    @abstractmethod
    def EmbedDocuments(self, documents: list[dict]) -> NDArray[np.float32]:
        raise NotImplementedError("Must overwrite")

    @abstractmethod
    def CalculateScores(self, query: list[str]) -> NDArray[np.float32]:
        """
        Embeds and computes the similarity scores between the queries and the document embeddings (requires index).
        @param query: The input text(s) to which relevant passages should be found.
        @return: A numpy array of shape (n_queries, n_documents) with the scores for each query-document pair.
        """
        raise NotImplementedError("Must overwrite")
