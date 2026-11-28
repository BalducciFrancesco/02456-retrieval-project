from src.retrievers.base_retriever import Retriever
from src.data_utils.query import Query

def retrieveQueryAndGetRelevancies(model: Retriever, queries: list[Query], k: int, batch_size: int):
    """
    @param model: Retrieval model
    @param queries: A list of queries.
    @param k: Top-k relevant documents to consider.
    @param batch_size: Queries scored at once.

    Returns a list of lists of booleans that are True when the retrieved document matches the ground truth of that query.
    """
    N = len(queries)
    retrieved_documents = model.Lookup(queries=[query.text for query in queries], k=k, batch_size=batch_size)
    relevancies = []
    for i in range(N):
        query_relevancies = []
        for document in retrieved_documents[i]:
            if queries[i].isDocumentRelevant(document):
                query_relevancies.append(True)
            else:
                query_relevancies.append(False)
        relevancies.append(query_relevancies)
    return relevancies