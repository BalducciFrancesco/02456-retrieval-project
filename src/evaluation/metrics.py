
from src.data_utils.query import Query


def calculateRecall(relevancies: list[bool], query: Query, k: int):
    total_relevant = query.getNumberOfRelevantDocuments()
    if total_relevant == 0:
        return 0.0
    return sum(relevancies[:k]) / min(k, total_relevant)


def Evaluate(relevancies: list[list[bool]], queries: list[Query], k: int):
    assert len(relevancies) == len(queries), 'The number of relevant passages must match the number of queries'
    scores = {'recall': None,
              'precision': None,
              'MRR': None}
    n_queries = len(queries)
    total_recall = 0
    total_MRR = 0
    total_precision = 0

    for i in range(n_queries):
        total_recall += calculateRecall(relevancies[i], queries[i], k)
        total_MRR += calculateMRR(relevancies[i])
        total_precision += calculatePrecision(relevancies[i], k)

    scores['recall'] = total_recall / n_queries
    scores['precision'] = total_precision / n_queries
    scores['MRR'] = total_MRR / n_queries
    return scores


def calculatePrecision(relevancies: list[bool], k: int):
    return sum(relevancies[:k]) / k


def calculateMRR(relevancies: list[bool]):
    k = 0
    for i in range(len(relevancies)):

        if relevancies[i] == False:
            k = k + 1
        else:
            break

    if k == len(relevancies):
        return 0
    k = k + 1
    return 1/k
