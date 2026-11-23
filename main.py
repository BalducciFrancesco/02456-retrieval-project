
import hydra
from itertools import product
from pathlib import Path
from omegaconf import OmegaConf

from src.data_utils.corpus import Corpus
from src.retrievers.bm25 import BM25
from src.retrievers.dpr import DPR
from src.retrievers.tf_idf import TFIDF
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate


@hydra.main(config_path='config', config_name='fiqa_experiment.yaml', version_base=None)
def main(cfg):
    ### Load a dataset
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')
    train_queries, val_queries, test_queries = corpus.GetQuerySplits()
    index_path = cfg.retriever.index_path.format(dataset=cfg.dataset.name)

    ### Initialize the model
    match cfg.retriever.method:
        case "dpr":

            # Grid search parameters
            epochs = [10, 15]
            learning_rates = [1e-4, 5e-4]
            batch_sizes = [16, 32]
            pooling_types = ['mean', 'cls']

            best_mrr = -1
            best_model = None
            best_params = None

            # Prevent previous runs from being reused
            search_name = "grid_search_1"

            # Grid search
            for i, (epoch, lr, batch_size, pooling) in enumerate(
                product(epochs, learning_rates, batch_sizes, pooling_types),
                start=1
            ):

                print(f"\n{'=' * 50}")
                print(f"Grid search trial {i}")
                print(f"Epochs: {epoch}")
                print(f"Learning rate: {lr}")
                print(f"Batch size: {batch_size}")
                print(f"Pooling: {pooling}")

                # Copy configuration
                trial_cfg = OmegaConf.create(
                    OmegaConf.to_container(cfg, resolve=True)
                )

                # Update hyperparameters
                trial_cfg.retriever.training.epochs = epoch
                trial_cfg.retriever.training.lr = lr
                trial_cfg.batch_size = batch_size
                trial_cfg.retriever.pooling_type = pooling

                # Unique paths for each trial
                trial_name = f"{search_name}_trial_{i}"

                trial_cfg.retriever.index_path = (
                    f"index/{trial_name}_{{dataset}}.pkl"
                )

                trial_cfg.retriever.query_encoder.weights_path = (
                    f"weights/{trial_name}_query_{{dataset}}.pt"
                )

                trial_cfg.retriever.document_encoder.weights_path = (
                    f"weights/{trial_name}_document_{{dataset}}.pt"
                )

                trial_index_path = trial_cfg.retriever.index_path.format(
                    dataset=trial_cfg.dataset.name
                )

                # Make sure the trial starts from scratch
                trial_paths = [
                    trial_index_path,
                    trial_cfg.retriever.query_encoder.weights_path.format(
                        dataset=trial_cfg.dataset.name
                    ),
                    trial_cfg.retriever.document_encoder.weights_path.format(
                        dataset=trial_cfg.dataset.name
                    )
                ]

                if any(Path(p).exists() for p in trial_paths):
                    raise FileExistsError(
                        f"Trial {i} already has saved files. "
                        "Change search_name to start a new search."
                    )

                # Initialize and train DPR
                trial_model = DPR(
                    documents=corpus.all_documents,
                    cfg=trial_cfg,
                    index_path=trial_index_path
                )

                trial_model.Prepare(corpus.train)

                # Retrieve validation documents
                val_relevancies = retrieveQueryAndGetRelevancies(
                    trial_model, val_queries, k=cfg.top_k
                )

                # Evaluate using existing metrics function
                val_scores = Evaluate(val_relevancies, val_queries)

                print(f"Validation results: {val_scores}")

                # Keep the best model based on validation MRR
                if val_scores['MRR'] > best_mrr:

                    best_mrr = val_scores['MRR']
                    best_model = trial_model

                    best_params = {
                        "epochs": epoch,
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "pooling": pooling
                    }

            print(f"\n{'=' * 50}")
            print("BEST DPR HYPERPARAMETERS")
            print(best_params)
            print(f"Best validation MRR: {best_mrr:.4f}")

            # Use the best DPR model for final evaluation
            model = best_model

        case "tf_idf":
            model = TFIDF(documents=corpus.all_documents, index_path=index_path)

        case "bm25":
            model = BM25(documents=corpus.all_documents, index_path=index_path)

        case _:
            raise ValueError(f"Unknown retriever: {cfg.retriever}")

    print("Model ready, evaluating now...")

    ### Retrieve top-k relevant queries
    train_relevancies = retrieveQueryAndGetRelevancies(model, train_queries, k=cfg.top_k)
    val_relevancies = retrieveQueryAndGetRelevancies(model, val_queries, k=cfg.top_k)
    test_relevancies = retrieveQueryAndGetRelevancies(model, test_queries, k=cfg.top_k)

    ### Evaluate
    train_scores = Evaluate(train_relevancies, train_queries, k=cfg.top_k)
    val_scores = Evaluate(val_relevancies, val_queries, k=cfg.top_k)
    test_scores = Evaluate(test_relevancies, test_queries, k=cfg.top_k)

    print(f'Train results: {train_scores}')
    print(f'Val results: {val_scores}')
    print(f'Test results: {test_scores}')


if __name__ == '__main__':
    main()
