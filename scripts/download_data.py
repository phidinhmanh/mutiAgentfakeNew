"""Script to download and prepare ViFactCheck dataset."""
import argparse
import logging
from pathlib import Path

from datasets import load_dataset

from fake_news_detector.data.preprocessing import preprocess_for_embedding
from fake_news_detector.rag.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_NAME = "tranthaihoa/vifactcheck"


def download_dataset(output_dir: Path | None = None) -> None:
    """Download and save ViFactCheck dataset.

    Args:
        output_dir: Directory to save dataset
    """
    logger.info(f"Downloading dataset: {DATASET_NAME}")

    # trust_remote_code=True for datasets without loading scripts
    # trust_remote_files=True for parquet-based datasets
    ds_kwargs: dict[str, bool] = {
        "trust_remote_code": True,
        "trust_remote_files": True,
    }

    try:
        train_dataset = load_dataset(DATASET_NAME, split="train", **ds_kwargs)
        logger.info(f"Train samples: {len(train_dataset)}")
    except Exception as e:
        logger.warning(f"Could not load 'train' split: {e}")
        train_dataset = None

    try:
        val_dataset = load_dataset(DATASET_NAME, split="validation", **ds_kwargs)
        logger.info(f"Validation samples: {len(val_dataset)}")
    except Exception as e:
        logger.warning(f"Could not load 'validation' split: {e}")
        val_dataset = None

    try:
        test_dataset = load_dataset(DATASET_NAME, split="test", **ds_kwargs)
        logger.info(f"Test samples: {len(test_dataset)}")
    except Exception as e:
        logger.warning(f"Could not load 'test' split: {e}")
        test_dataset = None

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if train_dataset is not None:
            train_dataset.save_to_disk(str(output_dir / "train"))
        if val_dataset is not None:
            val_dataset.save_to_disk(str(output_dir / "validation"))
        if test_dataset is not None:
            test_dataset.save_to_disk(str(output_dir / "test"))

        logger.info(f"Saved dataset to {output_dir}")

    # Fallback: load entire dataset if named splits not found
    if train_dataset is None and val_dataset is None and test_dataset is None:
        logger.info("Attempting to load full dataset...")
        full_dataset = load_dataset(DATASET_NAME, **ds_kwargs)
        logger.info(f"Full dataset loaded: {full_dataset}")


def build_faiss_index(
    dataset_name: str = DATASET_NAME,
    output_dir: str = "./data/faiss_index",
    max_samples: int = 5000,
) -> None:
    """Build FAISS index from dataset.

    Args:
        dataset_name: HuggingFace dataset name
        output_dir: Output directory for index
        max_samples: Maximum samples to index
    """
    logger.info("Loading dataset for indexing...")
    # trust flags prevent 404 on dataset_infos.json/vifactcheck.py
    dataset = load_dataset(
        dataset_name, split="train", trust_remote_code=True, trust_remote_files=True
    )

    documents = []
    for i, item in enumerate(dataset):
        if i >= max_samples:
            break

        evidence = item.get("evidence", "")
        claim = item.get("claim", "")
        content = f"{claim} {evidence}".strip()
        if content:
            documents.append({
                "content": preprocess_for_embedding(content),
                "claim": claim,
                "evidence": evidence,
                "label": item.get("label", ""),
                "source": item.get("source", ""),
            })

    logger.info(f"Creating index from {len(documents)} documents")

    vector_store = VectorStore()
    vector_store.add_documents(documents)
    vector_store.save(output_dir)

    logger.info(f"Index saved to {output_dir}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download ViFactCheck dataset")
    parser.add_argument("--output-dir", type=str, default="./data")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--max-samples", type=int, default=5000)

    args = parser.parse_args()

    download_dataset(Path(args.output_dir))

    if args.build_index:
        build_faiss_index(
            output_dir=f"{args.output_dir}/faiss_index",
            max_samples=args.max_samples,
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
