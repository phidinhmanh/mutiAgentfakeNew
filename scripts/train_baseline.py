"""Train baseline model (PhoBERT fine-tuning placeholder)."""
import argparse
import logging
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_NAME = "tranthaihoa/vifactcheck"
MODEL_NAME = "vinai/phobert-base"


def prepare_dataset(tokenizer: AutoTokenizer, max_length: int = 256) -> tuple[Dataset, Dataset]:
    """Prepare dataset for training.

    Args:
        tokenizer: PhoBERT tokenizer
        max_length: Maximum sequence length

    Returns:
        Train and validation datasets
    """
    dataset = load_dataset(DATASET_NAME)

    def tokenize_function(examples: dict) -> dict:
        return tokenizer(
            examples["claim"],
            examples.get("evidence", [""] * len(examples["claim"])),
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    label_map = {"REAL": 0, "FAKE": 1, "true": 1, "false": 0}

    def map_labels(example: dict) -> dict:
        label = example.get("label", "REAL")
        if isinstance(label, str):
            label = label_map.get(label.lower(), 0)
        elif isinstance(label, bool):
            label = int(label)
        return {"labels": label}

    train_dataset = dataset["train"].map(tokenize_function, batched=True)
    val_dataset = dataset["validation"].map(tokenize_function, batched=True)

    train_dataset = train_dataset.map(map_labels)
    val_dataset = val_dataset.map(map_labels)

    train_dataset = train_dataset.filter(
        lambda x: x["labels"] in [0, 1], desc="Filtering invalid labels"
    )
    val_dataset = val_dataset.filter(
        lambda x: x["labels"] in [0, 1], desc="Filtering invalid labels"
    )

    return train_dataset, val_dataset


def train_baseline(
    output_dir: str = "./models/phobert-fake-news",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
) -> None:
    """Train baseline model.

    Args:
        output_dir: Model output directory
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
    """
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    logger.info("Preparing dataset...")
    train_dataset, val_dataset = prepare_dataset(tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train baseline model")
    parser.add_argument("--output-dir", type=str, default="./models/phobert-fake-news")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{args.output_dir}/logs").mkdir(parents=True, exist_ok=True)

    train_baseline(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()