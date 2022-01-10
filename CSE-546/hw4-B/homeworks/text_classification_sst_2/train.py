import torch
import torch.optim as optim

if __name__ == "__main__":
    from problems import (  # type: ignore
        RNNBinaryClassificationModel,
        collate_fn,
        get_parameters,
    )
else:
    from .problems import RNNBinaryClassificationModel, collate_fn, get_parameters

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import load_dataset, load_embedding_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    # Get parameters from problems
    params = get_parameters()
    TRAINING_BATCH_SIZE = params["TRAINING_BATCH_SIZE"]
    NUM_EPOCHS = params["NUM_EPOCHS"]
    LEARNING_RATE = params["LEARNING_RATE"]
    VAL_BATCH_SIZE = params["VAL_BATCH_SIZE"]

    # Load datasets
    train_dataset, val_dataset = load_dataset("sst-2")

    # Create data loaders for creating and iterating over batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=VAL_BATCH_SIZE, collate_fn=collate_fn
    )

    # Print out some random examples from the data
    print("Data examples:")
    random_indices = torch.randperm(len(train_dataset))[:8].tolist()
    for index in random_indices:
        sequence_indices, label = (
            train_dataset.sentences[index],
            train_dataset.labels[index],
        )
        sentiment = "Positive" if label == 1 else "Negative"
        sequence = train_dataset.indices_to_tokens(sequence_indices)
        print(f"Sentiment: {sentiment}. Sentence: {sequence}")
    print()

    embedding_matrix = load_embedding_matrix(train_dataset.vocab)

    for model_type in ["LSTM", "RNN", "GRU"]:
        model = RNNBinaryClassificationModel(embedding_matrix.clone(), model_type).to(
            DEVICE
        )
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(NUM_EPOCHS):
            # Total loss across train data
            train_loss = 0.0
            # Total number of correctly predicted training labels
            train_correct = 0
            # Total number of training sequences processed
            train_seqs = 0

            print(f"Model Type: {model_type} Epoch {epoch + 1}/{NUM_EPOCHS}")
            tqdm_train_loader = tqdm(train_loader, leave=False)

            model.train()
            for batch_idx, (sentences_batch, labels_batch) in enumerate(
                tqdm_train_loader
            ):
                sentences_batch, labels_batch = (
                    sentences_batch.to(DEVICE),
                    labels_batch.to(DEVICE),
                )

                # Make predictions
                logits = model(sentences_batch)

                # Compute loss and number of correct predictions
                loss = model.loss(logits, labels_batch)
                correct = model.accuracy(logits, labels_batch).item() * len(logits)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate metrics and update status
                train_loss += loss.item()
                train_correct += correct
                train_seqs += len(sentences_batch)
                tqdm_train_loader.set_description_str(
                    f"[Loss]: {train_loss / (batch_idx + 1):.4f} [Acc]: {train_correct / train_seqs:.4f}"
                )
            tqdm_train_loader.close()
            print()

            avg_train_loss = train_loss / len(tqdm_train_loader)
            train_accuracy = train_correct / train_seqs
            print(
                f"\t[Training Loss]: {avg_train_loss:.4f} [Training Accuracy]: {train_accuracy:.4f}"
            )

            # Total loss across validation data
            val_loss = 0.0
            # Total number of correctly predicted validation labels
            val_correct = 0
            # Total number of validation sequences processed
            val_seqs = 0

            tqdm_val_loader = tqdm(val_loader, leave=False)

            model.eval()
            for batch_idx, (sentences_batch, labels_batch) in enumerate(
                tqdm_val_loader
            ):
                sentences_batch, labels_batch = (
                    sentences_batch.to(DEVICE),
                    labels_batch.to(DEVICE),
                )

                with torch.no_grad():
                    # Make predictions
                    logits = model(sentences_batch)

                    # Compute loss and number of correct predictions and accumulate metrics and update status
                    val_loss += model.loss(logits, labels_batch).item()
                    val_correct += model.accuracy(logits, labels_batch).item() * len(
                        logits
                    )
                    val_seqs += len(sentences_batch)
                    tqdm_val_loader.set_description_str(
                        f"[Loss]: {val_loss / (batch_idx + 1):.4f} [Acc]: {val_correct / val_seqs:.4f}"
                    )
            tqdm_val_loader.close()
            print()

            avg_val_loss = val_loss / len(tqdm_val_loader)
            val_accuracy = val_correct / val_seqs
            print(
                f"\t[Validation Loss]: {avg_val_loss:.4f} [Validation Accuracy]: {val_accuracy:.4f}"
            )


if __name__ == "__main__":
    train()
