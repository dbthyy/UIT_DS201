import torch
import argparse

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from data_utils.phoner import PhoNERVocab, PhoNERDataset, collate_fn
from models.bilstm import BiLSTMEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label_ids"].to(device)

        logits = model(input_ids)
        logits = logits.view(-1, logits.size(-1)) # (B*T, C)
        labels = labels.view(-1) # (B*T,)

        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader):
    model.eval()
    all_trues = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label_ids"].to(device)

            logits = model(input_ids)
            preds = torch.argmax(logits, dim=-1)

            all_trues.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    flat_true = [x for seq in all_trues for x in seq]
    flat_pred = [x for seq in all_preds for x in seq]

    return flat_true, flat_pred


def compute_score(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='macro', zero_division=0)
    }

if __name__ == "__main__":
    HIDDEN_SIZE = 256
    N_LAYERS = 5

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bilstm")
    parser.add_argument("--base", type=str, default="syllable")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    model_name = args.model_name
    base = args.base
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    vocab = PhoNERVocab(base)
    train_ds = PhoNERDataset("train", base, vocab)
    dev_ds   = PhoNERDataset("dev", base, vocab)
    test_ds  = PhoNERDataset("test", base, vocab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader   = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = BiLSTMEncoder(
        vocab_size=len(vocab.w2id),
        hidden_size=HIDDEN_SIZE,
        n_layers=N_LAYERS,
        n_labels=len(vocab.l2id),
        padding_idx=vocab.w2id[vocab.pad]
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.w2id[vocab.pad])

    best_f1 = 0
    best_epoch = 0
    best_model_state = None
    patience = 5
    epochs_no_improve = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        loss = train(model, train_loader, loss_fn, optimizer)
        print(f"Train Loss: {loss:.4f}", end=" | ")

        trues, preds = evaluate(model, dev_loader)
        scores = compute_score(trues, preds)
        for k,v in scores.items():
            print(f"{k}: {v:.4f}", end=" | ")
        print()

        if scores["f1"] > best_f1:
            best_f1 = scores["f1"]
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            torch.save(best_model_state, f"checkpoints/{model_name}_{base}_best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopped")
                break
            
    print(f"Best F1 at epoch {best_epoch}:  {best_f1}")

    best_model = BiLSTMEncoder(
        vocab_size=len(vocab.w2id),
        hidden_size=HIDDEN_SIZE,
        n_layers=N_LAYERS,
        n_labels=len(vocab.l2id),
        padding_idx=vocab.w2id[vocab.pad]
    ).to(device)
    best_model.load_state_dict(torch.load(f"checkpoints/{model_name}_{base}_best_model.pth"))

    trues, preds = evaluate(best_model, test_loader)
    scores = compute_score(trues, preds)
    print("Evaluation on test set: ")
    for k,v in scores.items():
        print(f"{k}: {v:.4f}", end=" | ")
    print()