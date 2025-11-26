import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from data_utils.vsfc import VSFCVocab, VSFCDataset, collate_fn
from models.lstm import LstmModule
from models.gru import GruModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    trues = []
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            preds = torch.argmax(logits, dim=-1)

            trues.extend(labels.cpu().numpy())
            preds.extend(preds.cpu().numpy())

    return trues, preds

def compute_score(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

def model_selector(model_name: str, 
                   vocab_size, 
                   hidden_size, 
                   n_layer, 
                   n_labels, 
                   padding_idx):
    if model_name == "lstm":
        return LstmModule(vocab_size, hidden_size, n_layer, n_labels, padding_idx)
    elif model_name == "gru":
        return GruModule(vocab_size, hidden_size, n_layer, n_labels, padding_idx)

if __name__ == "__main__":
    HIDDEN_SIZE = 256
    N_LAYER = 5

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    model_name = args.model_name
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    vocab = VSFCVocab()
    train_ds = VSFCDataset("train", vocab)
    dev_ds = VSFCDataset("dev", vocab)
    test_ds  = VSFCDataset("test", vocab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = model_selector(
        model_name, 
        vocab_size=len(vocab.w2id), 
        hidden_size=HIDDEN_SIZE, 
        n_layer=N_LAYER, 
        n_labels=len(vocab.l2id), 
        padding_idx=vocab.w2id["<p>"]
        ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()

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
            torch.save(best_model_state, f"checkpoints/{model_name}_best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopped")
                break

    print(f"\nBest F1 at epoch {best_epoch}:  {best_f1}")

    best_model = model_selector(
        vocab_size=len(vocab.w2id), 
        hidden_size=HIDDEN_SIZE, 
        n_layer=N_LAYER, 
        n_labels=len(vocab.l2id), 
        padding_idx=vocab.w2id["<p>"]
        ).to(device)
    best_model.load_state_dict(torch.load(f"checkpoints/{model_name}_best_model.pth"))
    
    trues, preds = evaluate(best_model, test_loader)
    scores = compute_score(trues, preds)
    print("Evaluation on test set: ")
    for k,v in scores.items():
        print(f"{k}: {v:.4f}", end=" | ")
    print()

    labels = list(vocab.l2id.keys())
    cm = confusion_matrix(trues, preds)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)