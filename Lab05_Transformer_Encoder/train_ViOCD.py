import os
import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_score, recall_score

from vocabs.ViOCD import ViOCDVocab
from data_utils.ViOCD import ViOCDDataset, collate_fn
from models.classification import TransformerModel

from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for item in progress_bar:
        input_ids = item["input_ids"].to(device)
        labels = item["labels"].to(device)

        optimizer.zero_grad()

        _, loss = model(input_ids, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    trues = []
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, loss = model(input_ids, labels)
            total_loss += loss.item()

            trues.extend(labels.cpu().numpy())
            batch_preds = torch.argmax(logits, dim=1)
            preds.extend(batch_preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, trues, preds

def compute_score(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

if __name__ == "__main__":
    D_MODEL = 256       
    N_HEAD = 4         
    N_LAYERS = 3        
    D_FF = 1024         
    DROPOUT = 0.2  
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = ViOCDVocab("datasets/UIT-ViOCD/train.json")

    train_dataset = ViOCDDataset("datasets/UIT-ViOCD/train.json", vocab)
    train_loader = DataLoader(
        train_dataset, 
        args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    dev_dataset = ViOCDDataset("datasets/UIT-ViOCD/dev.json", vocab)
    dev_loader = DataLoader(
        dev_dataset, 
        args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    test_dataset = ViOCDDataset("datasets/UIT-ViOCD/test.json", vocab)
    test_loader = DataLoader(
        test_dataset, 
        args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    model = TransformerModel(
        d_model=D_MODEL,
        head=N_HEAD,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        vocab=vocab
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # TRAINING & EVALUATING
    os.makedirs("model_save", exist_ok=True)
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}", end=" | ")

        val_loss, trues, preds = evaluate(model, dev_loader, device)
        print(f"Val Loss: {val_loss:.4f}", end=" | ")
        
        scores = compute_score(trues, preds)
        for k,v in scores.items():
            if k == "f1":
                val_f1 = v
            print(f"{k}: {v:.4f}", end=" | ")
        print()

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "model_save/best_cls_model.pt")
            print("Best model at epoch", epoch)

    # TESTING
    test_loss, test_true, test_pred = evaluate(model, test_loader, device)
    print(f"\nTest Loss: {test_loss:.4f}", end=" | ")
    scores = compute_score(test_true, test_pred)
    for k,v in scores.items():
        print(f"{k}: {v:.4f}", end=" | ")
    print()

    target_names = [vocab.idx2label[i] for i in range(len(vocab.label2idx))]
    report = classification_report(test_true, test_pred, target_names=target_names)
    print(report)