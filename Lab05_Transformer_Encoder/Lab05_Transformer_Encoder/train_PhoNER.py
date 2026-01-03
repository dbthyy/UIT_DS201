import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score

from vocabs.PhoNER import PhoNERVocab
from data_utils.PhoNER import PhoNERDataset, collate_fn
from models.ner import TransformerModel
from train_ViOCD import train_epoch, compute_score

def evaluate_ner(model, dataloader, vocab, device):
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, loss = model(input_ids, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)

            for p, l in zip(preds, labels):
                sent_pred = []
                sent_true = []

                for pi, li in zip(p, l):
                    if li.item() == -100:
                        continue
                    sent_pred.append(vocab.idx2label[pi.item()])
                    sent_true.append(vocab.idx2label[li.item()])

                all_preds.append(sent_pred)
                all_trues.append(sent_true)

    return total_loss / len(dataloader), all_trues, all_preds

def compute_score(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
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

    vocab = PhoNERVocab("syllable")

    train_dataset = PhoNERDataset("datasets/PhoNER_COVID19/syllable/train_syllable.conll", vocab)
    train_loader = DataLoader(
        train_dataset, 
        args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    dev_dataset = PhoNERDataset("datasets/PhoNER_COVID19/syllable/dev_syllable.conll", vocab)
    dev_loader = DataLoader(
        dev_dataset, 
        args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    test_dataset = PhoNERDataset("datasets/PhoNER_COVID19/syllable/test_syllable.conll", vocab)
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

        val_loss, trues, preds = evaluate_ner(model, dev_loader, vocab, device)
        print(f"Val Loss: {val_loss:.4f}", end=" | ")
        
        scores = compute_score(trues, preds)
        for k,v in scores.items():
            if k == "f1":
                val_f1 = v
            print(f"{k}: {v:.4f}", end=" | ")
        print()

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "model_save/best_ner_model.pt")
            print("Best model at epoch", epoch)

    # TESTING
    test_loss, test_true, test_pred = evaluate_ner(model, test_loader, vocab, device)
    print(f"\nTest Loss: {test_loss:.4f}", end=" | ")
    
    scores = compute_score(test_true, test_pred)
    for k,v in scores.items():
        print(f"{k}: {v:.4f}", end=" | ")
    print()

    report = classification_report(test_true, test_pred)
    print(report)