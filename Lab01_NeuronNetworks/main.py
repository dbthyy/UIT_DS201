import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from Mnist_Dataset import MinstDataset, collate_fn
from MLP_1_Layer import MLP1Layer
from MLP_3_Layer import MLP3Layer

def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(dataloader, model, device) -> dict:
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            outputs = torch.argmax(outputs, dim=1)

            preds.extend(outputs.cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())

    return compute_score(trues, preds)

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

def model_selector(model_name: str):
    if model_name == "MLP_1_Layer":
        return MLP1Layer(
            input_size=28*28,
            output_size=10,
            dropout_rate=0.1
        )
    elif model_name == "MLP_3_Layer":
        return MLP3Layer(
            input_size=28*28,
            hidden_size1=128,
            hidden_size2=64,
            output_size=10,
            dropout_rate=0.1
        )
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["MLP_1_Layer", "MLP_3_Layer"], required=True)
    args = parser.parse_args()

    EPOCHS = 10
    BATCH_SIZE = 32
    LR = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MinstDataset(
        img_path="data/train-images.idx3-ubyte",
        label_path="data/train-labels.idx1-ubyte"
    )
    test_dataset = MinstDataset(
        img_path="data/t10k-images.idx3-ubyte",
        label_path="data/t10k-labels.idx1-ubyte"
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = model_selector(args.model).to(device)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    best_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}")
        loss = train(model, train_loader, loss_fn, optimizer, device)
        print(f"Train Loss: {loss:.4f} | ", end="")

        scores = evaluate(test_loader, model, device)
        for score_name, score_value in scores.items():
            print(f"{score_name}: {score_value:.4f} | ", end="")
        print()

        current_acc = scores["accuracy"]
        if current_acc > best_acc:
            best_acc = current_acc
            torch.save(
                model.state_dict(),
                f"checkpoints/{args.model}_best_model.pth"
            )        
    print(f"\nBest Accuracy for Model {args.model}:", best_acc)