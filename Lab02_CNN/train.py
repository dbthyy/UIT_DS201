import torch
from transformers import AutoImageProcessor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def train(model, dataloader, loss_fn, optimizer, device, is_resnet50=False):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        images, labels = batch
        images = batch["image"].to(device)
        if is_resnet50:
            processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
            images = processor(images, return_tensors="pt")
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
    return trues, preds

def compute_metrics(trues, preds):
    accuracy = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro")
    precision = precision_score(trues, preds, average="macro")
    recall = recall_score(trues, preds, average="macro")
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }