import os
import argparse
import torch
from torch.utils.data import DataLoader
from typing import Tuple

from data_utils.Vocab import Vocab
from data_utils.dataset import PhoMTDataset, collate_fn
from models.LSTM import Seq2SeqLSTM
from models.LSTM_Bahdanau import Seq2SeqLSTM as Seq2SeqLSTM_Bahdanau
from models.LSTM_Luong import Seq2SeqLSTM as Seq2SeqLSTM_Luong

torch.manual_seed(42)
BASE_DIR = "datasets/small-PhoMT"
SRC_LANGUAGE = "english"
TGT_LANGUAGE = "vietnamese"
HIDDEN_SIZE = 256
NUM_LAYERS = 3

def rouge_l(hyp: str, ref: str) -> float:
    hyp_tokens, ref_tokens = hyp.split(), ref.split()
    n, m = len(hyp_tokens), len(ref_tokens)
    if n == 0 or m == 0:
        return 0.0

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = (
                dp[i - 1][j - 1] + 1
                if hyp_tokens[i - 1] == ref_tokens[j - 1]
                else max(dp[i - 1][j], dp[i][j - 1])
            )

    lcs = dp[n][m]
    precision, recall = lcs / n, lcs / m

    if precision + recall == 0:
        return 0.0

    rouge_l = 2 * precision * recall / (precision + recall)
    return rouge_l

def batch_rouge_l(model, src_ids, tgt_ids, vocab, tgt_language, max_len) -> Tuple[float, int]:
    pred_ids = model.predict(src_ids, max_len)
    total = 0.0
    for b in range(src_ids.size(0)):
        hyp = vocab.decode_sentence(pred_ids[b], tgt_language)
        ref = vocab.decode_sentence(tgt_ids[b], tgt_language)
        total += rouge_l(hyp, ref)
    return total, src_ids.size(0)

def train_epoch(model, dataloader, optimizer, device, vocab,
                src_language="english", tgt_language="vietnamese"):
    model.train()
    total_loss, total_rouge = 0.0, 0.0

    for item in dataloader:
        src_ids = item[src_language].to(device)
        tgt_ids = item[tgt_language].to(device)

        optimizer.zero_grad()
        logits = model(src_ids, tgt_ids)
        loss = model.loss_fn(logits.reshape(-1, logits.size(-1)), tgt_ids[:, 1:].reshape(-1),)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        rouge_sum, _ = batch_rouge_l(model, src_ids, tgt_ids, vocab, tgt_language, tgt_ids.size(1))
        total_rouge += rouge_sum / src_ids.size(0)

    avg_loss = total_loss / len(dataloader)
    avg_rouge = total_rouge / len(dataloader)

    return avg_loss, avg_rouge

@torch.no_grad()
def evaluate(model, dataloader, vocab, device,
             src_language="english", tgt_language="vietnamese"):
    model.eval()
    total_loss, total_rouge, n_samples = 0.0, 0.0, 0

    for item in dataloader:
        src_ids = item[src_language].to(device)
        tgt_ids = item[tgt_language].to(device)

        logits = model(src_ids, tgt_ids)
        loss = model.loss_fn(logits.reshape(-1, logits.size(-1)), tgt_ids[:, 1:].reshape(-1),)
        total_loss += loss.item() * src_ids.size(0)

        rouge_sum, batch_size = batch_rouge_l(model, src_ids, tgt_ids, vocab, tgt_language, tgt_ids.size(1))
        total_rouge += rouge_sum
        n_samples += batch_size

    avg_loss = total_loss / len(dataloader.dataset)
    avg_rouge = total_rouge / n_samples

    return avg_loss, avg_rouge

@torch.no_grad()
def test(model, dataloader, vocab, device, max_len,
         src_language="english", tgt_language="vietnamese"):
    model.eval()
    total_rouge, n_samples = 0.0, 0

    for i, item in enumerate(dataloader):
        src_ids = item[src_language].to(device)
        tgt_ids = item[tgt_language].to(device)

        pred_ids = model.predict(src_ids, max_len=max_len)
        for b in range(src_ids.size(0)):
            hyp = vocab.decode_sentence(pred_ids[b], tgt_language)
            ref = vocab.decode_sentence(tgt_ids[b], tgt_language)
            score = rouge_l(hyp, ref)

            total_rouge += score
            n_samples += 1

            if i < 3:
                print("\nSRC:", vocab.decode_sentence(src_ids[b], src_language))
                print("HYP:", hyp)
                print("REF:", ref)
                print("ROUGE:", score)

    return total_rouge / n_samples

def select_model(name, vocab, device, hidden_size, num_layers):
    models = {
        "lstm": Seq2SeqLSTM, #BAI1
        "lstm_bahdanau": Seq2SeqLSTM_Bahdanau, #BAI2
        "lstm_luong": Seq2SeqLSTM_Luong, #BAI3
    }
    model = models[name](
        d_model=hidden_size, 
        num_layers=num_layers, 
        vocab=vocab
        )
    return model.to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "lstm_bahdanau", "lstm_luong"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocab(BASE_DIR, SRC_LANGUAGE, TGT_LANGUAGE)

    train_dataset = PhoMTDataset(os.path.join(BASE_DIR, "small-train.json"), vocab)
    dev_dataset = PhoMTDataset(os.path.join(BASE_DIR, "small-dev.json"), vocab)
    test_dataset = PhoMTDataset(os.path.join(BASE_DIR, "small-test.json"), vocab)

    train_loader = DataLoader(
        train_dataset, 
        args.batch_size, 
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, vocab.pad_idx)
    )
    dev_loader = DataLoader(
        dev_dataset, 
        args.batch_size, 
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, vocab.pad_idx)
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, vocab.pad_idx)
    )

    model = select_model(
        name=args.model, 
        vocab=vocab, 
        device=device, 
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # TRAINING
    for epoch in range(1, args.epochs + 1):
        loss, rouge = train_epoch(model, train_loader, optimizer, device, vocab)
        print(f"[Epoch {epoch}] Train Loss: {loss:.4f} | Rouge-L: {rouge:.4f}")

    os.makedirs("model_save", exist_ok=True)
    torch.save(model.state_dict(), f"model_save/{args.model}.pt")

    # EVALUATING
    print("Evaluating...")
    dev_loss, dev_rouge = evaluate(model, dev_loader, vocab, device)
    print(f"Dev Loss: {dev_loss:.4f} | Rouge-L: {dev_rouge:.4f}")

    # model = select_model(args.model, vocab, device, HIDDEN_SIZE, NUM_LAYERS)
    # model.load_state_dict(torch.load(f"model_save/{args.model}.pt"))
    # model.to(device)
    # model.eval()

    # TESTING
    test_rouge = test(model, test_loader, vocab, device)
    print(f"Test Rouge-L: {test_rouge:.4f}")
