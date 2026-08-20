import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptEmbeddingPredictor(nn.Module):
    def __init__(self, d_model=768, hidden=1536, dropout=0.1):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):
        return x + self.token_mlp(x)


def cosine_loss(pred, target):
    pred_flat = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)
    return F.cosine_similarity(pred_flat, target_flat, dim=1)


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    dataset_path = "training_dataset.pt"
    model_path = "prompt_predictor.pt"
    out_csv = "predictor_benchmark.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = torch.load(dataset_path, map_location="cpu")
    ckpt = torch.load(model_path, map_location="cpu")

    model = PromptEmbeddingPredictor(
        d_model=ckpt["d_model"],
        hidden=ckpt["hidden"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cond_embeddings = dataset["cond_embeddings"].float()
    opt_aconds = dataset["opt_aconds"].float()
    source_files = dataset.get("source_files", [f"sample_{i}" for i in range(len(cond_embeddings))])

    rows = []
    total_pred_mse = 0.0
    total_pred_cos = 0.0
    total_id_mse = 0.0
    total_id_cos = 0.0
    total_pred_ms = 0.0

    with torch.no_grad():
        for i in range(cond_embeddings.shape[0]):
            x = cond_embeddings[i : i + 1].to(device)
            y = opt_aconds[i : i + 1].to(device)

            # Predictor timing
            for _ in range(10):
                _ = model(x)
            sync_if_cuda(device)

            t0 = time.perf_counter()
            for _ in range(100):
                pred = model(x)
            sync_if_cuda(device)
            pred_ms = (time.perf_counter() - t0) * 1000.0 / 100.0

            # Metrics
            pred_mse = F.mse_loss(pred, y).item()
            pred_cos = cosine_loss(pred, y).item()
            pred_mae = torch.mean(torch.abs(pred - y)).item()

            # Identity baseline: do nothing, i.e. use cond_embedding as output
            id_mse = F.mse_loss(x, y).item()
            id_cos = cosine_loss(x, y).item()

            delta_norm = torch.norm((y - x).reshape(1, -1), p=2).item()

            rows.append(
                {
                    "index": i,
                    "source_file": source_files[i],
                    "pred_mse": pred_mse,
                    "pred_cosine": pred_cos,
                    "pred_mae": pred_mae,
                    "identity_mse": id_mse,
                    "identity_cosine": id_cos,
                    "delta_l2_norm": delta_norm,
                    "predictor_latency_ms": pred_ms,
                }
            )

            total_pred_mse += pred_mse
            total_pred_cos += pred_cos
            total_id_mse += id_mse
            total_id_cos += id_cos
            total_pred_ms += pred_ms

            print(
                f"[{i}] pred_mse={pred_mse:.6f} pred_cos={pred_cos:.6f} "
                f"id_mse={id_mse:.6f} id_cos={id_cos:.6f} latency_ms={pred_ms:.3f}"
            )

    n = len(rows)
    summary = {
        "index": "mean",
        "source_file": "ALL",
        "pred_mse": total_pred_mse / n,
        "pred_cosine": total_pred_cos / n,
        "pred_mae": sum(r["pred_mae"] for r in rows) / n,
        "identity_mse": total_id_mse / n,
        "identity_cosine": total_id_cos / n,
        "delta_l2_norm": sum(r["delta_l2_norm"] for r in rows) / n,
        "predictor_latency_ms": total_pred_ms / n,
    }

    rows.append(summary)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", out_csv)
    print("Summary:")
    print(summary)


if __name__ == "__main__":
    main()
