import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def read_prompts(prompt_file: str) -> set[str]:
    return {
        p.strip()
        for p in Path(prompt_file).read_text(encoding="utf-8").splitlines()
        if p.strip()
    }


def subset_dataset_by_prompts(data: dict, prompt_file: str) -> dict:
    keep = read_prompts(prompt_file)
    prompts = data["prompts"]
    indices = [i for i, p in enumerate(prompts) if p in keep]

    if not indices:
        raise RuntimeError(f"No prompts from {prompt_file} were found in the dataset.")

    out = {}
    for k, v in data.items():
        if torch.is_tensor(v):
            out[k] = v[indices]
        elif isinstance(v, list):
            out[k] = [v[i] for i in indices]
        else:
            out[k] = v
    return out


class EmbeddingDataset(Dataset):
    def __init__(self, data: dict):
        self.prompts = data["prompts"]
        self.cond_embeddings = data["cond_embeddings"].float()
        self.opt_aconds = data["opt_aconds"].float()

        if self.cond_embeddings.shape != self.opt_aconds.shape:
            raise ValueError(
                f"Shape mismatch: cond_embeddings={self.cond_embeddings.shape}, "
                f"opt_aconds={self.opt_aconds.shape}"
            )

    def __len__(self):
        return self.cond_embeddings.shape[0]

    def __getitem__(self, idx):
        return self.cond_embeddings[idx], self.opt_aconds[idx]


class PromptEmbeddingPredictor(nn.Module):
    def __init__(self, d_model: int = 768, hidden: int = 1536, dropout: float = 0.1):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):
        return x + self.token_mlp(x)


def cosine_sim(pred, target):
    pred_flat = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)
    return F.cosine_similarity(pred_flat, target_flat, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="training_dataset.pt")
    parser.add_argument("--test_prompts", default="prompts_test.txt")
    parser.add_argument("--ckpt", default="prompt_predictor.pt")
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    data = torch.load(args.dataset, map_location="cpu")
    test_data = subset_dataset_by_prompts(data, args.test_prompts)
    test_dataset = EmbeddingDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = PromptEmbeddingPredictor(
        d_model=ckpt["d_model"],
        hidden=ckpt["hidden"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mse_sum = 0.0
    mae_sum = 0.0
    cos_sum = 0.0
    id_mse_sum = 0.0
    id_cos_sum = 0.0
    count = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)

            bsz = x.size(0)
            mse_sum += F.mse_loss(pred, y, reduction="sum").item()
            mae_sum += F.l1_loss(pred, y, reduction="sum").item()
            cos_sum += cosine_sim(pred, y).sum().item()

            id_mse_sum += F.mse_loss(x, y, reduction="sum").item()
            id_cos_sum += cosine_sim(x, y).sum().item()
            count += bsz

    print(f"Test samples: {count}")
    print(f"pred_mse: {mse_sum / count:.6f}")
    print(f"pred_mae: {mae_sum / count:.6f}")
    print(f"pred_cosine: {cos_sum / count:.6f}")
    print(f"identity_mse: {id_mse_sum / count:.6f}")
    print(f"identity_cosine: {id_cos_sum / count:.6f}")


if __name__ == "__main__":
    main()
