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
    def __init__(self, data: dict, shuffle_targets: bool = False, seed: int = 42):
        self.prompts = data["prompts"]
        self.cond_embeddings = data["cond_embeddings"].float()
        self.opt_aconds = data["opt_aconds"].float()

        if self.cond_embeddings.shape != self.opt_aconds.shape:
            raise ValueError(
                f"Shape mismatch: cond_embeddings={self.cond_embeddings.shape}, "
                f"opt_aconds={self.opt_aconds.shape}"
            )

        if shuffle_targets:
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(self.opt_aconds.shape[0], generator=g)
            self.opt_aconds = self.opt_aconds[perm]

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
        delta = self.token_mlp(x)
        return x + delta


def cosine_loss(pred, target):
    pred_flat = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)
    return 1.0 - F.cosine_similarity(pred_flat, target_flat, dim=1).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="training_dataset.pt")
    parser.add_argument("--train_prompts", default="prompts_train.txt")
    parser.add_argument("--val_prompts", default="prompts_val.txt")
    parser.add_argument("--out_model", default="prompt_predictor.pt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_targets", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    data = torch.load(args.dataset, map_location="cpu")

    train_data = subset_dataset_by_prompts(data, args.train_prompts)
    val_data = subset_dataset_by_prompts(data, args.val_prompts)

    if args.shuffle_targets:
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(train_data["opt_aconds"].shape[0], generator=g)
        train_data["opt_aconds"] = train_data["opt_aconds"][perm]
        print("WARNING: training targets shuffled for negative control.")

    train_dataset = EmbeddingDataset(train_data, shuffle_targets=False, seed=args.seed)
    val_dataset = EmbeddingDataset(val_data, shuffle_targets=False, seed=args.seed)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = PromptEmbeddingPredictor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            mse = F.mse_loss(pred, y)
            cos = cosine_loss(pred, y)
            loss = mse + 0.1 * cos

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                mse = F.mse_loss(pred, y)
                cos = cosine_loss(pred, y)
                loss = mse + 0.1 * cos
                val_loss_sum += loss.item() * x.size(0)

        val_loss = val_loss_sum / len(val_loader.dataset)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "d_model": 768,
                    "hidden": 1536,
                    "dropout": 0.1,
                    "best_val_loss": best_val,
                },
                args.out_model,
            )

    print(f"Saved best model to {args.out_model}")
    print(f"Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
