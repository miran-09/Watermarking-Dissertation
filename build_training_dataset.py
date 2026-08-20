import argparse
import os
from pathlib import Path

import torch


REQUIRED_KEYS = {"prompt", "cond_embedding", "opt_acond", "opt_wm"}


def load_pair(path: Path):
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        return None
    if not REQUIRED_KEYS.issubset(obj.keys()):
        return None
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_dir", required=True, help="Directory with pair .pt files")
    parser.add_argument("--out_file", required=True, help="Output dataset file")
    parser.add_argument("--manifest_file", required=True, help="Text file listing used pairs")
    args = parser.parse_args()

    pairs_dir = Path(args.pairs_dir)
    if not pairs_dir.exists():
        raise FileNotFoundError(f"Pairs directory not found: {pairs_dir}")

    pair_files = sorted([p for p in pairs_dir.rglob("*.pt") if p.is_file()])

    records = []
    used_files = []

    for path in pair_files:
        pair = load_pair(path)
        if pair is None:
            continue
        records.append(pair)
        used_files.append(str(path))

    if not records:
        raise RuntimeError(f"No valid pair files found in {pairs_dir}")

    prompts = [r["prompt"] for r in records]
    cond_embeddings = torch.cat([r["cond_embedding"].float().cpu() for r in records], dim=0)
    opt_aconds = torch.cat([r["opt_acond"].float().cpu() for r in records], dim=0)
    opt_wms = torch.cat([r["opt_wm"].cpu() for r in records], dim=0)

    dataset = {
        "source_files": used_files,
        "prompts": prompts,
        "cond_embeddings": cond_embeddings,
        "opt_aconds": opt_aconds,
        "opt_wms": opt_wms,
    }

    torch.save(dataset, args.out_file)

    with open(args.manifest_file, "w", encoding="utf-8") as f:
        for item in used_files:
            f.write(item + "\n")

    print(f"Saved dataset to: {args.out_file}")
    print(f"Saved manifest to: {args.manifest_file}")
    print(f"Num valid samples: {len(prompts)}")
    print(f"cond_embeddings shape: {tuple(cond_embeddings.shape)}")
    print(f"opt_aconds shape: {tuple(opt_aconds.shape)}")
    print(f"opt_wms shape: {tuple(opt_wms.shape)}")
    print("First 5 prompts:")
    for p in prompts[:5]:
        print(" -", p)


if __name__ == "__main__":
    main()
