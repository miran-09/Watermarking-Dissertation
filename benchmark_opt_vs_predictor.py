import argparse
import copy
import os
import time

import torch
import torch.nn as nn
from diffusers import DPMSolverMultistepScheduler

import gen_watermark as gw
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from optim_utils import get_watermarking_mask, get_watermarking_pattern


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


def load_predictor(path: str, device: str):
    ckpt = torch.load(path, map_location="cpu")
    model = PromptEmbeddingPredictor(
        d_model=ckpt["d_model"],
        hidden=ckpt["hidden"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--predictor_path", default="prompt_predictor.pt")
    parser.add_argument("--save_path", default="benchmark_ckpts")
    parser.add_argument("--dataset", default="Stable-Diffusion-Prompts")
    parser.add_argument("--gen_seed", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=50)

    # watermark args
    parser.add_argument("--w_seed", default=999999, type=int)
    parser.add_argument("--w_channel", default=0, type=int)
    parser.add_argument("--w_pattern", default="rand")
    parser.add_argument("--w_mask_shape", default="circle")
    parser.add_argument("--w_up_radius", default=30, type=int)
    parser.add_argument("--w_low_radius", default=5, type=int)
    parser.add_argument("--w_measurement", default="l1_complex")
    parser.add_argument("--w_injection", default="complex")
    parser.add_argument("--w_pattern_const", default=0, type=float)

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # gen_watermark.py expects a module-level args variable
    gw.args = args

    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        revision="main",
    ).to(device)

    predictor = load_predictor(args.predictor_path, device)

    # reuse the same clean-image dataset structure as ROBIN
    dataset = gw.OptimizedDataset(
        data_root=args.data_root,
        repeats=1,
        center_crop=False,
        set="train",
    )
    train_dataloader = gw.create_dataloader(dataset, train_batch_size=1)

    # same watermark setup as ROBIN
    init_latents_w = pipe.get_random_latents()
    opt_watermark = get_watermarking_pattern(pipe, args, device)
    mask = get_watermarking_mask(init_latents_w, args, device).detach().cpu()

    hyperparameters = copy.deepcopy(gw.hyperparameters)
    hyperparameters["max_train_steps"] = args.max_train_steps

    # baseline timing: ROBIN optimizer
    start = time.perf_counter()
    opt_wm, opt_acond = pipe.optimizer_wm_prompt(
        train_dataloader,
        hyperparameters,
        mask,
        opt_watermark,
        args.save_path,
        args,
    )
    robin_time = time.perf_counter() - start

    # predictor timing: one forward pass for the same prompt embedding
    sample_prompt = dataset[0]["prompt"]
    start = time.perf_counter()
    with torch.no_grad():
        cond_embedding = pipe.get_text_embedding(sample_prompt).to(device).float()
        pred_opt_acond = predictor(cond_embedding)
    predictor_time = time.perf_counter() - start

    print("\n=== Benchmark Results ===")
    print(f"ROBIN optimizer time: {robin_time:.3f} seconds")
    print(f"Predictor time:        {predictor_time:.6f} seconds")
    print(f"Speed-up factor:       {robin_time / predictor_time:.2f}x")
    print(f"Pred opt_acond shape:  {tuple(pred_opt_acond.shape)}")
    print(f"Opt_acond shape:       {tuple(opt_acond.shape)}")


if __name__ == "__main__":
    main()
