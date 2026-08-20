import argparse
import torch
import torch.nn.functional as F
from diffusers import DPMSolverMultistepScheduler
from inverse_stable_diffusion import InversableStableDiffusionPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-2-1-base")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision="main",
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    opt_acond = ckpt["opt_acond"].to(device)

    empty_emb = pipe.get_text_embedding("").to(device)

    diff = opt_acond - empty_emb

    cos = F.cosine_similarity(
        opt_acond.flatten(1), empty_emb.flatten(1), dim=1
    ).item()
    mse = torch.mean(diff ** 2).item()
    mae = torch.mean(torch.abs(diff)).item()
    max_abs = torch.max(torch.abs(diff)).item()

    print("Checkpoint:", args.ckpt)
    print("opt_acond shape:", tuple(opt_acond.shape))
    print("empty_emb shape:", tuple(empty_emb.shape))
    print("Cosine similarity:", cos)
    print("MSE:", mse)
    print("MAE:", mae)
    print("Max abs diff:", max_abs)


if __name__ == "__main__":
    main()
