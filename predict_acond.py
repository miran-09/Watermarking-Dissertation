import argparse
import torch
import torch.nn as nn

from diffusers import DPMSolverMultistepScheduler
from inverse_stable_diffusion import InversableStableDiffusionPipeline


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


def load_pipeline(model_id: str, device: str):
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        model_id, subfolder="scheduler"
    )
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        revision="main",
    )
    return pipe.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--predictor_path", default="prompt_predictor.pt")
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--out", default="predicted_opt_acond.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    predictor = load_predictor(args.predictor_path, device)
    pipe = load_pipeline(args.model_id, device)

    with torch.no_grad():
        cond_embedding = pipe.get_text_embedding(args.prompt).to(device).float()
        pred_opt_acond = predictor(cond_embedding)

    print("prompt:", args.prompt)
    print("cond_embedding shape:", tuple(cond_embedding.shape))
    print("pred_opt_acond shape:", tuple(pred_opt_acond.shape))

    torch.save(
        {
            "prompt": args.prompt,
            "cond_embedding": cond_embedding.detach().cpu(),
            "pred_opt_acond": pred_opt_acond.detach().cpu(),
        },
        args.out,
    )
    print("Saved to:", args.out)


if __name__ == "__main__":
    main()
