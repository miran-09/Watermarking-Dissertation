"""
Run ROBIN baseline and predictor attack sweeps for two prompt sets,
log every run, and collect all metrics into one CSV.

Expected files in the ROBIN project root:
- new_unseen_prompts_20.txt
- prompts_100.txt
- inject_wm_baseline.py
- inject_wm_with_predictor.py
- ckpts/optimized_wm5-30_embedding-step-50.pt

Usage:
    python run_attack_sweep.py
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

MODEL_ID = "runwayml/stable-diffusion-v1-5"
WM_PATH = "ckpts/optimized_wm5-30_embedding-step-50.pt"

BASELINE_SCRIPT = "inject_wm_baseline.py"
PREDICTOR_SCRIPT = "inject_wm_with_predictor.py"

OUT_ROOT = PROJECT_ROOT / "results" / "sweep_py"
LOG_ROOT = OUT_ROOT / "logs"
CSV_PATH = OUT_ROOT / "attack_sweep_results.csv"

PROMPT_SETS = [
    {"name": "unseen20", "prompt_file": "new_unseen_prompts_20.txt", "end": 20},
    {"name": "prompts100", "prompt_file": "prompts_100.txt", "end": 100},
]


@dataclass(frozen=True)
class Case:
    name: str
    args: Tuple[str, ...]


def case(name: str, *args: str) -> Case:
    return Case(name=name, args=tuple(args))


SINGLE_ATTACK_CASES: List[Case] = [
    case("jpeg_95", "--jpeg_ratio", "95"),
    case("jpeg_75", "--jpeg_ratio", "75"),
    case("jpeg_50", "--jpeg_ratio", "50"),
    case("jpeg_25", "--jpeg_ratio", "25"),

    # Cropping uses kept fraction.
    case("crop_5", "--crop_scale", "0.95", "--crop_ratio", "0.95"),
    case("crop_20", "--crop_scale", "0.80", "--crop_ratio", "0.80"),
    case("crop_40", "--crop_scale", "0.60", "--crop_ratio", "0.60"),
    case("crop_60", "--crop_scale", "0.40", "--crop_ratio", "0.40"),

    case("blur_3", "--gaussian_blur_r", "3"),
    case("blur_5", "--gaussian_blur_r", "5"),
    case("blur_9", "--gaussian_blur_r", "9"),
    case("blur_13", "--gaussian_blur_r", "13"),

    case("rot_5", "--rotation_angle", "5"),
    case("rot_15", "--rotation_angle", "15"),
    case("rot_30", "--rotation_angle", "30"),
    case("rot_45", "--rotation_angle", "45"),

    case("noise_01", "--noise_std", "0.01"),
    case("noise_03", "--noise_std", "0.03"),
    case("noise_05", "--noise_std", "0.05"),
    case("noise_10", "--noise_std", "0.10"),

    case("cj_12", "--color_jitter_brightness", "1.2"),
    case("cj_2", "--color_jitter_brightness", "2"),
    case("cj_4", "--color_jitter_brightness", "4"),
    case("cj_6", "--color_jitter_brightness", "6"),
]

CHAIN_ATTACK_CASES: List[Case] = [
    case("jpeg75_crop20", "--jpeg_ratio", "75", "--crop_scale", "0.80", "--crop_ratio", "0.80"),
    case("blur5_noise03", "--gaussian_blur_r", "5", "--noise_std", "0.03"),
    case("rot15_jpeg50", "--rotation_angle", "15", "--jpeg_ratio", "50"),
    case("crop40_blur9", "--crop_scale", "0.60", "--crop_ratio", "0.60", "--gaussian_blur_r", "9"),
]


ATTACK_BLOCK_RE = re.compile(
    r"attack:\s*(?P<attack>[^\r\n]+)\r?\n"
    r"auc:\s*(?P<auc>[-0-9.eE]+),\s*acc:\s*(?P<acc>[-0-9.eE]+),\s*TPR@1%FPR:\s*(?P<tpr>[-0-9.eE]+)\r?\n"
    r"mse_mean:\s*(?P<mse>[-0-9.eE]+),\s*w_mse_mean:\s*(?P<wmse>[-0-9.eE]+)",
    re.MULTILINE,
)

META_RE = re.compile(
    r"steps:\s*(?P<steps>\d+),\s*radius:\s*(?P<radius>[^,]+),\s*wm_seed:\s*(?P<wm_seed>\d+),\s*opt wi&opt wt"
)

QUALITY_RE = re.compile(
    r"psnr:\s*(?P<psnr>[-0-9.eE]+),\s*ssim:\s*(?P<ssim>[-0-9.eE]+),\s*msssim:\s*(?P<msssim>[-0-9.eE]+)"
)


def ensure_dirs() -> None:
    (OUT_ROOT / "logs" / "baseline").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "logs" / "predictor").mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def build_base_args(prompt_file: str, end: int) -> List[str]:
    return [
        "--dataset", "custom",
        "--prompt_file", prompt_file,
        "--start", "0",
        "--end", str(end),
        "--model_id", MODEL_ID,
        "--wm_path", WM_PATH,
    ]


def build_cmd(script_path: str, prompt_file: str, end: int, extra_args: Sequence[str]) -> List[str]:
    return [PYTHON, script_path, *build_base_args(prompt_file, end), *extra_args]


def run_command(cmd: Sequence[str], log_path: Path) -> Tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        list(cmd),
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    lines: List[str] = []
    assert proc.stdout is not None
    with log_path.open("w", encoding="utf-8") as f:
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            lines.append(line)

    return_code = proc.wait()
    return return_code, "".join(lines)


def parse_log(text: str) -> Tuple[Dict[str, str], Dict[str, str], List[Dict[str, str]]]:
    meta_match = META_RE.search(text)
    quality_match = QUALITY_RE.search(text)
    attack_matches = list(ATTACK_BLOCK_RE.finditer(text))

    meta = meta_match.groupdict() if meta_match else {}
    quality = quality_match.groupdict() if quality_match else {}

    rows: List[Dict[str, str]] = []
    for m in attack_matches:
        row = m.groupdict()
        row.update(meta)
        row.update(quality)
        rows.append(row)

    return meta, quality, rows


def append_csv(rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        return

    fieldnames = [
        "prompt_set",
        "method",
        "case_name",
        "attack",
        "steps",
        "radius",
        "wm_seed",
        "auc",
        "acc",
        "tpr_at_1pct_fpr",
        "mse_mean",
        "w_mse_mean",
        "psnr",
        "ssim",
        "msssim",
        "status",
        "return_code",
        "log_file",
    ]

    write_header = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_case(prompt_set: Dict[str, object], method: str, case_obj: Case) -> None:
    prompt_file = str(prompt_set["prompt_file"])
    end = int(prompt_set["end"])
    prompt_set_name = str(prompt_set["name"])

    script_path = BASELINE_SCRIPT if method == "baseline" else PREDICTOR_SCRIPT
    log_dir = LOG_ROOT / method
    log_path = log_dir / f"{prompt_set_name}_{case_obj.name}.txt"

    cmd = build_cmd(script_path, prompt_file, end, case_obj.args)

    print()
    print("#" * 60)
    print(f"PROMPT SET: {prompt_set_name}")
    print(f"METHOD: {method}")
    print(f"CASE: {case_obj.name}")
    print(f"LOG: {log_path}")
    print("#" * 60)

    return_code, text = run_command(cmd, log_path)

    meta, quality, parsed_rows = parse_log(text)

    if return_code != 0:
        failure_row = {
            "prompt_set": prompt_set_name,
            "method": method,
            "case_name": case_obj.name,
            "attack": "",
            "steps": meta.get("steps", ""),
            "radius": meta.get("radius", ""),
            "wm_seed": meta.get("wm_seed", ""),
            "auc": "",
            "acc": "",
            "tpr_at_1pct_fpr": "",
            "mse_mean": "",
            "w_mse_mean": "",
            "psnr": quality.get("psnr", ""),
            "ssim": quality.get("ssim", ""),
            "msssim": quality.get("msssim", ""),
            "status": "failed",
            "return_code": str(return_code),
            "log_file": str(log_path),
        }
        append_csv([failure_row])
        raise RuntimeError(f"Experiment failed: {prompt_set_name} / {method} / {case_obj.name}")

    csv_rows: List[Dict[str, str]] = []
    for r in parsed_rows:
        csv_rows.append(
            {
                "prompt_set": prompt_set_name,
                "method": method,
                "case_name": case_obj.name,
                "attack": r.get("attack", ""),
                "steps": r.get("steps", ""),
                "radius": r.get("radius", ""),
                "wm_seed": r.get("wm_seed", ""),
                "auc": r.get("auc", ""),
                "acc": r.get("acc", ""),
                "tpr_at_1pct_fpr": r.get("tpr", ""),
                "mse_mean": r.get("mse", ""),
                "w_mse_mean": r.get("wmse", ""),
                "psnr": r.get("psnr", ""),
                "ssim": r.get("ssim", ""),
                "msssim": r.get("msssim", ""),
                "status": "ok",
                "return_code": str(return_code),
                "log_file": str(log_path),
            }
        )

    append_csv(csv_rows)
    print(f"Saved {len(csv_rows)} rows to CSV.")


def main() -> int:
    os.chdir(PROJECT_ROOT)
    ensure_dirs()

    if not Path(PROMPT_SETS[0]["prompt_file"]).exists():
        print(f"Missing prompt file: {PROMPT_SETS[0]['prompt_file']}")
        return 1
    if not Path(PROMPT_SETS[1]["prompt_file"]).exists():
        print(f"Missing p rompt file: {PROMPT_SETS[1]['prompt_file']}")
        return 1

    total_runs = len(PROMPT_SETS) * (len(SINGLE_ATTACK_CASES) + len(CHAIN_ATTACK_CASES)) * 2
    completed = 0

    for prompt_set in PROMPT_SETS:
        for case_obj in SINGLE_ATTACK_CASES:
            for method in ("baseline", "predictor"):
                completed += 1
                print(f"\n[{completed}/{total_runs}] {prompt_set['name']} | {method} | {case_obj.name}")
                run_case(prompt_set, method, case_obj)

        for case_obj in CHAIN_ATTACK_CASES:
            for method in ("baseline", "predictor"):
                completed += 1
                print(f"\n[{completed}/{total_runs}] {prompt_set['name']} | {method} | {case_obj.name}")
                run_case(prompt_set, method, case_obj)

    print()
    print("All attack sweeps completed.")
    print(f"CSV saved to: {CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
