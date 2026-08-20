from pathlib import Path
import sys

new_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("new_unseen_prompts_20.txt")
base_files = [Path("prompts_train.txt"), Path("prompts_val.txt"), Path("prompts_test.txt")]

def read_set(p):
    return {x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()}

new_prompts = read_set(new_file)
print(f"New prompts: {len(new_prompts)}")

for bf in base_files:
    if not bf.exists():
        print(f"Missing base file: {bf}")
        continue
    base = read_set(bf)
    overlap = sorted(new_prompts & base)
    print(f"{bf.name}: overlap={len(overlap)}")
    for item in overlap[:5]:
        print("  -", item)

print("Done.")
