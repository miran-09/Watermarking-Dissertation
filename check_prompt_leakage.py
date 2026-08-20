from pathlib import Path

def read_prompts(path: str) -> set[str]:
    return {p.strip() for p in Path(path).read_text(encoding="utf-8").splitlines() if p.strip()}

train = read_prompts("prompts_train.txt")
val = read_prompts("prompts_val.txt")
test = read_prompts("prompts_test.txt")

overlap_tv = train & val
overlap_tt = train & test
overlap_vt = val & test

print("trainâˆ©val:", len(overlap_tv))
print("trainâˆ©test:", len(overlap_tt))
print("valâˆ©test:", len(overlap_vt))

if overlap_tv or overlap_tt or overlap_vt:
    raise SystemExit("Leakage found: some prompts overlap.")

print("No prompt overlap found.")
