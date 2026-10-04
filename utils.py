import json, random, math
from pathlib import Path
import numpy as np
import torch
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def ensure_dirs(root):
    root = Path(root)
    for sub in ("checkpoints", "logs", "figures"): (root / sub).mkdir(parents=True, exist_ok=True)
    return root
def append_jsonl(path, row):
    def clean(value):
        if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [clean(v) for v in value]
        if isinstance(value,float) and not math.isfinite(value): return None
        return value
    with open(path, "a", encoding="utf-8") as f: f.write(json.dumps(clean(row), allow_nan=False) + "\n")
def save_checkpoint(path, **items):
    Path(path).parent.mkdir(parents=True, exist_ok=True); torch.save(items, path)
