import argparse, re, shutil, sys
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--heads", type=int, default=4, help="the --heads value you trained with (default 4)")
ap.add_argument("--path", default="model.pt")
a = ap.parse_args()

ck = torch.load(a.path, map_location="cpu")
if isinstance(ck, dict) and "cfg" in ck:
    print("This file is already in the new format. Nothing to do.")
    sys.exit(0)


vocab, dim = ck["tok.weight"].shape
block = ck["pos.weight"].shape[0]
layers = 1 + max(int(m.group(1)) for k in ck if (m := re.match(r"blocks\.(\d+)\.", k)))
if dim % a.heads:
    sys.exit(f"--heads {a.heads} doesn't divide the model width {dim}. Pass the value you trained with.")

cfg = dict(vocab=vocab, block=block, d=dim, n_layer=layers, n_head=a.heads)
shutil.copy(a.path, "model_old.pt")
torch.save({"cfg": cfg, "state": ck}, a.path)
print("Converted:", cfg)
print("Backup of the original saved as model_old.pt")