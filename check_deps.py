"""
check_deps.py — verify all project dependencies are installed.

Run with:
    python check_deps.py
"""

import importlib
import sys


DEPS = [
    # (import_name, pip_name, notes)
    ("torch",           "torch",            "Install from https://pytorch.org for your platform"),
    ("torchvision",     "torchvision",      "Install alongside torch"),
    ("numpy",           "numpy",            None),
    ("scipy",           "scipy",            None),
    ("einops",          "einops",           None),
    ("transformers",    "git+https://github.com/huggingface/transformers", "Must be git install for SAM2/SAM3 support"),
    ("huggingface_hub", "huggingface_hub",  None),
    ("timm",            "timm",             None),
    ("PIL",             "pillow",           None),
    ("cv2",             "opencv-python",    None),
    ("pycocotools",     "pycocotools",      None),
    ("tqdm",            "tqdm",             None),
    ("psutil",          "psutil",           None),
]


def check():
    ok      = []
    missing = []

    for import_name, pip_name, note in DEPS:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "?")
            ok.append((import_name, version))
        except ImportError:
            missing.append((import_name, pip_name, note))

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'Package':<20} {'Version'}")
    print("-" * 35)
    for name, version in ok:
        print(f"  {name:<18} {version}")

    if missing:
        print(f"\nMISSING ({len(missing)}):")
        for import_name, pip_name, note in missing:
            print(f"  ✗ {import_name}")
            print(f"      install: pip install {pip_name}")
            if note:
                print(f"      note:    {note}")

        print("\nTo install everything at once (except torch and transformers):")
        print("  pip install -r requirements.txt")
        print("  pip install git+https://github.com/huggingface/transformers")
        sys.exit(1)
    else:
        print(f"\nAll {len(ok)} dependencies OK.")

    # ── Extra: check MPS / CUDA ───────────────────────────────────────────────
    try:
        import torch
        if torch.backends.mps.is_available():
            print("GPU: Apple MPS available.")
        elif torch.cuda.is_available():
            print(f"GPU: CUDA available ({torch.cuda.get_device_name(0)}).")
        else:
            print("GPU: None (CPU only — inference will be slow).")
    except Exception:
        pass


if __name__ == "__main__":
    check()
