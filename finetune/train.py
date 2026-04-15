"""
Fine-tune YOLO on the THEKER 5-class dataset.

Steps:
  1. Starts from the project's existing yolo26n.pt weights.
  2. Freezes the first 10 backbone layers (feature extractor stays frozen,
     only the detection head and upper neck are trained).
  3. Trains for 50 epochs with early stopping.

Run:
  python finetune/train.py
  python finetune/train.py --epochs 30 --batch 8 --freeze 0   # unfreeze all
  python finetune/train.py --resume                            # resume last run

After training, best weights land at:
  runs/detect/theker_finetune/weights/best.pt

Update segment_module/segment.py MODEL_ID to point at those weights.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT        = Path(__file__).parent.parent
YAML        = ROOT / "finetune" / "dataset.yaml"
BASE_MODEL  = ROOT / "yolo26n.pt"
PROJECT     = ROOT / "runs" / "detect"
RUN_NAME    = "theker_finetune"
BEST_WEIGHTS = PROJECT / RUN_NAME / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",  type=int,   default=50)
    p.add_argument("--batch",   type=int,   default=16,
                   help="Batch size. Reduce to 8 or 4 if OOM.")
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--freeze",  type=int,   default=10,
                   help="Number of backbone layers to freeze (0 = train all).")
    p.add_argument("--lr",      type=float, default=0.001)
    p.add_argument("--workers", type=int,   default=4)
    p.add_argument("--device",  type=str,   default=None,
                   help="Device: 'cpu', '0', 'mps'. Auto-detected if omitted.")
    p.add_argument("--resume",  action="store_true",
                   help="Resume the last interrupted run.")
    p.add_argument("--force",   action="store_true",
                   help="Retrain even if best.pt already exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not YAML.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found at {YAML}\n"
            "Run: python finetune/convert_labels.py first."
        )

    # Skip training if weights already exist (unless --resume or --force)
    if BEST_WEIGHTS.exists() and not args.resume and not args.force:
        print(f"Fine-tuned weights already exist at:\n  {BEST_WEIGHTS}")
        print("\nSkipping training. Use --force to retrain from scratch.")
        print(f"\nTo use in the pipeline, MODEL_ID is already set automatically.")
        return

    if args.resume:
        last_run = PROJECT / RUN_NAME / "weights" / "last.pt"
        if not last_run.exists():
            raise FileNotFoundError(f"No checkpoint to resume: {last_run}")
        model = YOLO(str(last_run))
        print(f"Resuming from {last_run}")
    else:
        if not BASE_MODEL.exists():
            raise FileNotFoundError(
                f"Base model not found: {BASE_MODEL}\n"
                "Place yolo26n.pt in the project root."
            )
        model = YOLO(str(BASE_MODEL))
        print(f"Starting from {BASE_MODEL}")

    print(f"\nTraining config:")
    print(f"  dataset : {YAML}")
    print(f"  epochs  : {args.epochs}")
    print(f"  batch   : {args.batch}")
    print(f"  imgsz   : {args.imgsz}")
    print(f"  freeze  : {args.freeze} backbone layers")
    print(f"  lr0     : {args.lr}")
    print()

    model.train(
        data       = str(YAML),
        epochs     = args.epochs,
        batch      = args.batch,
        imgsz      = args.imgsz,
        freeze     = args.freeze,
        lr0        = args.lr,
        workers    = args.workers,
        device     = args.device,
        project    = str(PROJECT),
        name       = RUN_NAME,
        exist_ok   = True,
        patience   = 10,      # early stop after 10 epochs without improvement
        save       = True,
        plots      = True,
        resume     = args.resume,
    )

    best = PROJECT / RUN_NAME / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best}")
    print(f"\nTo use fine-tuned model in the pipeline, update segment_module/segment.py:")
    print(f'  MODEL_ID = "{best}"')


if __name__ == "__main__":
    main()
