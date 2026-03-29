"""Unified training entry point for biovision-lab.

Usage
-----
    python train.py --config configs/train.yaml
    python train.py --config configs/train.yaml --task detection
    python train.py --config configs/train.yaml --task segmentation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="biovision-lab – unified training entry point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["detection", "segmentation"],
        default=None,
        help="Override the task defined in the config file.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------

def run_detection(cfg: dict) -> None:
    """Train a YOLO / RT-DETR detection model."""
    from src.core.yolo_model import YOLODetector

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    logging_cfg = cfg.get("logging", {})

    model_variant = cfg.get("model_variant", model_cfg.get("model_variant", "yolov8n"))
    detector = YOLODetector(
        model_variant=model_variant,
        num_classes=model_cfg.get("num_classes", 2),
        pretrained=model_cfg.get("pretrained", True),
    )

    data_yaml = data_cfg.get("data_yaml", "data/dataset.yaml")
    detector.train(
        data_yaml=data_yaml,
        epochs=train_cfg.get("epochs", 100),
        batch=train_cfg.get("batch_size", 8),
        imgsz=data_cfg.get("image_size", 512),
        project=logging_cfg.get("save_dir", "experiments"),
        name="detection_run",
    )

    export_cfg = cfg.get("deployment", {})
    for fmt in export_cfg.get("formats", ["onnx"]):
        out = detector.export(format=fmt, export_dir=export_cfg.get("export_dir", "deployments"))
        print(f"[INFO] Model exported to {out}")


def run_segmentation(cfg: dict) -> None:
    """Train a U-Net segmentation model with MONAI."""
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    from src.core.unet_model import UNetSegmentor
    from src.loss.dice_loss import DiceLoss
    from src.utils.metrics import compute_dice_score

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    logging_cfg = cfg.get("logging", {})

    segmentor = UNetSegmentor(
        spatial_dims=2,
        in_channels=model_cfg.get("input_channels", 1),
        out_channels=model_cfg.get("num_classes", 2),
    )
    model = segmentor.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = DiceLoss(softmax=True, include_background=False)
    lr = train_cfg.get("learning_rate", 1e-3)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=train_cfg.get("weight_decay", 5e-4))

    save_dir = Path(logging_cfg.get("save_dir", "experiments"))
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = train_cfg.get("epochs", 100)
    print(f"[INFO] Starting segmentation training for {epochs} epochs on {device}.")
    print(f"[INFO] Checkpoints will be saved to {save_dir}/")

    # NOTE: Replace the placeholder DataLoader below with your real dataset.
    # Example:
    #   from src.data_loader.nifti_loader import NiftiLoader
    #   dataset = YourMedicalDataset(cfg["data"]["train_dir"])
    #   loader  = DataLoader(dataset, batch_size=train_cfg["batch_size"], ...)

    print("[WARN] No real dataset configured – running a minimal smoke-test loop.")
    dummy_x = torch.zeros(2, model_cfg.get("input_channels", 1), 64, 64)
    dummy_y = torch.zeros(2, 64, 64, dtype=torch.long)
    loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=2)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % logging_cfg.get("log_interval", 10) == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"[Epoch {epoch:>4d}/{epochs}] loss={avg_loss:.4f}")

    ckpt_path = save_dir / "segmentation_final.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[INFO] Final checkpoint saved to {ckpt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)

    task = args.task or cfg.get("task", "detection")
    print(f"[INFO] Task: {task}")
    print(f"[INFO] Config: {args.config}")

    if task == "detection":
        run_detection(cfg)
    elif task == "segmentation":
        run_segmentation(cfg)
    else:
        print(f"[ERROR] Unknown task: {task}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
