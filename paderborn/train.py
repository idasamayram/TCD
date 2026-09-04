"""Train the CNN1D bearing fault classifier on data produced by preprocessing.py.

Example:
    python train.py --data-dir ./processed --out-dir ./runs/exp1 --epochs 50
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PUBearingWindows, load_meta
from model import CNN1D


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid div by zero for absent classes
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_n += x.size(0)
    torch.set_grad_enabled(True)
    return total_loss / total_n, total_correct / total_n


def evaluate_test(model, loader, device, class_names):
    from sklearn.metrics import classification_report, confusion_matrix
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_targets.append(y.numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    report = classification_report(targets, preds, target_names=class_names, output_dict=True, zero_division=0)
    report_str = classification_report(targets, preds, target_names=class_names, zero_division=0)
    cm = confusion_matrix(targets, preds).tolist()
    return report, report_str, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Dir with train.npz/val.npz/test.npz/meta.json")
    parser.add_argument("--out-dir", type=str, default="./runs/exp1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on val loss.")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-class-weights", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(args.data_dir)
    class_names = meta["classes"]
    num_classes = len(class_names)

    train_ds = PUBearingWindows(Path(args.data_dir) / "train.npz")
    val_ds = PUBearingWindows(Path(args.data_dir) / "val.npz")
    test_ds = PUBearingWindows(Path(args.data_dir) / "test.npz")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CNN1D(in_channels=1, num_classes=num_classes,
                   base_channels=args.base_channels, dropout=args.dropout).to(device)

    if args.no_class_weights:
        weights = None
    else:
        weights = compute_class_weights(train_ds.y, num_classes).to(device)
        print(f"Class weights ({class_names}): {weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})
        print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} acc {train_acc:.4f} "
              f"| val_loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (best epoch was {best_epoch}).")
                break

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # reload best checkpoint and evaluate on held-out test set
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    report, report_str, cm = evaluate_test(model, test_loader, device, class_names)
    print("\nTest set performance (best val-loss checkpoint):")
    print(report_str)
    print("Confusion matrix (rows=true, cols=pred):")
    print(np.array(cm))

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({"classification_report": report, "confusion_matrix": cm,
                    "class_names": class_names, "best_epoch": best_epoch}, f, indent=2)

    # save everything needed to reload the model later (e.g. for LRP/zennit)
    config = vars(args)
    config["class_names"] = class_names
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved best_model.pt, history.json, test_metrics.json, run_config.json to {out_dir}")


if __name__ == "__main__":
    main()
