from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .common import device_from_system, load_config, read_splits, seed_everything
from .losses import DiceBCELoss
from .models import build_model
from .pipeline import make_dataset, make_eval_loader, make_train_loader


def epoch_pass(model, loader, criterion, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    for images, masks, _, _ in tqdm(loader, leave=False):
        images, masks = images.to(device), masks.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss = criterion(model(images), masks)
            if training:
                loss.backward()
                optimizer.step()
        total += loss.item()
    return total / len(loader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    splits = read_splits(config["data"]["split_file"])
    train_loader = make_train_loader(make_dataset(config, splits, "train", True), config)
    val_loader = make_eval_loader(make_dataset(config, splits, "validation"), config)
    device = device_from_system()
    model = build_model(config["model"]).to(device)
    criterion = DiceBCELoss()
    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    checkpoint = Path(config["training"]["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    results = Path(config["results_dir"])
    results.mkdir(parents=True, exist_ok=True)
    best, stale, history = float("inf"), 0, []

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss = epoch_pass(model, train_loader, criterion, device, optimizer)
        val_loss = epoch_pass(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save({"model": model.state_dict(), "config": config}, checkpoint)
        else:
            stale += 1
            if stale >= config["training"]["patience"]:
                print("Erken durduruldu.")
                break

    with (results / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader(); writer.writerows(history)
    plt.plot([x["epoch"] for x in history], [x["train_loss"] for x in history], label="train")
    plt.plot([x["epoch"] for x in history], [x["val_loss"] for x in history], label="validation")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(results / "loss_curve.png", dpi=160); plt.close()


if __name__ == "__main__":
    main()

