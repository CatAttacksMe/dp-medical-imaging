"""Study A training: full end-to-end fine-tuning of an ImageNet-pretrained
DenseNet-121 on NIH ChestX-ray14 pneumothorax labels, across the 90/10,
70/30, 50/50 sex-imbalance sweep. Writes one predictions CSV per run.

See CLAUDE.md, "Study A — Baseline" (Design Decisions / Backbone
Initialization / Seed Replication) for the rules this module implements.
Hyperparameters below (LEARNING_RATE, MAX_EPOCHS, EARLY_STOP_PATIENCE,
BATCH_SIZE, GRAD_CLIP_NORM) are engineering defaults within CLAUDE.md's
constraints (LR in [1e-5, 1e-4], identical across arms) rather than
separately-decided values.

GPU training is made bit-reproducible via torch.use_deterministic_algorithms
(see configure_determinism) so seed 42 is an actual, not just nominal,
reproducibility anchor, per CLAUDE.md's Seeding decision.
"""

import os

# Required for torch.use_deterministic_algorithms(True) on CUDA — must be
# set before the first CUDA op. See configure_determinism().
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision.models import DenseNet121_Weights, densenet121

import data_loading as dl

SEED = dl.SEED  # canonical seed: patient split, undersampling, and each
                 # arm's canonical training run
ARMS = ["90_10", "70_30", "50_50"]

# Only the oracle-gated 90/10 arm is replicated (5 seeds: canonical 42 +
# 43-46) to check its gap isn't a one-run fluke. 70/30 and 50/50 aren't
# independently oracle-gated, so they stay single-run. See CLAUDE.md,
# Study A Seed Replication.
REPLICATION_SEEDS = {"90_10": [42, 43, 44, 45, 46]}

LEARNING_RATE = 3e-5
MAX_EPOCHS = 20
EARLY_STOP_PATIENCE = 5
BATCH_SIZE = 32
NUM_WORKERS = 4
GRAD_CLIP_NORM = 1.0

RESULTS_DIR = os.path.join(dl.REPO_ROOT, "results", "study_a")
REPLICATION_DIR = os.path.join(RESULTS_DIR, "seed_replication")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
# .pth is gitignored by extension regardless of directory (see CLAUDE.md,
# Conventions: checkpoints are never committed).
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")


def configure_determinism():
    """Process-wide; call once. Makes GPU training bit-reproducible from a
    fixed seed (slower than the non-deterministic defaults) — see
    CLAUDE.md, Study A Seeding.
    """
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model():
    """ImageNet-pretrained DenseNet-121, full fine-tuning (all params
    trainable by default — nothing frozen), single-logit head replacing
    the 1000-class ImageNet classifier for binary pneumothorax prediction.
    """
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


def make_loader(df, batch_size=BATCH_SIZE, shuffle=False, generator=None):
    dataset = dl.NIHPneumothoraxDataset(df)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        generator=generator if shuffle else None,
        pin_memory=torch.cuda.is_available(),
    )


def _patient_level_auc(patient_ids, labels, scores):
    agg = pd.DataFrame(
        {"patient_id": patient_ids, "label": labels, "score": scores}
    ).groupby("patient_id").agg(label=("label", "max"), score=("score", "mean"))
    return roc_auc_score(agg["label"], agg["score"])


@torch.no_grad()
def evaluate(model, loader, device):
    """Runs inference over a loader; returns patient-level AUC (labels
    binarized, scores averaged across a patient's images before ranking —
    see CLAUDE.md, Multi-image aggregation) plus the raw per-image results.
    """
    model.eval()
    patient_ids, image_ids, labels, scores = [], [], [], []
    for batch in loader:
        logits = model(batch["image"].to(device)).squeeze(1)
        scores.extend(torch.sigmoid(logits).cpu().numpy())
        labels.extend(batch["label"].numpy())
        patient_ids.extend(batch["patient_id"])
        image_ids.extend(batch["image_id"])
    auc = _patient_level_auc(patient_ids, labels, scores)
    per_image = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "image_id": image_ids,
            "true_label": np.array(labels, dtype=int),
            "predicted_score": scores,
        }
    )
    return auc, per_image


def _output_path(arm, run_seed):
    """Canonical seed=42 run writes the frozen predictions_{arm}.csv;
    replicate seeds write to seed_replication/, outside Study B's read
    contract (see CLAUDE.md, Study A Seed Replication).
    """
    if run_seed == SEED:
        return os.path.join(RESULTS_DIR, f"predictions_{arm}.csv")
    return os.path.join(REPLICATION_DIR, f"predictions_{arm}_seed{run_seed}.csv")


def train_one_arm(arm, metadata, split_df, device, run_seed=SEED, max_epochs=MAX_EPOCHS, force=False):
    output_path = _output_path(arm, run_seed)
    if os.path.exists(output_path) and not force:
        print(f"[{arm} seed={run_seed}] {output_path} already exists, skipping (--force to redo)")
        return pd.read_csv(output_path, dtype={"patient_id": str, "image_id": str})

    # Seed covers weight init (the new classifier head) and data-loader
    # shuffling for this run — see CLAUDE.md, Seeding / Seed Replication.
    set_seed(run_seed)

    # Patient split and per-arm undersampling always use the canonical
    # seed, regardless of run_seed — replicate runs must train on the
    # *same* patients, varying only weight init/batch order, or they'd
    # conflate composition noise with training noise.
    train_df = dl.build_training_set(metadata, split_df, arm, seed=SEED)
    val_df = dl.get_fixed_eval_set(metadata, split_df, "val")
    test_df = dl.get_fixed_eval_set(metadata, split_df, "test")

    pos_weight = dl.compute_pos_weight(train_df).to(device)

    train_generator = torch.Generator().manual_seed(run_seed)
    train_loader = make_loader(train_df, shuffle=True, generator=train_generator)
    val_loader = make_loader(val_df, shuffle=False)
    test_loader = make_loader(test_df, shuffle=False)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"densenet121_{arm}_seed{run_seed}.pth")
    log_path = os.path.join(LOG_DIR, f"train_log_{arm}_seed{run_seed}.csv")

    best_val_auc = -1.0
    epochs_without_improvement = 0
    log_rows = []

    for epoch in range(max_epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches

        val_auc, _ = evaluate(model, val_loader, device)
        if np.isnan(val_auc):
            raise RuntimeError(
                f"[{arm} seed={run_seed}] val AUC is NaN at epoch {epoch} — training diverged"
            )
        print(
            f"[{arm} seed={run_seed}] epoch {epoch}: "
            f"train_loss={train_loss:.4f} val patient-AUC={val_auc:.4f}"
        )
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc})

        if val_auc > best_val_auc:
            best_val_auc, epochs_without_improvement = val_auc, 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"[{arm} seed={run_seed}] early stopping at epoch {epoch}")
                break

    pd.DataFrame(log_rows).to_csv(log_path, index=False)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_auc, predictions = evaluate(model, test_loader, device)
    print(f"[{arm} seed={run_seed}] best val patient-AUC={best_val_auc:.4f}, test patient-AUC={test_auc:.4f}")

    predictions = predictions.merge(
        metadata[["patient_id", "image_id", "true_sex", "true_age"]],
        on=["patient_id", "image_id"],
        how="left",
    )
    predictions = predictions[
        ["patient_id", "image_id", "true_label", "predicted_score", "true_sex", "true_age"]
    ]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"[{arm} seed={run_seed}] wrote {output_path} ({len(predictions)} rows)")
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS + ["all"], default="all")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument(
        "--force", action="store_true", help="redo runs even if their output CSV already exists"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    configure_determinism()

    metadata = dl.load_metadata()
    split_df = dl.get_patient_split(metadata)

    arms = ARMS if args.arm == "all" else [args.arm]
    for arm in arms:
        for run_seed in REPLICATION_SEEDS.get(arm, [SEED]):
            train_one_arm(
                arm, metadata, split_df, device,
                run_seed=run_seed, max_epochs=args.max_epochs, force=args.force,
            )


if __name__ == "__main__":
    main()
