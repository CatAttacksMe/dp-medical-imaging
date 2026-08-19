"""Study A data loading: patient-level split, sex-imbalance sweep, and the
pneumothorax-labeled image dataset for the NIH ChestX-ray14 corpus.

See CLAUDE.md, "Study A — Baseline" and its Design Decisions / Backbone
Initialization subsections for the rules this module implements.
"""

import os

import numpy as np
import pandas as pd
import skimage.transform
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

SEED = 42
LABEL = "Pneumothorax"
IMAGE_SIZE = 224

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Fixed minority sex across the sweep — see CLAUDE.md, Study A Design
# Decisions ("Minority sex = female").
MAJORITY_SEX = "M"
MINORITY_SEX = "F"
IMBALANCE_RATIOS = {
    "90_10": {MAJORITY_SEX: 0.90, MINORITY_SEX: 0.10},
    "70_30": {MAJORITY_SEX: 0.70, MINORITY_SEX: 0.30},
    "50_50": {MAJORITY_SEX: 0.50, MINORITY_SEX: 0.50},
}

# ImageNet stats, not torchxrayvision's normalize() range — required for
# the ImageNet-pretrained backbone's conv1 weights (see CLAUDE.md, Study A
# Backbone Initialization).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_METADATA_CSV = os.path.join(
    REPO_ROOT, "data", "raw", "Data_Entry_2017_v2020.csv"
)
DEFAULT_IMAGE_DIR = os.path.join(REPO_ROOT, "data", "raw")
PATIENT_SPLIT_PATH = os.path.join(REPO_ROOT, "results", "study_a", "patient_split.csv")

# Split-sensitivity seeds — distinct from the 42-46 range used for
# weight-init/data-loader-shuffle seeding, so the two kinds of seed are
# never confused. See CLAUDE.md, Study A Split Sensitivity.
SPLIT_SENSITIVITY_SEEDS = [101, 102, 103]
SPLIT_SENSITIVITY_DIR = os.path.join(REPO_ROOT, "results", "study_a", "split_sensitivity")

# Exploratory sample-size-sensitivity run: a single fixed N_total well below
# the canonical majority-pool-sized budget, used to check whether the
# ratio's effect on the subgroup gap looks different when no arm has
# "abundant" minority data. See CLAUDE.md, Study A Sample-Size Sensitivity.
N_SENSITIVITY_TOTAL = 5000
N_SENSITIVITY_DIR = os.path.join(REPO_ROOT, "results", "study_a", "n_sensitivity")


def load_metadata(csv_path=DEFAULT_METADATA_CSV):
    """Loads the NIH metadata CSV into the columns Study A needs."""
    raw = pd.read_csv(csv_path)
    return pd.DataFrame(
        {
            "patient_id": raw["Patient ID"].astype(str),
            "image_id": raw["Image Index"],
            "true_label": raw["Finding Labels"]
            .str.split("|")
            .apply(lambda labels: int(LABEL in labels)),
            "true_sex": raw["Patient Sex"],
            "true_age": raw["Patient Age"].astype(int),
        }
    )


def _generate_split_df(patient_ids, seed):
    """Patient-level 70/15/15 split assignment for one seed. Shared by the
    canonical split and the split-sensitivity alternate splits — the two
    differ only in which seed and output path they use.

    Sorts patient_ids before permuting so the split is a function of the
    seed and the patient-ID set alone, not of metadata's row order (which
    otherwise reflects the source CSV's arbitrary order and isn't a
    reproducibility guarantee across re-downloads/mirrors).
    """
    shuffled = np.random.RandomState(seed).permutation(np.sort(patient_ids))

    n = len(shuffled)
    n_train = int(round(n * SPLIT_FRACTIONS["train"]))
    n_val = int(round(n * SPLIT_FRACTIONS["val"]))
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)

    return pd.DataFrame({"patient_id": shuffled, "split": splits})


def get_patient_split(metadata, split_path=PATIENT_SPLIT_PATH, seed=SEED):
    """Loads the frozen 70/15/15 patient-level split, generating it once on
    the first call. An existing split_path is loaded as-is and never
    regenerated — it's the reproducibility anchor for the whole study.
    """
    if os.path.exists(split_path):
        return pd.read_csv(split_path, dtype={"patient_id": str})

    split_df = _generate_split_df(metadata["patient_id"].unique(), seed)
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    split_df.to_csv(split_path, index=False)
    return split_df


def get_alternate_patient_split(metadata, seed, split_dir=SPLIT_SENSITIVITY_DIR):
    """Loads (or generates, on first call) one additional patient-level
    70/15/15 split for the split-sensitivity robustness check — a new
    draw of which patients land in train/val/test, independent of the
    canonical patient_split.csv, which this never touches or regenerates.
    See CLAUDE.md, Study A Split Sensitivity.
    """
    split_path = os.path.join(split_dir, f"patient_split_seed{seed}.csv")
    if os.path.exists(split_path):
        return pd.read_csv(split_path, dtype={"patient_id": str})

    split_df = _generate_split_df(metadata["patient_id"].unique(), seed)
    os.makedirs(split_dir, exist_ok=True)
    split_df.to_csv(split_path, index=False)
    return split_df


def _patient_sex(metadata):
    """One row per patient_id -> true_sex (constant per patient)."""
    return metadata.drop_duplicates("patient_id").set_index("patient_id")["true_sex"]


def build_training_set(metadata, split_df, ratio_name, seed=SEED, n_total=None):
    """Undersampled training-set images for one imbalance arm.

    Undersampling is patient-level (whole patients dropped, never
    individual images). By default, drawn from a fixed budget shared
    across the three canonical arms: N_total = min(available_majority,
    2 * available_minority), so the arms are comparable and only
    composition varies. Passing an explicit n_total overrides this
    budget — used only by the exploratory sample-size-sensitivity runs
    (see CLAUDE.md, Study A Sample-Size Sensitivity), which check
    whether the ratio's effect on the subgroup gap looks different at a
    smaller, fixed training-set size than the canonical sweep's.
    """
    ratio = IMBALANCE_RATIOS[ratio_name]
    train_patients = split_df.loc[split_df["split"] == "train", "patient_id"]
    sex_by_patient = _patient_sex(metadata).reindex(train_patients)

    majority_pool = sex_by_patient[sex_by_patient == MAJORITY_SEX].index.to_numpy()
    minority_pool = sex_by_patient[sex_by_patient == MINORITY_SEX].index.to_numpy()

    if n_total is None:
        n_total = min(len(majority_pool), 2 * len(minority_pool))
    n_minority = int(round(n_total * ratio[MINORITY_SEX]))
    n_majority = n_total - n_minority
    assert n_majority <= len(majority_pool) and n_minority <= len(minority_pool), (
        f"n_total={n_total} infeasible for {ratio_name}: needs {n_majority} majority / "
        f"{n_minority} minority patients, pools have {len(majority_pool)} / {len(minority_pool)}"
    )

    rng = np.random.RandomState(seed)
    chosen = np.concatenate(
        [
            rng.choice(majority_pool, size=n_majority, replace=False),
            rng.choice(minority_pool, size=n_minority, replace=False),
        ]
    )
    return metadata[metadata["patient_id"].isin(chosen)].reset_index(drop=True)


def get_fixed_eval_set(metadata, split_df, split_name):
    """Val or test images for a split: fixed and sex-representative,
    identical across all three imbalance arms — only training composition
    varies per arm.
    """
    assert split_name in ("val", "test")
    patients = split_df.loc[split_df["split"] == split_name, "patient_id"]
    return metadata[metadata["patient_id"].isin(patients)].reset_index(drop=True)


def compute_pos_weight(df, label_col="true_label"):
    """Inverse-frequency positive class weight for BCEWithLogitsLoss —
    same formula applied identically across all three imbalance arms.
    """
    n_pos = (df[label_col] == 1).sum()
    n_neg = (df[label_col] == 0).sum()
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


class NIHPneumothoraxDataset(Dataset):
    """Pneumothorax-labeled NIH ChestX-ray14 images, preprocessed for the
    ImageNet-pretrained backbone: resized to 224x224 with
    skimage.transform.resize, then normalized with ImageNet mean/std as
    3-channel input. Deliberately does not depend on torchxrayvision —
    not for weights, and not for preprocessing either — to keep Study A
    structurally free of any contact with the chest-X-ray-pretrained
    ecosystem (see CLAUDE.md, Study A Backbone Initialization).
    """

    def __init__(self, df, image_dir=DEFAULT_IMAGE_DIR, image_size=IMAGE_SIZE):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.image_size = image_size
        self._normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def _load_image(self, image_id):
        img = Image.open(os.path.join(self.image_dir, image_id)).convert("L")
        arr = np.array(img, dtype=np.float32)[None, :, :]  # (1, H, W), 0-255
        arr = skimage.transform.resize(
            arr,
            (1, self.image_size, self.image_size),
            mode="constant",
            preserve_range=True,
        ).astype(np.float32)
        arr = arr / 255.0  # (1, size, size), 0-1
        tensor = torch.from_numpy(arr).repeat(3, 1, 1)  # (3, size, size)
        return self._normalize(tensor)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "image": self._load_image(row["image_id"]),
            "label": torch.tensor(row["true_label"], dtype=torch.float32),
            "patient_id": row["patient_id"],
            "image_id": row["image_id"],
        }
