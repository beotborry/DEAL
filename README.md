# DEAL

Group robustness-aware MRI training pipeline for Alzheimer's disease (AD) prediction, combining 3D MRI with tabular clinical/demographic features via multiple fusion methods.

## Overview

DEAL supports:

- **Targets**: binary AD classification (`ad`), MCI-to-AD transition (`ad_transition`), and time-to-AD Cox regression (`ad_time_cox`)
- **MRI backbones**: MONAI ResNet, EfficientNet, Vision Transformer (ViT-T/S/B), with optional 2D sagittal view
- **Fusion methods**: feature concatenation, FiLM, FiLM+demographics, DAFT, tabular concatenation (with/without linear projection), tabular attention, and tabular-only baselines
- **Group Robustness**: Group DRO, SUBG, and age-based classifier decoupling

## Requirements

- Python 3
- PyTorch
- [TorchIO](https://github.com/fepegar/torchio) (MRI loading and transforms)
- MONAI (for `monai_resnet18` backbone)
- pandas, numpy, scipy, scikit-learn

Place data under `--data-dir` (default: `./data/`). The pipeline expects NIfTI MRI volumes (e.g. `brain_to_MNI_syn_n4.nii.gz`) and a CSV/table with subject IDs, labels, and tabular features (e.g. MMSE, CDRSB, ADAS11, age, gender, education, FAQ). Dataset splitting and target definitions follow the logic in `datasets/generic_dataset.py` and `datasets/mri_dataset.py`.

## AD/CN pretrained weights
Weights of pretrained models trained with AD/CN clssification are provided in https://drive.google.com/drive/folders/1VgpcKJDJJza9II9LzZVwy7OgQ96m8zel?usp=sharing. 

## Quick start

**Train (example: FiLM + demographics, AD transition, pretrained ResNet):**

```bash
python main.py \
  --method film_demo \
  --film-only-last \
  --film-aux-net mlp \
  --film-aux-net-act silu \
  --tabular-input mmse cdrsb adas11 age_dummy gender educat faq \
  --target ad_transition \
  --mri-arch monai_resnet18 \
  --resolution 1mm \
  --pretrained \
  --normalize \
  --epochs 30 \
  --batch-size 4 \
  --lr 1e-5 \
  --mri-modelpath /path/to/pretrained_mri.pt \
  --save-model \
  --date my_experiment
```

## Main options

| Category    | Arguments |
|------------|-----------|
| **Data**   | `--data-dir`, `--resolution` (1mm / 2mm), `--target` (ad / ad_transition / ad_time_cox), `--normalize`, `--merge-train-val`, `--sagittal` |
| **Backbone** | `--mri-arch` (efficientnet, vit_t, vit_s, vit_b, monai_resnet18), `--resnet-backbone`, `--efficientnet-backbone` |
| **Method** | `--method` (scratch, feature_concat, film, film_demo, daft, tab_concat, tab_concat_lin_proj, tab_attention, tabular_only) |
| **FiLM/DAFT** | `--film-aux-net`, `--film-aux-net-act`, `--film-only-last`, `--tabular-input`, `--daft-input` |
| **Optimization** | `--epochs`, `--batch-size`, `--lr`, `--optimizer`, `--scheduler`, `--weight-decay` |
| **Group Robustness** | `--group-dro`, `--mri-subg`, `--decouple`, `--group-weight-lr` |
| **Evaluation** | `--test-set-id`, `--seed`, `--last-eval-only`, `--no-best-th`, `--save-model` |

See `argument.py` for the full list.

## Scripts

- `scripts/run_DEAL.sh` – example runs for DEAL
- `scripts/run_deal_arch_abl.sh` – architecture ablations (EfficientNet, ViT-B, 2D sagittal) for `ad_transition`
- `scripts/run_baselines.sh` – baseline configurations

## Outputs

- **Checkpoints**: `./trained_models/<date>/mri/<target>/`
- **Results/logs**: `./results/<date>/mri/<target>/`

## License

See repository for license terms.
