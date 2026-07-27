# Pytorch-UNet

This repository keeps the original plain U-Net architecture and adds reproducible training, volume/frame accuracy evaluation, and real-sample benchmarking for Synapse, ACDC, and Cataract1k. The original Carvana loader and `predict.py` workflow remain available.

## Installation

Install a PyTorch build appropriate for the machine, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The medical workflows use SciPy/HDF5/OpenCV/Pandas for data, MedPy for Dice/Jaccard/HD95, SimpleITK for optional volume output, nibabel as an optional ACDC spacing source, and the benchmark utilities listed in `requirements.txt`. Weights & Biases is opt-in: training never imports or contacts it unless `--wandb` is supplied.

## Dataset contracts

The central registry in `utils/dataset_registry.py` is shared by training and testing. CLI path values override these defaults.

| Dataset | Default training root | Default validation/test root | Input | Classes | Validation/test protocol |
| --- | --- | --- | --- | --- | --- |
| Synapse | `/data/halyusuf/data/Synapse/train_npz/` | `/data/halyusuf/data/Synapse/test_vol_h5` | 1-channel grayscale | 9 | reconstructed 3-D volumes |
| ACDC | `/data/halyusuf/data/ACDC` | same | 1-channel grayscale | 4 | reconstructed 3-D volumes with physical spacing |
| Cataract1k | `/data/halyusuf/data/CataractData/` | same | 3-channel RGB | 5 | GT-present foreground classes per frame |

`Catrakt1k` is accepted as a legacy CLI alias and is immediately canonicalized to `Cataract1k` in logs, checkpoints, output paths, protocol names, and JSON.

Class order is fixed:

- Synapse: Background, Aorta, Gallbladder, Kidney(L), Kidney(R), Liver, Pancreas, Spleen, Stomach.
- ACDC: Background, Right Ventricle, Myocardium, Left Ventricle.
- Cataract1k: Background, Pupil, Cornea, Lens, Instruments. `Pupil`/`pupil1`, `Cornea`/`cornea1`, `Lens`, `Instruments`, and the supplied surgical-instrument titles are mapped by `datasets/dataset_cataract.py`.

### Expected layouts

```text
/data/halyusuf/data/Synapse/
├── train_npz/
│   └── case####_slice###.npz
└── test_vol_h5/
    └── case####.npy.h5

./lists/lists_Synapse/
├── train.txt
└── test_vol.txt
```

The Synapse split lists are included in this repository. A custom `--list-dir` must contain both expected files.

```text
/data/halyusuf/data/ACDC/
├── ACDC_training_slices/
│   └── patient###_*.h5
└── ACDC_training_volumes/
    └── patient###_*.h5
```

The supplied active ACDC split is intentionally preserved: patients `021..100` train, while patients `001..020` are used for both validation and testing. This is not five-fold cross-validation. `--fold-id` remains compatibility-only and does not change the split.

```text
/data/halyusuf/data/CataractData/
├── train.csv                 # column: imgs
├── test.csv                  # column: imgs
├── img/
│   └── frame.png
└── ann/
    └── frame.png.json
```

Training and validation resize to the fixed `--img-size` (default 224). Cataract1k uses ImageNet mean/std normalization during both. Accuracy testing intentionally loads raw HWC RGB frames; the inference helper performs cubic RGB resize, optional ImageNet normalization, model inference, and nearest-neighbor resizing of the integer prediction to the original mask size.

## Training

Medical training uses the datasets' explicit splits—never `random_split`. The unchanged U-Net is constructed from the registry with 1 input channel for Synapse/ACDC and true 3-channel RGB for Cataract1k. The objective remains:

```text
ce_weight * cross_entropy + dice_weight * dice_loss
```

Both weights default to `1.0`. RMSprop, `ReduceLROnPlateau(mode="max")`, AMP, gradient clipping, and the original transposed-convolution default are retained. Python, NumPy, PyTorch, CUDA, and DataLoader workers are seeded; deterministic behavior defaults on and can be disabled with `--no-deterministic`.

Training starts from random initialization because no U-Net pretrained checkpoint was supplied. To initialize from user-provided compatible weights, use `--init-checkpoint PATH` (also `--pretrained-checkpoint`). Use `--resume PATH` only for strict continuation; it requires and restores the epoch, optimizer, scheduler, AMP scaler, Python/NumPy/PyTorch/CUDA RNG, and DataLoader-generator states. A weights-only or incomplete checkpoint must be supplied through `--init-checkpoint` instead. The two modes cannot be combined. Explicitly supplied missing paths fail immediately.

Each output directory contains:

- `best_model.pth`, selected by the dataset's foreground validation mean Dice protocol;
- `last_model.pth`, including the full continuation state;
- `checkpoint_epoch_N.pth` when `--save-every N` is configured.

Structured checkpoints record dataset, channels, classes, U-Net upsampling mode, image size, class names, normalization, arguments, epoch, and best Dice. Loading also supports raw state dictionaries, legacy `mask_values`, and `module.` prefixes. Strict test/resume loading rejects metadata or tensor incompatibility. `--allow-partial-init` is initialization-only and reports every loaded, missing, unexpected, and shape-incompatible key.

Weights & Biases is disabled by default. Add `--wandb` only when desired.

### Synapse

```bash
python train.py \
  --dataset Synapse \
  --img-size 224 \
  --epochs 150 \
  --batch-size 24 \
  --learning-rate 1e-5 \
  --checkpoint-dir outputs/unet/synapse \
  --amp

python test.py \
  --dataset Synapse \
  --checkpoint outputs/unet/synapse/best_model.pth \
  --img-size 224 \
  --benchmark-dir benchmark
```

### ACDC

```bash
python train.py \
  --dataset ACDC \
  --img-size 224 \
  --epochs 150 \
  --batch-size 24 \
  --learning-rate 1e-5 \
  --checkpoint-dir outputs/unet/acdc \
  --amp

python test.py \
  --dataset ACDC \
  --checkpoint outputs/unet/acdc/best_model.pth \
  --img-size 224 \
  --acdc-zspacing 5.0 \
  --benchmark-dir benchmark
```

### Cataract1k

```bash
python train.py \
  --dataset Cataract1k \
  --img-size 224 \
  --epochs 150 \
  --batch-size 24 \
  --learning-rate 1e-5 \
  --checkpoint-dir outputs/unet/cataract1k \
  --amp

python test.py \
  --dataset Cataract1k \
  --checkpoint outputs/unet/cataract1k/best_model.pth \
  --img-size 224 \
  --normalize-present-class-eval \
  --benchmark-dir benchmark
```

Override defaults with `--root-path`, `--volume-path`, and `--list-dir`. No fictional pretrained path is used in these commands.

## Accuracy protocols

All reported accuracy excludes background. Per-class masks use MedPy `dc`, `jc`, and `hd95` exactly. When prediction and ground truth are both present, those values are returned directly. When exactly one is empty, Dice and IoU are zero and HD95 is the maximum physical diagonal `norm((shape - 1) * voxelspacing)`. When both are empty, Synapse/ACDC receive Dice/IoU 1 with undefined (`NaN`) HD95; the no-absent-reward diagnostic returns all `NaN`. Undefined HD95 is never changed to zero.

Synapse and ACDC are reconstructed at original in-plane resolution one slice at a time. Image slices use cubic interpolation into the fixed model input; integer predictions use nearest-neighbor interpolation back. Metrics form `[case, foreground class, metric]`, are nan-averaged over cases first, then over classes. Synapse uses unit/unspecified metric spacing. ACDC first consumes dataset spacing metadata, normalizes it to positive `zyx`, and logs its source per case. If absent, `--acdc-zspacing` constructs `(z, 1, 1)`.

Cataract1k uses the official `Cataract1k_frame_present_background_excluded` headline: determine the foreground classes present in each GT frame, average only those classes within the frame, then nan-average the evaluated frame means. A completely missed present class gets zero overlap and the 2-D diagonal HD95 penalty. GT-absent classes cannot improve the headline; predicted absent classes are counted separately. Background-only frames are discounted. Per-class and case-prefix-grouped values are diagnostics, not the headline aggregation.

Optional `--save-predictions [PATH]` writes Synapse/ACDC image, label, and prediction volumes through SimpleITK, or Cataract1k PNG prediction masks. Saving is off by default and does not change metric inputs or benchmark timing.

## System benchmark and JSON

Accuracy inference runs first. The same real test split then supplies fixed-size benchmark tensors: a deterministic center slice for Synapse/ACDC volumes and true RGB frames for Cataract1k. Channel count must equal `model.n_channels`; grayscale is never expanded to RGB. Loading, resizing, and CPU normalization occur before the timed forward pass.

Defaults are batch size 36, 20 warmups, 50 measured batches, 50 single-image warmups, 1,000 single-image latency samples, cuDNN fixed-input benchmarking enabled, benchmark AMP off, and one repeated run (valid range 1–10). Results include throughput, mean and p50/p90/p95/p99 latency, exact and million-scale parameter counts, model size, CUDA runtime allocator memory when available, input shape/channels, and repeated-run arrays/means/population standard deviations. CPU runtime memory is explicitly marked unavailable rather than reported as zero.

The fallback U-Net FLOP counter includes `Conv2d`, `ConvTranspose2d`, and any `Linear` layers; pooling, normalization, activation, concatenation, padding, and interpolation are omitted. The convention is applied once: `FLOPs = 2 * MACs`.

Accuracy and system statistics are printed and saved as one `BenchmarkResults` object under:

```text
benchmark/<canonical_dataset>/<checkpoint_stem>_<img_size>_<YYYYMMDD_HHMMSS>.json
```

The JSON has exactly `metrics` and `notes` at the top level. NumPy/PyTorch/path/dataclass values are converted recursively, and non-finite values become JSON `null` while remaining `NaN` during in-memory aggregation.

`--verbose` or `--max-test-cases N` enters clearly logged partial smoke-test mode and skips the system benchmark. `--skip-system-benchmark` is also available for accuracy-only diagnostics; these runs are not official combined benchmarks.

## Legacy Carvana prediction

Carvana training remains available with `python train.py --dataset Carvana`; its image/mask defaults remain `data/imgs` and `data/masks`. Existing single-image prediction behavior is unchanged:

```bash
python predict.py --input image.jpg --output mask.png
```

The original Carvana release checkpoint is not downloaded or reused by the medical commands.

## Verification

Focused tests cover model channel/output shapes, robust logits extraction, metric edge cases and aggregation order, Cataract1k present-class behavior, ACDC spacing, benchmark collation/parameters/FLOPs/JSON, checkpoint compatibility, and registry defaults/aliases. Run:

```bash
python -m compileall train.py test.py unet datasets utils tests
python train.py --help
python test.py --help
pytest -q
```

The focused checks for this revision were run with Python 3.10 and PyTorch 2.3.0+cu121 on a CPU-only execution host. CUDA accuracy timing and CUDA allocator-memory measurement were therefore not exercised in that environment.

No training, accuracy, throughput, latency, FLOP, memory, or model-size numbers are embedded in this README; generate them from an actual checkpoint and real test split.
