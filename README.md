# From CRFOR to CRFOLD: Adaptive and Fast LiDAR Snow Removal

CRFOLD extends the spatio-temporal Conditional Random Field (CRFOR) framework for LiDAR snow removal with three improvements:

- **Dynamic parameters** — a lightweight Tiny-MLP predicts per-frame joint-feature constants from global scene statistics.
- **Long-range weighting** — distance and intensity terms are adjusted for far-range points to reduce false removal of sparse objects.
- **Voxel planarity** — irregularity is computed on voxels instead of per-point PCA, reducing runtime while preserving accuracy.

## Requirements

- numpy
- scikit-learn

## Dataset layout

Place WADS-style sequences under `sequences/`:

```
sequences/
  <sequence_id>/
    velodyne/*.bin
    labels/*.label
```

Each `.bin` file stores `N×4` float32 point clouds (x, y, z, intensity). Labels use `110` for active falling snow.

## Evaluate

Single-frame CRFOLD:

```bash
python CRFOLD.py
```

Spatio-temporal CRFOLD (uses the previous 2 frames after the sequence warm-up):

```bash
python CRFOLD_t.py
```

Both scripts scan all sequences under `sequences/`, evaluate up to 50 frames per sequence, and write JSON metrics:

- `result_CRFOLD.json` — single-frame
- `result_CRFOLD_t.json` — temporal

Metrics include average precision, recall, F1, and per-frame runtime.

## Tuned parameters (single-frame)

Hyperparameters were tuned on WADS-style evaluation with **temporal mode off** (`CRFOLD.py`). Defaults in `CRFOLD.py` and `CRFOLD_t.py` now use the tuned values.

| Parameter        | Original | Tuned |
| ---------------- | -------- | ----- |
| `knn_num`        | 8        | 10    |
| `low_threshold`  | -0.4     | -0.55 |
| `high_threshold` | 0.6      | 0.68  |
