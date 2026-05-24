import glob
import json
import math
import os
from time import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree

_TINY_MLP_SEED = 42
_MLP_LR_IC_BLEND = 0.28
_MLP_LR_IC_SCALE = 0.55

_LR_T_ONSET = 0.50
_LR_D_ALPHA = 0.30
_LR_D_POWER = 1.12
_LR_I_SCALE = 0.050
_LR_I_THRESH = 0.37
_LR_BLEND_MAX = 0.40
_LR_BLEND_EXP = 2.0


def read_from_WADS(data_path="", label_path="", duplicated_removal=True):
    data = np.fromfile(data_path, np.float32).reshape([-1, 4])
    label = np.fromfile(label_path, np.uint32).reshape([-1])
    label = np.where(label == 110, 1, 0)
    if duplicated_removal:
        data, idx_unique = np.unique(data, axis=0, return_index=True)
        label = label[idx_unique]
    return data, label


def compute_metrics(pc_label, snows):
    """Return precision, recall, and F1 for snow detection."""
    tp = tn = fp = fn = 0
    for j in range(len(pc_label)):
        if pc_label[j] == 1:
            if j in snows:
                tp += 1
            else:
                fn += 1
        elif j in snows:
            fp += 1
        else:
            tn += 1
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1_score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score


def _long_range_di_weights(r, intensity, snow_detection_range, dens_ratio):
    """Per-point dynamic weighting of neighbor-distance and intensity in the joint feature."""
    t = min(1.0, max(0.0, r / max(float(snow_detection_range), 1e-3)))
    if t <= _LR_T_ONSET:
        return dens_ratio, 0.0

    u = (t - _LR_T_ONSET) / max(1.0 - _LR_T_ONSET, 1e-6)
    blend = _LR_BLEND_MAX * (u ** _LR_BLEND_EXP)
    density_w = 1.0 / (1.0 + _LR_D_ALPHA * (u ** _LR_D_POWER))
    intensity_bonus = _LR_I_SCALE * u * max(0.0, intensity - _LR_I_THRESH)
    dens_weighted = density_w * dens_ratio
    dens_term = (1.0 - blend) * dens_ratio + blend * dens_weighted
    return dens_term, intensity_bonus


def _global_scene_stats(pc_data, excluded):
    """Global statistics for Tiny-MLP: mean intensity, std intensity, mean range, log-density proxy."""
    active = ~excluded
    n = int(np.sum(active))
    if n == 0:
        return np.zeros(4, dtype=np.float64)
    pts = pc_data[active, :3]
    intens = pc_data[active, 3]
    ranges = np.linalg.norm(pts, axis=1)
    r95 = float(np.percentile(ranges, 95)) + 1e-6
    vol = (4.0 / 3.0) * math.pi * (r95 ** 3)
    density = n / vol
    return np.array(
        [np.mean(intens), np.std(intens), np.mean(ranges), math.log1p(density)],
        dtype=np.float64,
    )


def _tiny_mlp_predict_params(stats, base_i, base_n, base_p, mlp_scale=1.0):
    """Train-free Tiny-MLP (numpy): outputs per-frame scales for (a, b, c) joint-feature constants."""
    rng = np.random.RandomState(_TINY_MLP_SEED)
    W1 = rng.randn(8, 4) * 0.15
    b1 = rng.randn(8) * 0.05
    W2 = rng.randn(3, 8) * 0.15
    b2 = np.zeros(3)
    x = stats.copy()
    x[0] = (x[0] - 0.5) / 2.0
    x[1] = (x[1] - 0.5) / 2.0
    x[2] = (x[2] - 15.0) / 15.0
    x[3] = (x[3] - 5.0) / 5.0
    h = np.tanh(W1 @ x + b1)
    out = np.tanh(W2 @ h + b2)
    s = float(mlp_scale)
    intensity_const = float(max(0.25, base_i * (1.0 + s * 0.25 * out[0])))
    nbs_mean_d_const = float(max(1.0, base_n * (1.0 + s * 0.2 * out[1])))
    pca_const = float(max(0.15, base_p * (1.0 + s * 0.15 * out[2])))
    return intensity_const, nbs_mean_d_const, pca_const


def _irregularity_pca(pca_model, neighbor_pts):
    pca_model.fit(neighbor_pts)
    return float(pca_model.explained_variance_ratio_[2])


def _irregularity_eigh(neighbor_pts):
    c = neighbor_pts - np.mean(neighbor_pts, axis=0, keepdims=True)
    cov = (c.T @ c) / max(len(neighbor_pts), 1)
    w = np.linalg.eigh(cov)[0]
    s = np.sum(w) + 1e-12
    return float(w[0] / s)


def _build_voxel_irregularity(pts, excluded, voxel_size=0.45):
    """One lightweight eigh per occupied voxel; maps point index -> irregularity."""
    length = len(pts)
    irreg = np.full(length, -1.0, dtype=np.float64)
    buckets = {}
    for i in range(length):
        if excluded[i]:
            continue
        p = pts[i]
        key = (
            int(math.floor(p[0] / voxel_size)),
            int(math.floor(p[1] / voxel_size)),
            int(math.floor(p[2] / voxel_size)),
        )
        buckets.setdefault(key, []).append(i)
    pca_small = PCA(n_components=3)
    for _key, idxs in buckets.items():
        if len(idxs) < 3:
            continue
        block = pts[idxs]
        if len(idxs) <= 8:
            irreg_local = _irregularity_eigh(block)
        else:
            irreg_local = _irregularity_pca(pca_small, block)
        for ii in idxs:
            irreg[ii] = irreg_local
    return irreg


def _crfold_prepare(
    pc_data,
    knn_num=10,
    low_threshold=-0.55,
    high_threshold=0.68,
    intensity_const=1,
    nbs_mean_d_const=5,
    pca_const=1,
    intensity_threshold_constant=2,
    snow_detection_range=30,
    ground_removal=-1.8,
    voxel_size=0.45,
):
    """Joint-feature construction + normalization. Used by single-frame and temporal wrappers."""
    pts = pc_data[:, :3]
    length = len(pc_data)
    point_ranges = np.linalg.norm(pts, axis=1)

    excluded = pc_data[:, 3] > intensity_threshold_constant
    excluded |= point_ranges > snow_detection_range
    if ground_removal is not None:
        excluded |= pc_data[:, 2] < ground_removal

    stats = _global_scene_stats(pc_data, excluded)
    ic_m, _, _ = _tiny_mlp_predict_params(
        stats, intensity_const, nbs_mean_d_const, pca_const, mlp_scale=_MLP_LR_IC_SCALE
    )
    ic = float((1.0 - _MLP_LR_IC_BLEND) * intensity_const + _MLP_LR_IC_BLEND * ic_m)
    ic = float(np.clip(ic, 0.92, 1.15))
    ndc, pcc = nbs_mean_d_const, pca_const

    pca = PCA(n_components=3)
    kd_tree = KDTree(pts)
    Dists, Indexes = kd_tree.query(pts.reshape(-1, 3), k=knn_num)
    voxel_irreg = _build_voxel_irregularity(pts, excluded, voxel_size=voxel_size)

    Features = np.full([length], -1, dtype=np.float32)
    norm_min, norm_max = 1024.0, 0.0

    for i in range(length):
        if excluded[i]:
            continue
        dists, indexes = Dists[i], Indexes[i]
        neighbors_mean_dist = float(np.mean(dists))
        intensity = float(pc_data[i][3])
        r = float(point_ranges[i])

        if voxel_irreg[i] >= 0.0:
            pca_3rd_dim = float(voxel_irreg[i])
        else:
            pca_3rd_dim = _irregularity_pca(pca, pts[indexes])

        dens_ratio = neighbors_mean_dist / (0.1 + r)
        norm_min = min((pcc + pca_3rd_dim) * (ndc + dens_ratio) / (ic + intensity), norm_min)
        norm_max = max((pcc + pca_3rd_dim) * (ndc + dens_ratio) / (ic + intensity), norm_max)

        dens_term, intensity_bonus = _long_range_di_weights(
            r, intensity, snow_detection_range, dens_ratio
        )
        denom = ic + intensity + intensity_bonus
        Features[i] = (pcc + pca_3rd_dim) * (ndc + dens_term) / denom

    span = norm_max - norm_min
    if span < 1e-12:
        span = 1.0
    active = Features >= 0
    Features[active] = 2.0 * (Features[active] - norm_min) / span - 1.0

    return Features, excluded, Dists, Indexes, kd_tree, knn_num, low_threshold, high_threshold, pts


def _crfold_refine_single(Features, excluded, Dists, Indexes, knn_num, low_threshold, high_threshold):
    snows_idx_dict = dict()
    Features = Features.copy()
    for i in range(len(Features)):
        if not excluded[i]:
            if Features[i] > high_threshold:
                snows_idx_dict[i] = "f"
            elif Features[i] > low_threshold:
                s = Features[i]
                dists, indexes = Dists[i], Indexes[i]
                for j in range(1, knn_num):
                    f = Features[indexes[j]]
                    if f > high_threshold:
                        f = 2
                    elif f < low_threshold:
                        f = -2
                    s += f / (0.01 + dists[j])
                if s > 0:
                    Features[i] = 2
                    snows_idx_dict[i] = "f"
                else:
                    Features[i] = -2
    return snows_idx_dict


def _crfold_refine_temporal(
    Features, excluded, Dists, Indexes, knn_num, low_threshold, high_threshold, pts, pre_info, m
):
    snows_idx_dict = dict()
    Features = Features.copy()
    for i in range(len(Features)):
        if not excluded[i]:
            if Features[i] > high_threshold:
                snows_idx_dict[i] = "f"
            elif Features[i] > low_threshold:
                s = Features[i]
                dists, indexes = Dists[i], Indexes[i]
                for j in range(1, knn_num):
                    f = Features[indexes[j]]
                    if f > high_threshold:
                        f = 2
                    elif f < low_threshold:
                        f = -2
                    s += f / (0.01 + dists[j])
                for pre_frame in range(m):
                    pre_dists, pre_indexes = pre_info[pre_frame]["kdtree"].query(
                        pts[i].reshape(1, -1), k=knn_num
                    )
                    for pre_idx in range(1, knn_num):
                        if pre_indexes[0][pre_idx] in pre_info[pre_frame]["snows"].keys():
                            s += (0.5 ** (m - pre_frame - 1)) / (0.01 + pre_dists[0][pre_idx])
                        else:
                            s -= (0.5 ** (m - pre_frame - 1)) / (0.01 + pre_dists[0][pre_idx])
                if s > 0:
                    Features[i] = 2
                    snows_idx_dict[i] = "f"
                else:
                    Features[i] = -2
    return snows_idx_dict


def CRFOLD_desnow(
    pc_data,
    knn_num=10,
    low_threshold=-0.55,
    high_threshold=0.68,
    intensity_const=1,
    nbs_mean_d_const=5,
    pca_const=1,
    intensity_threshold_constant=2,
    snow_detection_range=30,
    ground_removal=-1.8,
    voxel_size=0.45,
):
    """Single-frame CRFOLD snow removal."""
    Features, excluded, Dists, Indexes, _kd_tree, knn_num, lt, ht, _pts = _crfold_prepare(
        pc_data,
        knn_num=knn_num,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        intensity_const=intensity_const,
        nbs_mean_d_const=nbs_mean_d_const,
        pca_const=pca_const,
        intensity_threshold_constant=intensity_threshold_constant,
        snow_detection_range=snow_detection_range,
        ground_removal=ground_removal,
        voxel_size=voxel_size,
    )
    return _crfold_refine_single(Features, excluded, Dists, Indexes, knn_num, lt, ht)


def run_evaluation(output_path="result_CRFOLD.json", max_frames_per_sequence=50):
    """Evaluate CRFOLD on WADS-style sequences under ``sequences/``."""
    duplicated_removal = True
    sequences_root = "sequences"
    sequence_dirs = sorted(
        path for path in glob.glob(os.path.join(sequences_root, "*")) if os.path.isdir(path)
    )
    if not sequence_dirs:
        print("No sequence directories found.")
        return None

    results = {"sequences": {}, "overall": {}}
    overall_total_precision = 0.0
    overall_total_recall = 0.0
    overall_total_f1 = 0.0
    overall_total_time = 0.0
    overall_total_frames = 0
    processed_sequences = 0

    for sequence_dir in sequence_dirs:
        sequence_name = os.path.basename(sequence_dir)
        velodyne_dir = os.path.join(sequence_dir, "velodyne")
        label_dir = os.path.join(sequence_dir, "labels")
        if not os.path.isdir(velodyne_dir) or not os.path.isdir(label_dir):
            print(f"Skipping sequence {sequence_name}: missing velodyne/labels directory.")
            continue

        bin_files = sorted(glob.glob(os.path.join(velodyne_dir, "*.bin")))
        if not bin_files:
            print(f"Skipping sequence {sequence_name}: no .bin files found.")
            continue
        bin_files = bin_files[:max_frames_per_sequence]

        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_time = 0.0
        frame_count = 0

        print(f"\n=== Processing sequence {sequence_name} ===")
        for data_path in bin_files:
            file_name = os.path.basename(data_path).replace(".bin", "")
            label_path = os.path.join(label_dir, file_name + ".label")
            if not os.path.exists(label_path):
                print(f"Warning: Label not found for {sequence_name}/{file_name}")
                continue

            data, label = read_from_WADS(data_path, label_path, duplicated_removal)
            t0 = time()
            snows_idx_dict = CRFOLD_desnow(data)
            total_time += time() - t0

            precision, recall, f1 = compute_metrics(label, snows_idx_dict)
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            frame_count += 1

        if frame_count == 0:
            print(f"Sequence {sequence_name}: no valid frames with labels.")
            continue

        seq_result = {
            "avg_precision": total_precision / frame_count,
            "avg_recall": total_recall / frame_count,
            "avg_f1": total_f1 / frame_count,
            "avg_run_time_per_frame_s": total_time / frame_count,
            "frames_processed": frame_count,
        }
        results["sequences"][sequence_name] = seq_result
        print(f"Sequence {sequence_name} results: {seq_result}")

        overall_total_precision += total_precision
        overall_total_recall += total_recall
        overall_total_f1 += total_f1
        overall_total_time += total_time
        overall_total_frames += frame_count
        processed_sequences += 1

    if overall_total_frames == 0:
        print("No frames with labels found across all sequences.")
        return None

    results["overall"] = {
        "avg_precision": overall_total_precision / overall_total_frames,
        "avg_recall": overall_total_recall / overall_total_frames,
        "avg_f1": overall_total_f1 / overall_total_frames,
        "avg_run_time_per_frame_s": overall_total_time / overall_total_frames,
        "total_frames_processed": overall_total_frames,
        "sequences_processed": processed_sequences,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Overall results ===")
    print(results["overall"])
    print(f"Saved all sequence results to {output_path}")
    return results


if __name__ == "__main__":
    run_evaluation()
