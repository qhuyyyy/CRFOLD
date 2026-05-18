import glob
import json
import os
from time import time

from CRFOLD import (
    _crfold_prepare,
    _crfold_refine_single,
    _crfold_refine_temporal,
    compute_metrics,
    read_from_WADS,
)


def CRFOLD_basic_desnow(
    pc_data,
    knn_num=8,
    low_threshold=-0.4,
    high_threshold=0.6,
    intensity_const=1,
    nbs_mean_d_const=5,
    pca_const=1,
    intensity_threshold_constant=2,
    snow_detection_range=30,
    ground_removal=-1.8,
    voxel_size=0.45,
):
    """First frames in a temporal sequence (no prior-frame context). Returns (snows_idx_dict, kd_tree)."""
    Features, excluded, Dists, Indexes, kd_tree, knn_num, lt, ht, _pts = _crfold_prepare(
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
    snows = _crfold_refine_single(Features, excluded, Dists, Indexes, knn_num, lt, ht)
    return snows, kd_tree


def CRFOLD_temporal_desnow(
    pc_data,
    pre_info,
    m,
    knn_num=8,
    low_threshold=-0.4,
    high_threshold=0.6,
    intensity_const=1,
    nbs_mean_d_const=5,
    pca_const=1,
    intensity_threshold_constant=2,
    snow_detection_range=30,
    ground_removal=-1.8,
    voxel_size=0.45,
):
    """Temporal CRFOLD using the previous ``m`` frames. Returns (snows_idx_dict, kd_tree)."""
    Features, excluded, Dists, Indexes, kd_tree, knn_num, lt, ht, pts = _crfold_prepare(
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
    snows = _crfold_refine_temporal(
        Features, excluded, Dists, Indexes, knn_num, lt, ht, pts, pre_info, m
    )
    return snows, kd_tree


def run_evaluation(m=2, output_path="result_CRFOLD_t.json", max_frames_per_sequence=50):
    """Evaluate spatio-temporal CRFOLD on WADS-style sequences under ``sequences/``."""
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

        valid_pairs = []
        for data_path in bin_files:
            file_name = os.path.basename(data_path).replace(".bin", "")
            label_path = os.path.join(label_dir, file_name + ".label")
            if not os.path.exists(label_path):
                print(f"Warning: Label not found for {sequence_name}/{file_name}")
                continue
            valid_pairs.append((data_path, label_path))

        if not valid_pairs:
            print(f"Sequence {sequence_name}: no valid frames with labels.")
            continue

        pre_info = [{"data": None, "snows": None, "kdtree": None} for _ in range(m)]
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_time = 0.0
        frame_count = 0

        print(f"\n=== Processing sequence {sequence_name} ===")
        for idx, (data_path, label_path) in enumerate(valid_pairs):
            data, label = read_from_WADS(data_path, label_path, duplicated_removal)
            t0 = time()
            if idx < m:
                snows_idx_dict, kdtree = CRFOLD_basic_desnow(data)
            else:
                snows_idx_dict, kdtree = CRFOLD_temporal_desnow(data, pre_info, m)
            total_time += time() - t0

            precision, recall, f1 = compute_metrics(label, snows_idx_dict)
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            frame_count += 1

            if idx < m:
                pre_info[idx] = {"data": data, "snows": snows_idx_dict, "kdtree": kdtree}
            else:
                for dummy_idx in range(m - 1):
                    pre_info[dummy_idx] = pre_info[dummy_idx + 1]
                pre_info[m - 1] = {"data": data, "snows": snows_idx_dict, "kdtree": kdtree}

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
