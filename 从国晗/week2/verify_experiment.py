"""验收已保存的实验：不重新训练、不调参，只核对权重、划分和记录。"""

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
import torch

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    latest = (ROOT / "outputs" / "latest_run.txt").read_text(encoding="utf-8").strip()
    run_dir = ROOT / "outputs" / (sys.argv[1] if len(sys.argv) > 1 else latest)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("digits_training", ROOT / "作业01.py")
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    with np.load(run_dir / "data_and_splits.npz", allow_pickle=False) as snapshot:
        features = snapshot["raw_features"]
        labels = snapshot["labels"]
        partitions = {name: snapshot[key] for name, key in [
            ("train", "train_indices"), ("validation", "val_indices"), ("test", "test_indices")]}
    assert features.shape == (summary["dataset_size"], 64)
    assert hashlib.sha256(features.tobytes() + labels.tobytes()).hexdigest() == summary["dataset_sha256"]
    joined = np.concatenate(list(partitions.values()))
    np.testing.assert_array_equal(np.sort(joined), np.arange(len(labels)))
    for name, indices in partitions.items():
        assert len(indices) == summary["split_sizes"][name]
        assert np.bincount(labels[indices], minlength=10).tolist() == summary["class_counts"][name]
    for experiment in summary["experiments"]:
        checkpoint = torch.load(run_dir / experiment["checkpoint_file"], map_location="cpu", weights_only=True)
        model = training.build_model(checkpoint["config"]["hidden_sizes"])
        model.load_state_dict(checkpoint["state_dict"])
        with (run_dir / experiment["history_file"]).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == experiment["config"]["epochs"]
        best = min(rows, key=lambda row: float(row["val_loss"]))
        assert int(best["epoch"]) == checkpoint["best_epoch"] == experiment["best_epoch"]
        val_indices = partitions["validation"]
        val_loss, _, _ = training.evaluate(model, torch.from_numpy(features[val_indices] / 16.0),
                                           torch.from_numpy(labels[val_indices]), torch.nn.CrossEntropyLoss())
        assert abs(val_loss - float(best["val_loss"])) < 1e-6
    selected_index = min(range(len(summary["experiments"])),
                         key=lambda index: summary["experiments"][index]["best_val_loss"])
    assert selected_index == summary["selected_experiment_index"]
    selected = summary["experiments"][selected_index]
    checkpoint = torch.load(run_dir / selected["checkpoint_file"], map_location="cpu", weights_only=True)
    model = training.build_model(checkpoint["config"]["hidden_sizes"])
    model.load_state_dict(checkpoint["state_dict"])
    indices = partitions["test"]
    loss, _, predictions = training.evaluate(model, torch.from_numpy(features[indices] / 16.0),
                                             torch.from_numpy(labels[indices]), torch.nn.CrossEntropyLoss())
    with (run_dir / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    np.testing.assert_array_equal([int(row["dataset_index"]) for row in records], indices)
    np.testing.assert_array_equal([int(row["predicted_label"]) for row in records], predictions)
    np.testing.assert_array_equal([int(row["true_label"]) for row in records], labels[indices])
    assert int((predictions == labels[indices]).sum()) == summary["test"]["correct"]
    assert abs(loss - summary["test"]["loss"]) < 1e-6
    assert abs(float(np.mean(predictions == labels[indices])) - summary["test"]["accuracy"]) < 1e-12
    assert confusion_matrix(labels[indices], predictions, labels=list(range(10))).tolist() == summary["test"]["confusion_matrix"]
    for name in ["summary.json", "data_and_splits.npz", "test_predictions.csv"]:
        assert (run_dir / name).stat().st_size > 100
    print("验收通过：数据完整且划分不交叉；两组权重对应各自最佳验证轮次；")
    print("保存的最终模型逐条复现测试预测，准确率及混淆矩阵与报告一致。")
    print("结果：", run_dir)


if __name__ == "__main__":
    main()
