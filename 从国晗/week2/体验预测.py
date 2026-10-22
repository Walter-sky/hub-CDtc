"""加载最近一次训练的模型，预测测试集中的一张图片。"""

import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
# 表示测试集中的位置，不是原始数据集编号。可以改成 1、2、3 等。
TEST_POSITION = 0


def main():
    latest_path = ROOT / "outputs" / "latest_run.txt"
    if not latest_path.exists():
        raise RuntimeError("请先运行 作业01.py，生成训练结果。")
    run_dir = ROOT / "outputs" / latest_path.read_text(encoding="utf-8").strip()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    selected = summary["experiments"][summary["selected_experiment_index"]]
    spec = importlib.util.spec_from_file_location("digits_training", ROOT / "作业01.py")
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    checkpoint = torch.load(run_dir / selected["checkpoint_file"], map_location="cpu", weights_only=True)
    model = training.build_model(checkpoint["config"]["hidden_sizes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with np.load(run_dir / "data_and_splits.npz", allow_pickle=False) as snapshot:
        test_indices = snapshot["test_indices"]
        if not 0 <= TEST_POSITION < len(test_indices):
            raise ValueError(f"TEST_POSITION 必须在 0～{len(test_indices) - 1} 之间。")
        dataset_index = int(test_indices[TEST_POSITION])
        raw_features = snapshot["raw_features"][dataset_index]
        true_label = int(snapshot["labels"][dataset_index])
    features = torch.from_numpy(raw_features / 16.0).unsqueeze(0)
    with torch.no_grad():
        scores = model(features)
        probabilities = torch.softmax(scores, dim=1)[0]
        prediction = int(scores.argmax(dim=1).item())
    print("模型来自：", run_dir)
    print("原始数据编号：", dataset_index)
    print("正确答案：", true_label)
    print("模型预测：", prediction)
    print("预测正确：", prediction == true_label)
    print("各类别的模型概率（未经校准）：")
    for label, probability in enumerate(probabilities.tolist()):
        print(f"  数字 {label}：{probability:.2%}")
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(raw_features.reshape(8, 8), cmap="gray", vmin=0, vmax=16)
    axes[0].set_title(f"正确答案 {true_label}，模型预测 {prediction}")
    axes[0].axis("off")
    axes[1].bar(range(10), probabilities.numpy(), color="#2563eb")
    axes[1].set(xticks=range(10), ylim=(0, 1), xlabel="数字类别", ylabel="模型概率（未经校准）")
    figure.tight_layout()
    destination = run_dir / "single_prediction.png"
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    print("预测图片：", destination)


if __name__ == "__main__":
    main()
